from gym import utils
from gym.envs.mujoco import mujoco_env
import numpy as np
import os

from modular.utils import *
from modular.wrappers import *
from metamorph.config import cfg


class ModularEnv(mujoco_env.MujocoEnv, utils.EzPickle):
    def __init__(self, xml):
        self.xml = xml
        
        self.full_limb_names = [
            'pelvis',
            'left_hip_pitch_link',
            'left_hip_roll_link', 
            'left_hip_yaw_link',
            'left_knee_link',
            'left_ankle_pitch_link',
            'left_ankle_roll_link',
            'right_hip_pitch_link',
            'right_hip_roll_link',
            'right_hip_yaw_link', 
            'right_knee_link',
            'right_ankle_pitch_link',
            'right_ankle_roll_link',
        ]
        
        # For multi-robot training with humanoids, we need a global naming scheme
        # Map G1 limbs to positions 0-12 in a 19-limb global array
        # (assuming MAX_LIMBS=19 to accommodate both G1 and humanoid variants)
        self.global_limb_names = [
            'pelvis',  # 0
            'left_hip_pitch_link',  # 1
            'left_hip_roll_link',  # 2
            'left_hip_yaw_link',  # 3
            'left_knee_link',  # 4
            'left_ankle_pitch_link',  # 5
            'left_ankle_roll_link',  # 6
            'right_hip_pitch_link',  # 7
            'right_hip_roll_link',  # 8
            'right_hip_yaw_link',  # 9
            'right_knee_link',  # 10
            'right_ankle_pitch_link',  # 11
            'right_ankle_roll_link',  # 12
            # Padding slots for other robot types
            'pad13', 'pad14', 'pad15', 'pad16', 'pad17', 'pad18',
        ]
        
        # Initialize placeholders (will be computed after MujocoEnv.__init__)
        self.edges = np.array([])
        self.traversals = np.array([])  # Must be numpy array, not Python list
        self.SWAT_RE = np.zeros([cfg.MODEL.MAX_LIMBS, cfg.MODEL.MAX_LIMBS, 3])
        self.context_features = np.array([])
        self._graph_built = False  # Flag to track if graph structure is ready
        
        # Call parent init (this will call reset_model which calls _get_obs)
        # MujocoEnv.__init__ will create self.metadata
        mujoco_env.MujocoEnv.__init__(self, xml, 4)
        utils.EzPickle.__init__(self)

        # Add our metadata after parent init
        # Unitree G1 12-DOF: 13 bodies, 12 joints
        self.metadata['num_limbs'] = 13
        self.metadata['num_joints'] = 12
        
        # agent_limb_names must be set after MujocoEnv.__init__ (when self.model exists)
        self.agent_limb_names = self.model.body_names[1:]
        
        # NOW build graph structure and context encoding (after sim is available)
        self._build_graph_structure()
        self._compute_context_encoding()
        
        # Mark that graph structure is ready
        self._graph_built = True

    def _build_graph_structure(self):
        """Build parent-child graph structure like UnimalEnv Agent module does"""
        # Get body indices (excluding world body at index 0)
        body_idxs = list(range(1, self.metadata['num_limbs'] + 1))
        
        # Build edges from joints
        # Skip the first joint (free joint for root body)
        # jnt_bodyid gives the body idx each joint is attached to (child)
        joint_to = self.sim.model.jnt_bodyid[1:].copy() - 1  # Skip free joint, subtract 1 for world body
        
        # Get parent body for each child
        body_parentids = self.sim.model.body_parentid.copy()
        joint_from = np.array([body_parentids[child + 1] - 1 for child in joint_to])
        
        # Verify we have the right number of joints
        assert len(joint_to) == self.metadata['num_joints'], \
            f"Expected {self.metadata['num_joints']} joints, got {len(joint_to)}"
        
        # Stack as [child, parent] pairs and flatten
        # This creates edge list: [child0, parent0, child1, parent1, ...]
        self.edges = np.vstack((joint_to, joint_from)).T.flatten().astype(np.int32)
        
        print(f"Built graph: {len(joint_to)} joints -> {len(self.edges)} edge values")
        
        # Build parent list for SWAT traversals
        # parents[i] = parent index of limb i, or -1 for root
        parents = [-1] * self.metadata['num_limbs']
        for i in range(len(joint_to)):
            child_idx = joint_to[i]
            parent_idx = joint_from[i]
            if 0 <= child_idx < len(parents) and parent_idx >= -1:
                parents[child_idx] = parent_idx
        
        # Generate SWAT traversals (depth-first traversal order)
        self.traversals = self._get_traversal(parents)
        
        # Generate SWAT relational encoding if needed
        if cfg.MODEL.TRANSFORMER.USE_SWAT_RE:
            self.SWAT_RE = self._get_graph_dict(parents)
        else:
            self.SWAT_RE = np.zeros([cfg.MODEL.MAX_LIMBS, cfg.MODEL.MAX_LIMBS, 3])

    def _get_traversal(self, parents):
        """Generate depth-first traversal order (SWAT traversal)"""
        # Find root (node with parent -1)
        root = parents.index(-1) if -1 in parents else 0
        
        # Build children map
        children = {i: [] for i in range(len(parents))}
        for i, parent in enumerate(parents):
            if parent >= 0:
                children[parent].append(i)
        
        # Depth-first traversal
        traversal = []
        
        def dfs(node):
            traversal.append(node)
            for child in sorted(children[node]):  # Sort for consistent ordering
                dfs(child)
        
        dfs(root)
        return np.array(traversal, dtype=np.int32)  # Return numpy array, not list

    def _get_graph_dict(self, parents):
        """Generate SWAT relational encoding (spatial relationships between nodes)"""
        n = len(parents)
        graph_dict = np.zeros([cfg.MODEL.MAX_LIMBS, cfg.MODEL.MAX_LIMBS, 3])
        
        # Build children map
        children = {i: [] for i in range(n)}
        for i, parent in enumerate(parents):
            if parent >= 0:
                children[parent].append(i)
        
        # Compute relationships
        for i in range(n):
            for j in range(n):
                if i == j:
                    graph_dict[i, j] = [1, 0, 0]  # Self
                elif parents[i] == j:
                    graph_dict[i, j] = [0, 1, 0]  # Parent
                elif j in children[i]:
                    graph_dict[i, j] = [0, 0, 1]  # Child
                # else: remains [0, 0, 0] for no direct relation
        
        return graph_dict

    def _compute_normalization_bounds(self):
        """Compute normalization bounds from the actual robot morphology"""
        num_limbs = self.metadata['num_limbs']
        num_joints = self.metadata['num_joints']
        body_idxs = list(range(1, num_limbs + 1))
        
        # Compute actual ranges from the robot
        limb_bounds = {}
        
        # Body position bounds
        body_pos = self.sim.model.body_pos[body_idxs, :]
        limb_bounds['body_pos'] = (body_pos.min(axis=0, keepdims=True), 
                                    body_pos.max(axis=0, keepdims=True))
        
        # Body ipos bounds
        body_ipos = self.sim.model.body_ipos[body_idxs, :]
        limb_bounds['body_ipos'] = (body_ipos.min(axis=0, keepdims=True), 
                                     body_ipos.max(axis=0, keepdims=True))
        
        # Quaternion bounds (typical range)
        limb_bounds['body_iquat'] = (np.array([[-1, -1, -1, -1]]), np.array([[1, 1, 1, 1]]))
        limb_bounds['geom_quat'] = (np.array([[-1, -1, -1, -1]]), np.array([[1, 1, 1, 1]]))
        
        # Body mass bounds
        body_mass = self.sim.model.body_mass[body_idxs]
        limb_bounds['body_mass'] = (np.array([[body_mass.min()]]), 
                                     np.array([[body_mass.max()]]))
        
        # Geom indices
        geom_idxs = []
        for body_idx in body_idxs:
            for geom_idx in range(self.sim.model.ngeom):
                if self.sim.model.geom_bodyid[geom_idx] == body_idx:
                    geom_idxs.append(geom_idx)
                    break
        
        if len(geom_idxs) == num_limbs:
            geom_size = self.sim.model.geom_size[geom_idxs, :2]
            limb_bounds['body_shape'] = (geom_size.min(axis=0, keepdims=True), 
                                         geom_size.max(axis=0, keepdims=True))
            geom_friction = self.sim.model.geom_friction[geom_idxs, 0:1]
            limb_bounds['body_friction'] = (geom_friction.min(axis=0, keepdims=True),
                                            geom_friction.max(axis=0, keepdims=True))
        else:
            limb_bounds['body_shape'] = (np.array([[0.01, 0.01]]), np.array([[0.2, 0.2]]))
            limb_bounds['body_friction'] = (np.array([[0.5]]), np.array([[1.5]]))
        
        # Joint bounds
        joint_bounds = {}
        
        jnt_pos = self.sim.model.jnt_pos[1:, :]
        joint_bounds['jnt_pos'] = (jnt_pos.min(axis=0, keepdims=True), 
                                   jnt_pos.max(axis=0, keepdims=True))
        
        joint_range = self.sim.model.jnt_range[1:, :]
        joint_bounds['joint_range'] = (joint_range.min(axis=0, keepdims=True),
                                       joint_range.max(axis=0, keepdims=True))
        
        joint_axis = self.sim.model.jnt_axis[1:, :]
        joint_bounds['joint_axis'] = (joint_axis.min(axis=0, keepdims=True),
                                      joint_axis.max(axis=0, keepdims=True))
        
        gear = self.sim.model.actuator_gear[:, 0:1]
        joint_bounds['gear'] = (gear.min(axis=0, keepdims=True),
                                gear.max(axis=0, keepdims=True))
        
        armature = self.sim.model.dof_armature[6:][:, np.newaxis]
        joint_bounds['armature'] = (armature.min(axis=0, keepdims=True),
                                    armature.max(axis=0, keepdims=True))
        
        damping = self.sim.model.dof_damping[6:][:, np.newaxis]
        joint_bounds['damping'] = (damping.min(axis=0, keepdims=True),
                                   damping.max(axis=0, keepdims=True))
        
        return limb_bounds, joint_bounds

    def _compute_context_encoding(self):
        """Compute normalized morphology context like UnimalEnv Agent module does"""
        num_limbs = self.metadata['num_limbs']
        num_joints = self.metadata['num_joints']
        
        # Get body indices (excluding world)
        body_idxs = list(range(1, num_limbs + 1))
        
        # Compute normalization bounds from actual robot
        limb_bounds, joint_bounds = self._compute_normalization_bounds()
        
        # Initialize context dictionary
        context_limb = {}
        context_joint = {}
        
        # ===== LIMB CONTEXT =====
        # Relative position and orientation
        context_limb['body_pos'] = self.sim.model.body_pos[body_idxs, :].copy()
        context_limb['body_ipos'] = self.sim.model.body_ipos[body_idxs, :].copy()
        context_limb['body_iquat'] = self.sim.model.body_iquat[body_idxs, :].copy()
        
        # Get geom indices for agent bodies
        geom_idxs = []
        for body_idx in body_idxs:
            for geom_idx in range(self.sim.model.ngeom):
                if self.sim.model.geom_bodyid[geom_idx] == body_idx:
                    geom_idxs.append(geom_idx)
                    break  # Only take first geom per body
        
        if len(geom_idxs) == num_limbs:
            context_limb['geom_quat'] = self.sim.model.geom_quat[geom_idxs, :].copy()
            context_limb['body_shape'] = self.sim.model.geom_size[geom_idxs, :2].copy()
            context_limb['body_friction'] = self.sim.model.geom_friction[geom_idxs, 0:1].copy()
        else:
            # Fallback if geom matching fails
            context_limb['geom_quat'] = np.tile([1, 0, 0, 0], (num_limbs, 1))
            context_limb['body_shape'] = np.zeros((num_limbs, 2))
            context_limb['body_friction'] = np.ones((num_limbs, 1))
        
        # Hardware properties
        context_limb['body_mass'] = self.sim.model.body_mass[body_idxs].copy()[:, np.newaxis]
        
        # Normalize limb context to [-1, 1] range using computed bounds
        for key in context_limb:
            if key in limb_bounds:
                lower, upper = limb_bounds[key]
                # Add small epsilon to avoid division by zero
                range_span = upper - lower + 1e-8
                # Normalize: -1 + 2 * (x - lower) / range
                context_limb[key] = -1.0 + 2.0 * (context_limb[key] - lower) / range_span
                # Clamp to [-1, 1] for safety
                context_limb[key] = np.clip(context_limb[key], -1.0, 1.0)
        
        # Select which observations to include based on config
        limb_features = self._select_context_obs(context_limb, cfg.MODEL.CONTEXT_OBS_TYPES)
        
        # ===== JOINT CONTEXT =====
        # Joint properties (skip first joint which is the free joint)
        context_joint['jnt_pos'] = self.sim.model.jnt_pos[1:, :].copy()
        context_joint['joint_range'] = self.sim.model.jnt_range[1:, :].copy()
        context_joint['joint_axis'] = self.sim.model.jnt_axis[1:, :].copy()
        context_joint['gear'] = self.sim.model.actuator_gear[:, 0:1].copy()
        
        # DOF properties (skip first 6 for free joint)
        context_joint['armature'] = self.sim.model.dof_armature[6:].copy()[:, np.newaxis]
        context_joint['damping'] = self.sim.model.dof_damping[6:].copy()[:, np.newaxis]
        
        # Normalize joint context to [-1, 1] range using computed bounds
        for key in context_joint:
            if key in joint_bounds:
                lower, upper = joint_bounds[key]
                range_span = upper - lower + 1e-8
                context_joint[key] = -1.0 + 2.0 * (context_joint[key] - lower) / range_span
                # Clamp to [-1, 1] for safety
                context_joint[key] = np.clip(context_joint[key], -1.0, 1.0)
        
        # Select which observations to include
        joint_features = self._select_context_obs(context_joint, cfg.MODEL.CONTEXT_OBS_TYPES)
        
        # Combine limb and joint features into node-centric format
        self.context_features = self._combine_limb_joint_context(limb_features, joint_features)
        
        print(f"Context encoding: {len(self.context_features)} features, "
              f"range [{self.context_features.min():.3f}, {self.context_features.max():.3f}]")

    def _select_context_obs(self, obs_dict, keys):
        """Select specific observation types from dictionary"""
        obs_to_ret = []
        for obs_type in keys:
            if obs_type in obs_dict:
                obs_to_ret.append(obs_dict[obs_type])
        
        if len(obs_to_ret):
            return np.hstack(tuple(obs_to_ret))
        else:
            # Return empty array instead of None
            return np.array([])

    def _combine_limb_joint_context(self, limb_obs, joint_obs):
        """Combine limb and joint observations into node-centric format"""
        num_limbs = self.metadata['num_limbs']
        num_joints = self.metadata['num_joints']
        
        # Handle empty observations
        if limb_obs is None or len(limb_obs) == 0:
            if joint_obs is None or len(joint_obs) == 0:
                return np.array([])
            return joint_obs.flatten()
        
        if joint_obs is None or len(joint_obs) == 0:
            return limb_obs.flatten()
        
        # Create node-centric observations where each node has limb + joint features
        # Each joint connects to a limb, so we need to map joints to limbs
        joint_obs_size = joint_obs.shape[1]
        
        # Create padded joint observations (2 joints max per limb in their architecture)
        joint_obs_padded = np.zeros((num_limbs, joint_obs_size * 2))
        
        # Map joints to limbs based on body attachment
        # For simplicity, assign each joint to its child body
        if len(self.edges) > 0:
            joint_to_limb = self.edges[::2]  # Get child indices from edges
            
            joint_assignments = np.zeros(num_limbs, dtype=int)
            for i, limb_idx in enumerate(joint_to_limb):
                # Adjust limb_idx (edges use 0-based indexing after subtracting world body)
                if 0 <= limb_idx < num_limbs and i < num_joints:
                    start_idx = joint_assignments[limb_idx] * joint_obs_size
                    end_idx = start_idx + joint_obs_size
                    if end_idx <= joint_obs_size * 2:
                        joint_obs_padded[limb_idx, start_idx:end_idx] = joint_obs[i]
                        joint_assignments[limb_idx] += 1
        
        # Combine limb and joint observations
        combined = np.hstack((limb_obs, joint_obs_padded))
        return combined.flatten()

    def _get_obs(self):
        """Get observations in dictionary format like UnimalEnv"""
        def _get_obs_per_limb(b):
            if "hip_pitch" in b:
                limb_type_vec = np.array((1, 0, 0, 0, 0, 0))
            elif "hip_roll" in b:
                limb_type_vec = np.array((0, 1, 0, 0, 0, 0))
            elif "hip_yaw" in b:
                limb_type_vec = np.array((0, 0, 1, 0, 0, 0))
            elif "knee" in b:
                limb_type_vec = np.array((0, 0, 0, 1, 0, 0))
            elif "ankle" in b:
                limb_type_vec = np.array((0, 0, 0, 0, 1, 0))
            else:
                limb_type_vec = np.array((0, 0, 0, 0, 0, 0))
            
            pelvis_x_pos = self.data.get_body_xpos("pelvis")[0]
            xpos = self.data.get_body_xpos(b).copy()
            xpos[0] -= pelvis_x_pos
            
            q = self.data.get_body_xquat(b)
            expmap = quat2expmap(q)
            
            xvelp = np.clip(self.data.get_body_xvelp(b), -10, 10)
            xvelr = self.data.get_body_xvelr(b)
            
            if b == "pelvis":
                angle = 0.0
                joint_range = np.array([0.0, 0.0])
            else:
                try:
                    body_id = self.sim.model.body_name2id(b)
                    jnt_adr = self.sim.model.body_jntadr[body_id]
                    
                    if jnt_adr == -1:
                        angle = 0.0
                        joint_range = np.array([0.0, 0.0])
                    else:
                        qpos_adr = self.sim.model.jnt_qposadr[jnt_adr]
                        angle = self.data.qpos[qpos_adr]
                        joint_range = self.sim.model.jnt_range[jnt_adr].copy()
                        
                        range_span = joint_range[1] - joint_range[0]
                        if range_span > 1e-8:
                            angle = (angle - joint_range[0]) / range_span
                        else:
                            angle = 0.0
                        
                        joint_range[0] = joint_range[0] / np.pi
                        joint_range[1] = joint_range[1] / np.pi
                except:
                    angle = 0.0
                    joint_range = np.array([0.0, 0.0])
            
            obs = np.concatenate([
                xpos,
                xvelp,
                xvelr,
                expmap,
                limb_type_vec,
                [angle],
                joint_range
            ])
            return obs

        # Get proprioceptive observations
        full_obs = np.concatenate(
            [_get_obs_per_limb(b) for b in self.model.body_names[1:]]
        ).ravel()
        
        # During __init__, graph structure isn't built yet
        # Return flat array for gym's observation space detection
        if not self._graph_built:
            return full_obs
        
        # After initialization, return dictionary format matching UnimalEnv
        return {
            'proprioceptive': full_obs,
            'context': self.context_features,
            'edges': self.edges,
            'traversals': self.traversals,
            'SWAT_RE': self.SWAT_RE,
        }

    def step(self, a):
        posbefore = self.sim.data.qpos[0]
        self.do_simulation(a, self.frame_skip)
        
        posafter = self.sim.data.qpos[0]
        height = self.sim.data.qpos[2]
        
        quat = self.sim.data.qpos[3:7]
        pitch = 2 * np.arcsin(np.clip(2 * (quat[0] * quat[2] - quat[3] * quat[1]), -1, 1))
        roll = 2 * np.arcsin(np.clip(2 * (quat[0] * quat[1] + quat[2] * quat[3]), -1, 1))
        
        # Reward components
        # 1. Alive bonus - reward for staying upright
        alive_bonus = 1.0
        
        # 2. Forward progress
        forward_reward = (posafter - posbefore) / self.dt
        
        # 3. Control cost
        ctrl_cost = 1e-3 * np.square(a).sum()
        
        # 4. Height reward - encourage staying at good height (stronger shaping)
        target_height = 0.79  # Updated to match actual standing height from resets
        height_reward = -2.0 * abs(height - target_height)
        
        # 5. Orientation reward - encourage staying upright
        upright_reward = -1.0 * (abs(pitch) + abs(roll))
        
        reward = (forward_reward + alive_bonus + height_reward + 
                 upright_reward - ctrl_cost)
        
        # Check termination conditions
        height_ok = (height > 0.4 and height < 1.2)
        pitch_ok = abs(pitch) < 0.8
        roll_ok = abs(roll) < 0.8
        done = not (height_ok and pitch_ok and roll_ok)
        
        # Track episode step count
        if not hasattr(self, '_episode_step'):
            self._episode_step = 0
            self._episode_return = 0
        
        self._episode_step += 1
        self._episode_return += reward
        
        info = {
            'name': 'unitree_g1_12dof',
            'forward_reward': forward_reward,
            'ctrl_cost': ctrl_cost,
            'height': height,
            'height_reward': height_reward,
            'upright_reward': upright_reward,
            'pitch': pitch,
            'roll': roll,
        }
        
        # CRITICAL FIX: Always add episode info when episode ends OR on timeout
        # TimeLimit wrapper will set done=True at step 1000, so we check step count
        episode_ended = done or self._episode_step >= 1000
        
        if episode_ended:
            info['episode'] = {
                'r': self._episode_return,
                'l': self._episode_step,
            }
            
            # Debug: print why episode ended (only first 20 episodes total across all envs)
            if not hasattr(self, '_episodes_logged'):
                self._episodes_logged = 0
            
            if self._episodes_logged < 20:
                if done:
                    reason = []
                    if not height_ok:
                        reason.append(f"height={height:.2f}")
                    if not pitch_ok:
                        reason.append(f"pitch={pitch:.2f}rad")
                    if not roll_ok:
                        reason.append(f"roll={roll:.2f}rad")
                    print(f"Episode ended at step {self._episode_step}, return={self._episode_return:.1f}, reason: {', '.join(reason)}")
                else:
                    print(f"Episode TIMEOUT at step {self._episode_step}, return={self._episode_return:.1f} (robot still upright)")
                self._episodes_logged += 1
            
            # Reset episode tracking ONLY if we're actually done (not timeout from wrapper)
            # The TimeLimit wrapper will reset the env, which calls reset_model()
            if done:
                self._episode_step = 0
                self._episode_return = 0
        
        return self._get_obs(), reward, done, info

    def reset_model(self):
        # Reset episode tracking
        self._episode_step = 0
        self._episode_return = 0
        
        qpos = self.init_qpos + self.np_random.uniform(
            low=-0.01,
            high=0.01, 
            size=self.model.nq
        )
        qvel = self.init_qvel + self.np_random.uniform(
            low=-0.01,
            high=0.01, 
            size=self.model.nv
        )
        
        qpos[2] = self.init_qpos[2] + self.np_random.uniform(low=-0.02, high=0.02)
        qpos[3:7] = qpos[3:7] / np.linalg.norm(qpos[3:7])
        
        self.set_state(qpos, qvel)
        
        # Debug: Log initial state very occasionally (only from one process)
        if not hasattr(self, '_reset_count'):
            self._reset_count = 0
            self._logged_reset = False
        self._reset_count += 1
        
        # Only log once when hitting certain milestones
        if self._reset_count in [1, 100, 500, 1000] and not self._logged_reset:
            height = qpos[2]
            quat = qpos[3:7]
            pitch = 2 * np.arcsin(np.clip(2 * (quat[0] * quat[2] - quat[3] * quat[1]), -1, 1))
            roll = 2 * np.arcsin(np.clip(2 * (quat[0] * quat[1] + quat[2] * quat[3]), -1, 1))
            print(f"[Reset #{self._reset_count}] Initial: height={height:.3f}m, pitch={pitch:.3f}rad, roll={roll:.3f}rad")
            self._logged_reset = True
        elif self._reset_count not in [1, 100, 500, 1000]:
            self._logged_reset = False
        
        return self._get_obs()

    def viewer_setup(self):
        self.viewer.cam.trackbodyid = 1
        self.viewer.cam.distance = self.model.stat.extent * 2.0
        self.viewer.cam.lookat[2] = 0.8
        self.viewer.cam.elevation = -20


def make_env(xml):
    """Create environment - xml is the path to XML file"""
    env = ModularEnv(xml)
    print(f"Made env for {xml}")
    
    if cfg.MODEL.MLP.CONSISTENT_PADDING:
        env = ConsistentModularObservationPadding(env)
        env = ConsistentModularActionPadding(env)
    else:
        env = ModularObservationPadding(env)
        env = ModularActionPadding(env)
    
    return env