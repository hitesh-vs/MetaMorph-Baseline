"""
G1 12DOF MuJoCo Environment with Graph-based Policy Support
Adapted reward structure from Unitree G1 (IsaacGym) for MuJoCo direct torque control

Reward structure based on Unitree G1:
  - base_height: Strong penalty for wrong height (encourages standing)
  - tracking_lin_vel: Reward for forward movement
  - orientation: Penalty for tilting
  - dof_vel: Penalty for high joint velocities (smooth motion)
  - dof_acc: Penalty for joint accelerations
  - action_rate: Penalty for jerky movements (smooth control)
  - contact: Reward for proper foot contact with gait phase awareness
  - feet_swing_height: Penalty for dragging feet during swing
  - alive: Small bonus for survival
"""

from gym import utils
from gym.envs.mujoco import mujoco_env
import numpy as np
import os

from modular.utils import *
from modular.wrappers import *
from metamorph.config import cfg

import re
import tempfile
import os

# For contact force computation (mujoco_py)
try:
    from mujoco_py import functions as mjf
except ImportError:
    mjf = None

# Graph parser lives in graphs/, not in the env
from graphs.parser import MujocoGraphParser


def quat_to_euler(quat):
    """
    Convert quaternion [w, x, y, z] to Euler angles [roll, pitch, yaw].
    
    Args:
        quat: Quaternion as [w, x, y, z]
        
    Returns:
        roll, pitch, yaw (in radians)
    """
    w, x, y, z = quat[0], quat[1], quat[2], quat[3]
    
    # Roll (rotation around X-axis)
    sinr_cosp = 2 * (w * x + y * z)
    cosr_cosp = 1 - 2 * (x * x + y * y)
    roll = np.arctan2(sinr_cosp, cosr_cosp)
    
    # Pitch (rotation around Y-axis)
    sinp = 2 * (w * y - z * x)
    pitch = np.arcsin(np.clip(sinp, -1, 1))
    
    # Yaw (rotation around Z-axis)
    siny_cosp = 2 * (w * z + x * y)
    cosy_cosp = 1 - 2 * (y * y + z * z)
    yaw = np.arctan2(siny_cosp, cosy_cosp)
    
    return roll, pitch, yaw


class ModularEnv(mujoco_env.MujocoEnv, utils.EzPickle):
    def __init__(self, xml):
        self.xml = xml

        # FIX 1 — derive robot name from XML filename, strip _stripped suffix
        base = os.path.basename(xml)                         # e.g. "g1_12dof_stripped.xml"
        base = os.path.splitext(base)[0]                     # e.g. "g1_12dof_stripped"
        self._robot_name = re.sub(r'_stripped$', '', base)   # e.g. "g1_12dof"

        # ── graph encoding mode ──────────────────────────────────────────
        # "none" | "onehot" | "topological"
        self.graph_encoding = cfg.MODEL.GRAPH_ENCODING

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

        # Placeholders — filled after MujocoEnv.__init__
        self.edges = np.array([])
        self.traversals = np.array([])
        self.SWAT_RE = np.zeros([cfg.MODEL.MAX_LIMBS, cfg.MODEL.MAX_LIMBS, 3])
        self.context_features = np.array([])

        # Graph data returned in obs (only populated when graph_encoding != "none")
        self._graph_node_features = None   # (N, feat_dim) numpy float32
        self._graph_A_norm = None          # (N, N) normalized adjacency, numpy float32

        self._graph_built = False

        # ── Standing pose angles (like Unitree G1) ────────────────────────
        self.standing_angles = {
            'left_hip_yaw_joint': 0.0,
            'left_hip_roll_joint': 0.0,
            'left_hip_pitch_joint': -0.1,
            'left_knee_joint': 0.3,
            'left_ankle_pitch_joint': -0.2,
            'left_ankle_roll_joint': 0.0,
            'right_hip_yaw_joint': 0.0,
            'right_hip_roll_joint': 0.0,
            'right_hip_pitch_joint': -0.1,
            'right_knee_joint': 0.3,
            'right_ankle_pitch_joint': -0.2,
            'right_ankle_roll_joint': 0.0,
        }

        # ✅ CRITICAL FIX: Initialize these BEFORE calling MujocoEnv.__init__
        # because MujocoEnv.__init__ calls step() which needs these attributes
        self._root_body = "pelvis"   # safe default; overwritten below
        self._init_height = 0.79     # safe default; overwritten below
        self._last_action = np.array([])  # Will be resized after model is loaded
        self._last_qpos = np.array([])
        self._last_qvel = np.array([])
        self._episode_step = 0
        self._phase = 0.0
        self._gait_period = 0.8
        self._episode_return = 0.0
        self._graph_built = False

        # NOW call MujocoEnv.__init__ (which internally calls step())
        mujoco_env.MujocoEnv.__init__(self, xml, 4)
        utils.EzPickle.__init__(self)

        # ── After MujocoEnv init, properly size the tracking arrays ──────
        self._last_action = np.zeros(self.model.nu)
        self._last_qpos = self.init_qpos.copy()
        self._last_qvel = np.zeros(self.model.nv)

        # FIX 2 — derive limb/joint counts from the loaded model, not hardcoded constants.
        # body_names[0] is the MuJoCo world body; subtract 1 to get real robot bodies.
        self.metadata['num_limbs'] = len(self.model.body_names) - 1
        self.metadata['num_joints'] = self.sim.model.nu

        self.agent_limb_names = self.model.body_names[1:]

        # FIX 3 — root body is the first real body, not assumed to be "pelvis"
        self._root_body = self.model.body_names[1]

        # FIX 4 — capture the robot's natural standing height from the initial qpos
        self._init_height = float(self.init_qpos[2])

        # ── Gait cycle imitation (optional) ──────────────────────────────
        try:
            self._gait_cycle = np.load("gait_cycle_clean.npy")  # (400, 12)
            self._gait_cycle_len = len(self._gait_cycle)
            self._imitation_weight = 0.5  # anneal to 0 later
        except FileNotFoundError:
            self._gait_cycle = None
            self._gait_cycle_len = 0
            self._imitation_weight = 0.0

        self._build_graph_structure()
        self._compute_context_encoding()

        # ── Parse XML graph and precompute static graph data ─────────────
        if self.graph_encoding != "none":
            self._init_graph_data()

        self._graph_built = True

        # ── Foot body indices for contact detection ──────────────────────
        self._init_foot_bodies()

    # ──────────────────────────────────────────────────────────────────────
    # Foot body initialization
    # ──────────────────────────────────────────────────────────────────────

    def _init_foot_bodies(self):
        """Initialize foot body indices for contact detection."""
        self.left_foot_body_id = None
        self.right_foot_body_id = None
        
        # Find foot bodies by name
        for body_id, body_name in enumerate(self.model.body_names):
            if 'left_ankle' in body_name.lower():
                self.left_foot_body_id = body_id
            elif 'right_ankle' in body_name.lower():
                self.right_foot_body_id = body_id
        
        # Fallback: use last two bodies if names don't match
        if self.left_foot_body_id is None or self.right_foot_body_id is None:
            num_bodies = len(self.model.body_names)
            self.left_foot_body_id = num_bodies - 2
            self.right_foot_body_id = num_bodies - 1

    # ──────────────────────────────────────────────────────────────────────
    # Contact force computation (mujoco_py compatible)
    # ──────────────────────────────────────────────────────────────────────

    def _get_contact_force_magnitude(self, body_id):
        """
        Compute total contact force magnitude on specified body.
        Uses mujoco_py API.
        
        Args:
            body_id: Body ID to check contacts for
            
        Returns:
            Total contact force magnitude (float)
        """
        if mjf is None:
            return 0.0
            
        total_force = 0.0
        
        for contact_idx in range(self.sim.data.ncon):
            contact = self.sim.data.contact[contact_idx]
            
            # Get which bodies are involved in this contact
            geom1_id = contact.geom1
            geom2_id = contact.geom2
            
            geom1_body = self.sim.model.geom_bodyid[geom1_id]
            geom2_body = self.sim.model.geom_bodyid[geom2_id]
            
            # Check if this contact involves our body
            if geom1_body == body_id or geom2_body == body_id:
                # Get contact force using mujoco_py
                force = np.zeros(6)
                mjf.mj_contactForce(self.sim.model, self.sim.data, contact_idx, force)
                
                # Extract normal force (first 3 components are force)
                force_magnitude = np.linalg.norm(force[:3])
                total_force += force_magnitude
        
        return total_force

    def _get_foot_position(self, body_id):
        """Get Z position of a foot body."""
        try:
            return self.sim.data.get_body_xpos(self.model.body_names[body_id])[2]
        except:
            return 0.0

    def _compute_gait_phase(self):
        """
        Compute biped gait phase (0-1 cycle).
        Left leg: phase [0, 1]
        Right leg: phase offset by 0.5 (180 degrees out of phase)
        """
        phase = (self._episode_step * self.dt) % self._gait_period / self._gait_period
        return phase, (phase + 0.5) % 1.0

    # ──────────────────────────────────────────────────────────────────────
    # Graph init (only called when graph_encoding != "none")
    # ──────────────────────────────────────────────────────────────────────

    def _init_graph_data(self):
        """
        Parse XML, build node features and normalised adjacency.
        These are STATIC for a given robot — computed once at init, not per step.
        """
        parser = MujocoGraphParser(self.xml)

        # Node features: shape (N, feat_dim)
        self._graph_node_features = parser.get_features(self.graph_encoding)

        # Normalised adjacency: shape (N, N)
        self._graph_A_norm = parser.normalized_adjacency()

        print(
            f"[{self.graph_encoding}] Graph: {parser.N} nodes, "
            f"features {self._graph_node_features.shape}, "
            f"A_norm {self._graph_A_norm.shape}"
        )

    # ──────────────────────────────────────────────────────────────────────
    # Kinematics graph (for SWAT / context — unchanged from original)
    # ──────────────────────────────────────────────────────────────────────

    def _build_graph_structure(self):
        body_idxs = list(range(1, self.metadata['num_limbs'] + 1))
        joint_to = self.sim.model.jnt_bodyid[1:].copy() - 1
        body_parentids = self.sim.model.body_parentid.copy()
        joint_from = np.array([body_parentids[child + 1] - 1 for child in joint_to])
        assert len(joint_to) == self.metadata['num_joints']
        self.edges = np.vstack((joint_to, joint_from)).T.flatten().astype(np.int32)

        parents = [-1] * self.metadata['num_limbs']
        for i in range(len(joint_to)):
            child_idx = joint_to[i]
            parent_idx = joint_from[i]
            if 0 <= child_idx < len(parents) and parent_idx >= -1:
                parents[child_idx] = parent_idx

        self.traversals = self._get_traversal(parents)

        if cfg.MODEL.TRANSFORMER.USE_SWAT_RE:
            self.SWAT_RE = self._get_graph_dict(parents)
        else:
            self.SWAT_RE = np.zeros([cfg.MODEL.MAX_LIMBS, cfg.MODEL.MAX_LIMBS, 3])

    def _get_traversal(self, parents):
        root = parents.index(-1) if -1 in parents else 0
        children = {i: [] for i in range(len(parents))}
        for i, parent in enumerate(parents):
            if parent >= 0:
                children[parent].append(i)
        traversal = []
        def dfs(node):
            traversal.append(node)
            for child in sorted(children[node]):
                dfs(child)
        dfs(root)
        return np.array(traversal, dtype=np.int32)

    def _get_graph_dict(self, parents):
        n = len(parents)
        graph_dict = np.zeros([cfg.MODEL.MAX_LIMBS, cfg.MODEL.MAX_LIMBS, 3])
        children = {i: [] for i in range(n)}
        for i, parent in enumerate(parents):
            if parent >= 0:
                children[parent].append(i)
        for i in range(n):
            for j in range(n):
                if i == j:
                    graph_dict[i, j] = [1, 0, 0]
                elif parents[i] == j:
                    graph_dict[i, j] = [0, 1, 0]
                elif j in children[i]:
                    graph_dict[i, j] = [0, 0, 1]
        return graph_dict

    # ──────────────────────────────────────────────────────────────────────
    # Context encoding (morphology properties — unchanged from original)
    # ──────────────────────────────────────────────────────────────────────

    def _compute_normalization_bounds(self):
        num_limbs = self.metadata['num_limbs']
        num_joints = self.metadata['num_joints']
        body_idxs = list(range(1, num_limbs + 1))
        limb_bounds = {}
        body_pos = self.sim.model.body_pos[body_idxs, :]
        limb_bounds['body_pos'] = (body_pos.min(axis=0, keepdims=True),
                                    body_pos.max(axis=0, keepdims=True))
        body_ipos = self.sim.model.body_ipos[body_idxs, :]
        limb_bounds['body_ipos'] = (body_ipos.min(axis=0, keepdims=True),
                                     body_ipos.max(axis=0, keepdims=True))
        limb_bounds['body_iquat'] = (np.array([[-1,-1,-1,-1]]), np.array([[1,1,1,1]]))
        limb_bounds['geom_quat'] = (np.array([[-1,-1,-1,-1]]), np.array([[1,1,1,1]]))
        body_mass = self.sim.model.body_mass[body_idxs]
        limb_bounds['body_mass'] = (np.array([[body_mass.min()]]),
                                     np.array([[body_mass.max()]]))
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
            limb_bounds['body_shape'] = (np.array([[0.01,0.01]]), np.array([[0.2,0.2]]))
            limb_bounds['body_friction'] = (np.array([[0.5]]), np.array([[1.5]]))
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
        num_limbs = self.metadata['num_limbs']
        num_joints = self.metadata['num_joints']
        body_idxs = list(range(1, num_limbs + 1))
        limb_bounds, joint_bounds = self._compute_normalization_bounds()
        context_limb = {}
        context_joint = {}
        context_limb['body_pos'] = self.sim.model.body_pos[body_idxs, :].copy()
        context_limb['body_ipos'] = self.sim.model.body_ipos[body_idxs, :].copy()
        context_limb['body_iquat'] = self.sim.model.body_iquat[body_idxs, :].copy()
        geom_idxs = []
        for body_idx in body_idxs:
            for geom_idx in range(self.sim.model.ngeom):
                if self.sim.model.geom_bodyid[geom_idx] == body_idx:
                    geom_idxs.append(geom_idx)
                    break
        if len(geom_idxs) == num_limbs:
            context_limb['geom_quat'] = self.sim.model.geom_quat[geom_idxs, :].copy()
            context_limb['body_shape'] = self.sim.model.geom_size[geom_idxs, :2].copy()
            context_limb['body_friction'] = self.sim.model.geom_friction[geom_idxs, 0:1].copy()
        else:
            context_limb['geom_quat'] = np.tile([1,0,0,0], (num_limbs,1))
            context_limb['body_shape'] = np.zeros((num_limbs, 2))
            context_limb['body_friction'] = np.ones((num_limbs, 1))
        context_limb['body_mass'] = self.sim.model.body_mass[body_idxs].copy()[:,np.newaxis]
        for key in context_limb:
            if key in limb_bounds:
                lower, upper = limb_bounds[key]
                range_span = upper - lower + 1e-8
                context_limb[key] = -1.0 + 2.0 * (context_limb[key] - lower) / range_span
                context_limb[key] = np.clip(context_limb[key], -1.0, 1.0)
        limb_features = self._select_context_obs(context_limb, cfg.MODEL.CONTEXT_OBS_TYPES)
        context_joint['jnt_pos'] = self.sim.model.jnt_pos[1:, :].copy()
        context_joint['joint_range'] = self.sim.model.jnt_range[1:, :].copy()
        context_joint['joint_axis'] = self.sim.model.jnt_axis[1:, :].copy()
        context_joint['gear'] = self.sim.model.actuator_gear[:, 0:1].copy()
        context_joint['armature'] = self.sim.model.dof_armature[6:].copy()[:,np.newaxis]
        context_joint['damping'] = self.sim.model.dof_damping[6:].copy()[:,np.newaxis]
        for key in context_joint:
            if key in joint_bounds:
                lower, upper = joint_bounds[key]
                range_span = upper - lower + 1e-8
                context_joint[key] = -1.0 + 2.0 * (context_joint[key] - lower) / range_span
                context_joint[key] = np.clip(context_joint[key], -1.0, 1.0)
        joint_features = self._select_context_obs(context_joint, cfg.MODEL.CONTEXT_OBS_TYPES)
        self.context_features = self._combine_limb_joint_context(limb_features, joint_features)

    def _select_context_obs(self, obs_dict, keys):
        obs_to_ret = []
        for obs_type in keys:
            if obs_type in obs_dict:
                obs_to_ret.append(obs_dict[obs_type])
        if len(obs_to_ret):
            return np.hstack(tuple(obs_to_ret))
        return np.array([])

    def _combine_limb_joint_context(self, limb_obs, joint_obs):
        num_limbs = self.metadata['num_limbs']
        num_joints = self.metadata['num_joints']
        if limb_obs is None or len(limb_obs) == 0:
            return joint_obs.flatten() if (joint_obs is not None and len(joint_obs)) else np.array([])
        if joint_obs is None or len(joint_obs) == 0:
            return limb_obs.flatten()
        joint_obs_size = joint_obs.shape[1]
        joint_obs_padded = np.zeros((num_limbs, joint_obs_size * 2))
        if len(self.edges) > 0:
            joint_to_limb = self.edges[::2]
            joint_assignments = np.zeros(num_limbs, dtype=int)
            for i, limb_idx in enumerate(joint_to_limb):
                if 0 <= limb_idx < num_limbs and i < num_joints:
                    start_idx = joint_assignments[limb_idx] * joint_obs_size
                    end_idx = start_idx + joint_obs_size
                    if end_idx <= joint_obs_size * 2:
                        joint_obs_padded[limb_idx, start_idx:end_idx] = joint_obs[i]
                        joint_assignments[limb_idx] += 1
        combined = np.hstack((limb_obs, joint_obs_padded))
        return combined.flatten()

    # ──────────────────────────────────────────────────────────────────────
    # Observation
    # ──────────────────────────────────────────────────────────────────────

    def _get_obs_per_limb(self, b):
        """
        Proprioceptive observation for a single body link.

        When graph_encoding == "none":  includes the 6-dim one-hot limb_type_vec
                                        (original behaviour, no GCN).
        When graph_encoding != "none":  limb_type_vec is OMITTED.
                                        The GCN embedding carries structural info
                                        and is concatenated by the model, not here.
        """
        root_x_pos = self.data.get_body_xpos(self._root_body)[0]
        xpos = self.data.get_body_xpos(b).copy()
        xpos[0] -= root_x_pos

        q = self.data.get_body_xquat(b)
        expmap = quat2expmap(q)

        xvelp = np.clip(self.data.get_body_xvelp(b), -10, 10)
        xvelr = self.data.get_body_xvelr(b)

        if b == self._root_body:
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

        # ── "none" baseline: include one-hot limb type (original behaviour) ──
        if self.graph_encoding == "none":
            n = b.lower()
            if "hip_pitch" in n:
                limb_type_vec = np.array((1, 0, 0, 0, 0, 0))
            elif "hip_roll" in n:
                limb_type_vec = np.array((0, 1, 0, 0, 0, 0))
            elif "hip_yaw" in n:
                limb_type_vec = np.array((0, 0, 1, 0, 0, 0))
            elif "knee" in n:
                limb_type_vec = np.array((0, 0, 0, 1, 0, 0))
            elif "ankle" in n:
                limb_type_vec = np.array((0, 0, 0, 0, 1, 0))
            else:
                limb_type_vec = np.array((0, 0, 0, 0, 0, 0))

            return np.concatenate([xpos, xvelp, xvelr, expmap, limb_type_vec,
                                   [angle], joint_range])

        # ── GCN modes: omit limb_type_vec, GCN embedding added by model ──────
        return np.concatenate([xpos, xvelp, xvelr, expmap, [angle], joint_range])

    def _get_obs(self):
        full_obs = np.concatenate(
            [self._get_obs_per_limb(b) for b in self.model.body_names[1:]]
        ).ravel()

        if not self._graph_built:
            return full_obs

        obs = {
            'proprioceptive': full_obs,
            'context':        self.context_features,
            'edges':          self.edges,
            'traversals':     self.traversals,
            'SWAT_RE':        self.SWAT_RE,
        }

        if self.graph_encoding != "none":
            obs['graph_node_features'] = self._graph_node_features
            obs['graph_A_norm']        = self._graph_A_norm

        return obs

    # ──────────────────────────────────────────────────────────────────────
    # Reward computation (Unitree G1 style with proper physics)
    # ──────────────────────────────────────────────────────────────────────

    def _compute_rewards(self, action):
        """
        G1-style rewards - MATCHING UNITREE IMPLEMENTATION EXACTLY
        
        All reward scales are applied with proper dt normalization.
        """
        
        # ── Guard: Handle uninitialized state ────────────────────────────────
        if len(self._last_action) == 0:
            self._last_action = np.zeros(len(action))
            self._last_qvel = np.zeros(self.model.nv)
            return 0.0, {'reward_base_height': 0.0, 'reward_tracking_lin_vel': 0.0,
                         'reward_lin_vel_z': 0.0, 'reward_ang_vel_xy': 0.0,
                         'reward_orientation': 0.0, 'reward_dof_vel': 0.0,
                         'reward_dof_acc': 0.0, 'reward_action_rate': 0.0,
                         'reward_contact': 0.0, 'reward_contact_no_vel': 0.0,
                         'reward_feet_swing_height': 0.0, 'reward_alive': 0.0,
                         'height': 0.0, 'forward_vel': 0.0, 'pitch': 0.0, 'roll': 0.0}
        
        # ── Calculate timestep ─────────────────────────────────────────────
        dt = self.frame_skip * self.model.opt.timestep  # Usually ~0.01s
        
        # ── Extract state information ──────────────────────────────────────
        height = self.sim.data.qpos[2]
        
        # Get quaternion and convert to Euler angles (proper conversion!)
        quat = self.sim.data.qpos[3:7]
        roll, pitch, yaw = quat_to_euler(quat)
        
        # Root body linear velocity in world frame (not position change!)
        root_vel = self.sim.data.get_body_xvelp(self._root_body)
        forward_vel = root_vel[0]  # X-axis velocity
        lin_vel_z = root_vel[2]    # Z-axis velocity
        
        # Root body angular velocity
        ang_vel_body = self.sim.data.get_body_xvelr(self._root_body)
        ang_vel_xy = ang_vel_body[:2]  # Roll/pitch angular velocity
        
        # Joint states (skip base 6-DOF)
        qvel = self.sim.data.qvel[6:]
        
        # ── UNITREE G1 REWARD SCALES (base values, multiplied by dt) ────────
        BASE_SCALES = {
            'base_height': -10.0,
            'tracking_lin_vel': 1.0,
            'lin_vel_z': -2.0,
            'ang_vel_xy': -0.05,
            'orientation': -1.0,
            'dof_vel': -1e-3,
            'dof_acc': -2.5e-7,
            'action_rate': -0.01,
            'contact': 0.18,
            'contact_no_vel': -0.2,
            'feet_swing_height': -20.0,
            'alive': 0.15,
        }
        
        # ── COMPUTE REWARDS ────────────────────────────────────────────────
        
        # 1. BASE HEIGHT (strong penalty for wrong height)
        target_height = self._init_height
        height_error = height - target_height
        reward_base_height = BASE_SCALES['base_height'] * (height_error ** 2) * dt
        
        # 2. TRACKING LINEAR VELOCITY (reward forward movement)
        reward_tracking_lin_vel = BASE_SCALES['tracking_lin_vel'] * np.exp(-0.25 * (forward_vel - 0.5) ** 2)
        
        # 3. LINEAR VELOCITY Z (penalize vertical motion)
        reward_lin_vel_z = BASE_SCALES['lin_vel_z'] * (lin_vel_z ** 2) * dt
        
        # 4. ANGULAR VELOCITY XY (penalize rolling/pitching)
        reward_ang_vel_xy = BASE_SCALES['ang_vel_xy'] * np.sum(ang_vel_xy ** 2) * dt
        
        # 5. ORIENTATION (keep upright)
        reward_orientation = BASE_SCALES['orientation'] * (pitch ** 2 + roll ** 2)
        
        # 6. DOF VELOCITY (smooth motion)
        reward_dof_vel = BASE_SCALES['dof_vel'] * np.sum(qvel ** 2) * dt
        
        # 7. DOF ACCELERATION (smooth control)
        dof_acc = np.sum(np.square((qvel - self._last_qvel[6:]) / dt))
        reward_dof_acc = BASE_SCALES['dof_acc'] * dof_acc * dt
        
        # 8. ACTION RATE (smooth policy)
        action_rate = np.sum(np.square(action - self._last_action))
        reward_action_rate = BASE_SCALES['action_rate'] * action_rate * dt
        
        # 9. CONTACT (feet in contact)
        left_contact_force = self._get_contact_force_magnitude(self.left_foot_body_id)
        right_contact_force = self._get_contact_force_magnitude(self.right_foot_body_id)
        left_contact = float(left_contact_force > 1.0)
        right_contact = float(right_contact_force > 1.0)
        reward_contact = BASE_SCALES['contact'] * (left_contact + right_contact)
        
        # 10. CONTACT NO VELOCITY (penalize slipping)
        left_vel = np.linalg.norm(self.sim.data.get_body_xvelp(
            self.model.body_names[self.left_foot_body_id]))
        right_vel = np.linalg.norm(self.sim.data.get_body_xvelp(
            self.model.body_names[self.right_foot_body_id]))
        reward_contact_no_vel = BASE_SCALES['contact_no_vel'] * (
            left_contact * (left_vel ** 2) + right_contact * (right_vel ** 2)
        ) * dt
        
        # 11. FEET SWING HEIGHT (penalize dragging feet)
        left_foot_z = self._get_foot_position(self.left_foot_body_id)
        right_foot_z = self._get_foot_position(self.right_foot_body_id)
        swing_height_target = 0.08
        
        # ✅ CORRECT: Penalize (negative) for low feet during swing
        left_swing = max(0, swing_height_target - left_foot_z) * (1.0 - left_contact)
        right_swing = max(0, swing_height_target - right_foot_z) * (1.0 - right_contact)
        reward_feet_swing_height = BASE_SCALES['feet_swing_height'] * (left_swing + right_swing) * dt
        
        # 12. ALIVE BONUS (small reward for surviving)
        reward_alive = BASE_SCALES['alive']
        
        # ── COMBINE ALL REWARDS ────────────────────────────────────────────
        total_reward = (
            reward_base_height +
            reward_tracking_lin_vel +
            reward_lin_vel_z +
            reward_ang_vel_xy +
            reward_orientation +
            reward_dof_vel +
            reward_dof_acc +
            reward_action_rate +
            reward_contact +
            reward_contact_no_vel +
            reward_feet_swing_height +
            reward_alive
        )
        
        # Return reward components for logging
        reward_dict = {
            'reward_base_height': reward_base_height,
            'reward_tracking_lin_vel': reward_tracking_lin_vel,
            'reward_lin_vel_z': reward_lin_vel_z,
            'reward_ang_vel_xy': reward_ang_vel_xy,
            'reward_orientation': reward_orientation,
            'reward_dof_vel': reward_dof_vel,
            'reward_dof_acc': reward_dof_acc,
            'reward_action_rate': reward_action_rate,
            'reward_contact': reward_contact,
            'reward_contact_no_vel': reward_contact_no_vel,
            'reward_feet_swing_height': reward_feet_swing_height,
            'reward_alive': reward_alive,
            'height': height,
            'forward_vel': forward_vel,
            'pitch': pitch,
            'roll': roll,
            'total_reward': total_reward,
        }
        
        self.current_forward_vel = forward_vel

        return total_reward, reward_dict

    # ──────────────────────────────────────────────────────────────────────
    # Step / Reset
    # ──────────────────────────────────────────────────────────────────────

    def step(self, a):
        """
        Step the environment with improved tracking.
        """
        self._last_qpos = self.sim.data.qpos.copy()
        self._last_qvel = self.sim.data.qvel.copy()
        self.do_simulation(a, self.frame_skip)
        
        # Compute reward
        reward, reward_dict = self._compute_rewards(a)
        
        # Get observations
        obs = self._get_obs()
        
        # Check termination conditions
        height = self.sim.data.qpos[2]
        quat = self.sim.data.qpos[3:7]
        roll, pitch, yaw = quat_to_euler(quat)
        
        # Terminate if robot falls or tips over
        height_ok = (height > 0.4 and height < 1.2)  # Height bounds
        pitch_ok = abs(pitch) < 0.8                    # Max pitch before falling
        roll_ok = abs(roll) < 0.8                      # Max roll before falling
        done = not (height_ok and pitch_ok and roll_ok)
        
        # Tracking for episode info
        self._episode_step += 1
        self._episode_return += reward
        
        # Update last action for next step
        self._last_action = a.copy()
        
        # Info dict
        info = {
            'name': self._robot_name,
        }
        info.update(reward_dict)
        
        # Episode terminal info
        if done or self._episode_step >= 1000:
            info['episode'] = {
                'r': self._episode_return, 
                'l': self._episode_step,
                'height': height,
                'pitch': pitch,
                'roll': roll,
            }
            if done:
                self._episode_step = 0
                self._episode_return = 0.0

        # Recalculate for debug (safest approach)
        root_vel = self.sim.data.get_body_xvelp(self._root_body)
        forward_vel = root_vel[0]
        
        return obs, reward, done, info

    def reset_model(self):
        """
        Reset the environment to a standing pose (like Unitree).
        
        Key differences from generic reset:
        1. Initialize joints to standing angles (not zero)
        2. Small random perturbations (like Unitree: 0.5-1.5x of defaults)
        3. Zero velocity
        4. Proper height initialization
        """
        self._episode_step = 0
        self._episode_return = 0.0
        
        # Start with init_qpos as base
        qpos = self.init_qpos.copy()
        
        # Number of actual DOFs (excluding free body's 7 DOF)
        # qpos: [x, y, z, qw, qx, qy, qz, joint1, joint2, ...]
        #        0  1  2   3   4   5   6    7      8     ...
        num_qpos_dofs = len(qpos)
        num_free_dofs = 7  # x, y, z + quaternion (4)
        num_joint_dofs = num_qpos_dofs - num_free_dofs
        
        # Set joint angles to standing pose (start from index 7)
        joint_start_idx = 7
        for i, joint_name in enumerate(self.model.joint_names[1:]):  # Skip free joint
            if i < num_joint_dofs:
                qpos_idx = joint_start_idx + i
                
                # Check if this joint has a standing angle defined
                if joint_name in self.standing_angles:
                    qpos[qpos_idx] = self.standing_angles[joint_name]
                else:
                    # Fallback: use current value or 0
                    qpos[qpos_idx] = 0.0
        
        # Add small random perturbations to joint positions only (not base)
        # Perturbations for joints only (indices 7 onward)
        qpos[joint_start_idx:joint_start_idx + num_joint_dofs] += \
            self.np_random.uniform(low=-0.02, high=0.02, size=num_joint_dofs)
        
        # Set height with small perturbation
        qpos[2] = self._init_height + self.np_random.uniform(low=-0.02, high=0.02)
        
        # Ensure quaternion is normalized (indices 3-6: qw, qx, qy, qz)
        quat = qpos[3:7]
        qpos[3:7] = quat / np.linalg.norm(quat)
        
        # Initialize velocity (mostly zero, small random perturbations for joints only)
        qvel = np.zeros(self.model.nv)
        
        # Add small perturbations to joint velocities only (skip base linear/angular vel)
        # First 6 DOF are base linear (0-2) and angular (3-5) velocities
        num_vel_dofs = self.model.nv - 6
        if num_vel_dofs > 0:
            qvel[6:6 + num_vel_dofs] = self.np_random.uniform(
                low=-0.01, high=0.01, size=num_vel_dofs
            )
        
        # Zero out base linear and angular velocities explicitly
        qvel[0:3] = 0.0  # Linear velocity
        qvel[3:6] = 0.0  # Angular velocity
        
        # Set the state
        self.set_state(qpos, qvel)
        
        # ✅ CRITICAL: Update tracking AFTER setting state
        # This ensures _last_qvel reflects the actual initial velocity
        self._last_qpos = self.sim.data.qpos.copy()
        self._last_qvel = self.sim.data.qvel.copy()
        self._last_action = np.zeros(self.model.nu)
        
        # Reset gait phase
        self._phase = 0.0
        self._episode_step = 0
        
        return self._get_obs()

    def viewer_setup(self):
        self.viewer.cam.trackbodyid = 1
        self.viewer.cam.distance = self.model.stat.extent * 2.0
        self.viewer.cam.lookat[2] = self._init_height
        self.viewer.cam.elevation = -20


def make_env(xml):
    """Create environment - xml is the path to XML file"""
    env = ModularEnv(xml)

    print(f"Made env for {xml} | graph_encoding={cfg.MODEL.GRAPH_ENCODING}")
    if cfg.MODEL.MLP.CONSISTENT_PADDING:
        env = ConsistentModularObservationPadding(env)
        env = ConsistentModularActionPadding(env)
    else:
        env = ModularObservationPadding(env)
        env = ModularActionPadding(env)
    return env