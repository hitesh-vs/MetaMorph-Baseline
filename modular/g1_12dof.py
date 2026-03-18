"""
modular/g1_12dof.py  (modified)

Three encoding modes controlled by cfg.MODEL.GRAPH_ENCODING:

  "none"         —  baseline. Obs = [prop | one-hot limb_type_vec].
                    Identical to the original ModularEnv behaviour.

  "onehot"       —  GCN with name-heuristic one-hot node features.
                    Obs dict carries raw graph data; GCN runs in the model.

  "topological"  —  GCN with topology-only node features (no name parsing).
                    Same obs dict shape as "onehot".

The env never instantiates the GCN.  It only:
  1. Parses the XML into a graph (MujocoGraphParser).
  2. Precomputes node features (onehot OR topological).
  3. Precomputes the normalized adjacency matrix.
  4. Returns these as static tensors in the obs dict so the model can use them.

This keeps env/model separation clean:
  - env  → data provider (observations, rewards, graph metadata)
  - model → learner      (GCN params trained by PPO)
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

# Graph parser lives in graphs/, not in the env
from graphs.parser import MujocoGraphParser

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

        # FIX 3 & 4 — pre-init sentinels so step() doesn't AttributeError when
        # MujocoEnv.__init__ calls step() internally before our __init__ completes.
        self._root_body   = "pelvis"   # safe default; overwritten below
        self._init_height = 0.79       # safe default; overwritten below

        mujoco_env.MujocoEnv.__init__(self, xml, 4)
        utils.EzPickle.__init__(self)

        # FIX 2 — derive limb/joint counts from the loaded model, not hardcoded constants.
        # body_names[0] is the MuJoCo world body; subtract 1 to get real robot bodies.
        self.metadata['num_limbs'] = len(self.model.body_names) - 1
        self.metadata['num_joints'] = self.sim.model.nu

        self.agent_limb_names = self.model.body_names[1:]

        # FIX 3 — root body is the first real body, not assumed to be "pelvis"
        self._root_body = self.model.body_names[1]

        # FIX 4 — capture the robot's natural standing height from the initial qpos
        self._init_height = float(self.init_qpos[2])

        self._gait_cycle = np.load("gait_cycle_clean.npy")  # (400, 12)
        self._gait_cycle_len = len(self._gait_cycle)
        self._imitation_weight = 0.5  # anneal to 0 later

        self._build_graph_structure()
        self._compute_context_encoding()

        # ── Parse XML graph and precompute static graph data ─────────────
        if self.graph_encoding != "none":
            self._init_graph_data()

        self._graph_built = True

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
        # FIX 3 — use self._root_body instead of hardcoded "pelvis"
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
            # Called during MujocoEnv.__init__ — return flat array for space detection
            return full_obs

        obs = {
            'proprioceptive': full_obs,
            'context':        self.context_features,
            'edges':          self.edges,
            'traversals':     self.traversals,
            'SWAT_RE':        self.SWAT_RE,
        }

        # ── Attach graph data for GCN modes ───────────────────────────────
        if self.graph_encoding != "none":
            # These are STATIC (same every step) — the model uses them to run GCN
            obs['graph_node_features'] = self._graph_node_features   # (N, feat_dim)
            obs['graph_A_norm']        = self._graph_A_norm           # (N, N)

        return obs

    # ──────────────────────────────────────────────────────────────────────
    # Step / Reset
    # ──────────────────────────────────────────────────────────────────────

    def step(self, a):
        posbefore = self.sim.data.qpos[0]
        self.do_simulation(a, self.frame_skip)
        posafter  = self.sim.data.qpos[0]
        height    = self.sim.data.qpos[2]

        quat  = self.sim.data.qpos[3:7]
        pitch = 2 * np.arcsin(np.clip(2 * (quat[0]*quat[2] - quat[3]*quat[1]), -1, 1))
        roll  = 2 * np.arcsin(np.clip(2 * (quat[0]*quat[1] + quat[2]*quat[3]), -1, 1))

        alive_bonus    = 1.0
        forward_reward = (posafter - posbefore) / self.dt
        ctrl_cost      = 1e-3 * np.square(a).sum()
        height_reward  = -2.0 * abs(height - 0.79)
        upright_reward = -1.0 * (abs(pitch) + abs(roll))

        reward = forward_reward + alive_bonus + height_reward + upright_reward - ctrl_cost

        height_ok = (height > 0.4 and height < 1.2)
        pitch_ok  = abs(pitch) < 0.8
        roll_ok   = abs(roll)  < 0.8
        done      = not (height_ok and pitch_ok and roll_ok)
        reward = forward_reward + alive_bonus + height_reward + upright_reward - ctrl_cost

        height_ok = (height > 0.4 and height < 1.2)
        pitch_ok  = abs(pitch) < 0.8
        roll_ok   = abs(roll)  < 0.8
        done      = not (height_ok and pitch_ok and roll_ok)

        if not hasattr(self, '_episode_step'):
            self._episode_step  = 0
            self._episode_return = 0
        self._episode_step   += 1
        self._episode_return += reward

        info = {
            # FIX 1 — report the correct per-robot name so TrainMeter can group correctly
            'name':           self._robot_name,
            'forward_reward': forward_reward,
            'ctrl_cost':      ctrl_cost,
            'height':         height,
            'height_reward':  height_reward,
            'upright_reward': upright_reward,
            'pitch':          pitch,
            'roll':           roll,
        }

        if done or self._episode_step >= 1000:
            info['episode'] = {'r': self._episode_return, 'l': self._episode_step}
            if done:
                self._episode_step  = 0
                self._episode_return = 0

        return self._get_obs(), reward, done, info

    def reset_model(self):
        self._episode_step  = 0
        self._episode_return = 0
        qpos = self.init_qpos + self.np_random.uniform(low=-0.01, high=0.01,
                                                        size=self.model.nq)
        qvel = self.init_qvel + self.np_random.uniform(low=-0.01, high=0.01,
                                                        size=self.model.nv)
        qpos[2]   = self.init_qpos[2] + self.np_random.uniform(low=-0.02, high=0.02)
        qpos[3:7] = qpos[3:7] / np.linalg.norm(qpos[3:7])
        self.set_state(qpos, qvel)
        return self._get_obs()

    def viewer_setup(self):
        self.viewer.cam.trackbodyid = 1
        self.viewer.cam.distance    = self.model.stat.extent * 2.0
        # FIX 4 — camera look-at height tracks the robot's natural standing height
        self.viewer.cam.lookat[2]   = self._init_height
        self.viewer.cam.elevation   = -20


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