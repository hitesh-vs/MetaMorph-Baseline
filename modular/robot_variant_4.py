"""
modular/g1_12dof.py

Generalised ModularEnv for floating-base robots (G1 family).

Fixes vs original:
  FIX 1  robot name from XML filename        (not hardcoded 'g1_12dof')
  FIX 2  num_limbs/num_joints from model     (not hardcoded 13/12)
  FIX 3  root body from model/XML            (not hardcoded 'pelvis')
  FIX 4  height target from init_qpos        (not hardcoded 0.79 m)

Reward overhaul — ported from legged_gym G1 config:
  The original reward had alive_bonus=1.0 dominating, causing the robot
  to learn "stand still" as a local optimum, then "lean forward" as the
  next local optimum after we upweighted forward_vel.

  The legged_gym reward uses 5 gait-forcing terms that together make
  actual stepping strictly better than leaning:
    lin_vel_z       penalise vertical bouncing during lean
    ang_vel_xy      penalise pitch/roll rate (leaning = large pitch_rate)
    hip_pos         penalise hip_roll/yaw deviation (lateral sway)
    contact         reward feet matching expected gait phase
    feet_swing      penalise feet not lifting high enough when swinging
    contact_no_vel  penalise foot sliding while planted

  Reward component map  (legged_gym name → our implementation):
  All coefficients pre-multiplied by DT=0.02 to match legged_gym
  calibration (legged_gym applies scale×dt in compute_reward).
  Total reward is clipped to zero (only_positive_rewards).
  sin/cos gait phase is included in observation so robot can see clock.

    tracking_lin_vel  → 1.0*DT * exp(-||forward_vel - target_vel||^2 / 0.25)
    lin_vel_z         → -2.0*DT  * root_linvel_z^2
    ang_vel_xy        → -0.05*DT * (pitch_rate^2 + roll_rate^2)
    orientation       → -1.0*DT  * (gravity_x^2 + gravity_y^2)   [tilt]
    base_height       → -10.0*DT * (height - target_height)^2
    dof_vel           → -1e-3*DT * sum(qvel[7:]^2)
    dof_acc           → -2.5e-7*DT * sum(((qvel-last_qvel)/dt)^2)
    action_rate       → -0.01*DT * sum((action - last_action)^2)
    hip_pos           → -1.0*DT  * sum(hip_roll^2 + hip_yaw^2)
    contact           → +0.18*DT * gait_phase_match
    feet_swing        → -20.0*DT * (foot_z - 0.08)^2 * swinging
    contact_no_vel    → -0.2*DT  * foot_vel^2 * in_contact
    alive             → +0.15*DT per step
"""

from gym import utils
from gym.envs.mujoco import mujoco_env
import numpy as np
import os
import xml.etree.ElementTree as ET

from modular.utils import *
from modular.wrappers import *
from metamorph.config import cfg
from graphs.parser import MujocoGraphParser


# ── Gait clock ────────────────────────────────────────────────────────────────
# Bipedal walking: period=0.8s, left and right 180° out of phase.
# A foot is in "stance phase" when phase < 0.55, swing phase otherwise.
GAIT_PERIOD  = 0.8   # seconds
GAIT_OFFSET  = 0.5   # right leg offset (half period = alternating gait)
STANCE_THRESH = 0.55  # fraction of period that is stance

# Foot body names — used for contact and swing-height rewards
FOOT_BODIES = ('left_ankle_roll_link', 'right_ankle_roll_link')

# Hip roll/yaw qpos indices (0-based within the full qpos vector)
# Free joint occupies qpos[0:7].  Actuated joints start at qpos[7].
# G1 joint order: L_hip_pitch(0) L_hip_roll(1) L_hip_yaw(2) L_knee(3)
#                 L_ankle_pitch(4) L_ankle_roll(5)
#                 R_hip_pitch(6)  R_hip_roll(7)  R_hip_yaw(8) R_knee(9)
#                 R_ankle_pitch(10) R_ankle_roll(11)
# → hip_roll indices in qpos: 7+1=8, 7+7=14
# → hip_yaw  indices in qpos: 7+2=9, 7+8=15
HIP_ROLL_YAW_QPOS = (8, 9, 14, 15)   # left roll, left yaw, right roll, right yaw

# Simulation dt used to scale reward coefficients to match legged_gym calibration.
# legged_gym multiplies all reward scales by dt inside _prepare_reward_function;
# we pre-multiply here so magnitudes are identical.
DT = 0.02

# Target forward velocity (m/s).  Robot gets maximum reward when it matches this.
TARGET_FORWARD_VEL = 1.0


class ModularEnv(mujoco_env.MujocoEnv, utils.EzPickle):

    def __init__(self, xml):
        self.xml = xml

        # FIX 1 — name from filename
        self._robot_name = os.path.splitext(os.path.basename(xml))[0]
        if self._robot_name.endswith('_stripped'):
            self._robot_name = self._robot_name[:-9]

        self.graph_encoding = cfg.MODEL.GRAPH_ENCODING

        # Placeholders — filled after MujocoEnv.__init__
        self.edges            = np.array([])
        self.traversals       = np.array([])
        self.SWAT_RE          = np.zeros([cfg.MODEL.MAX_LIMBS, cfg.MODEL.MAX_LIMBS, 3])
        self.context_features = np.array([])

        self._graph_node_features = None
        self._graph_A_norm        = None
        self._graph_built         = False

        # FIX 3/4 — must exist before MujocoEnv.__init__ calls step()
        self._root_body   = self._parse_root_body(xml)
        self._init_height = self._parse_init_height(xml)

        # Reward state buffers — must exist before first step()
        self._last_qvel   = None   # for dof_acc
        self._last_action = None   # for action_rate
        self._episode_step   = 0
        self._episode_return = 0.0
        self._episode_time   = 0.0  # seconds, for gait phase

        # Safe empty values so step() during MujocoEnv.__init__
        # returns 0 for contact/gait terms instead of crashing.
        # Populated with real values after MujocoEnv.__init__ completes.
        self._foot_body_ids     = []   # list of int body ids for feet
        self._hip_roll_yaw_qpos = []   # list of qpos indices for hip roll/yaw

        mujoco_env.MujocoEnv.__init__(self, xml, 4)
        utils.EzPickle.__init__(self)

        # FIX 2 — counts from model
        num_limbs  = len(self.model.body_names) - 1
        num_joints = self.sim.model.nu
        self.metadata['num_limbs']  = num_limbs
        self.metadata['num_joints'] = num_joints

        # FIX 3 — confirmed from model
        self._root_body = self.model.body_names[1]

        # FIX 4 — precise value from default qpos
        self._init_height = float(self.init_qpos[2])

        self.full_limb_names  = list(self.model.body_names[1:])
        self.agent_limb_names = list(self.model.body_names[1:])

        # Cache foot body indices for fast contact lookup
        self._foot_body_ids = []
        for fname in FOOT_BODIES:
            try:
                self._foot_body_ids.append(
                    self.sim.model.body_name2id(fname))
            except Exception:
                pass  # robot variant may have different foot names

        # Hip roll/yaw qpos indices — offset by 7 (free joint) from dof indices
        # Falls back to HIP_ROLL_YAW_QPOS constant; safe if list stays empty.
        self._hip_roll_yaw_qpos = list(HIP_ROLL_YAW_QPOS)

        self._build_graph_structure()
        self._compute_context_encoding()

        if self.graph_encoding != "none":
            self._init_graph_data()

        self._graph_built = True

    # ──────────────────────────────────────────────────────────────────────
    # Pre-init XML helpers (called before MujocoEnv.__init__)
    # ──────────────────────────────────────────────────────────────────────

    @staticmethod
    def _parse_root_body(xml_path: str) -> str:
        try:
            root = ET.parse(xml_path).getroot()
            wb   = root.find('worldbody')
            if wb is not None:
                b = wb.find('body')
                if b is not None:
                    return b.attrib.get('name', 'pelvis')
        except Exception:
            pass
        return 'pelvis'

    @staticmethod
    def _parse_init_height(xml_path: str) -> float:
        try:
            root = ET.parse(xml_path).getroot()
            wb   = root.find('worldbody')
            if wb is not None:
                b = wb.find('body')
                if b is not None:
                    pos_str = b.attrib.get('pos', '0 0 0.8')
                    z = float(pos_str.strip().split()[2])
                    return z if z > 0.01 else 0.8
        except Exception:
            pass
        return 0.8

    # ──────────────────────────────────────────────────────────────────────
    # Graph init
    # ──────────────────────────────────────────────────────────────────────

    def _init_graph_data(self):
        parser = MujocoGraphParser(self.xml)
        self._graph_node_features = parser.get_features(self.graph_encoding)
        self._graph_A_norm        = parser.normalized_adjacency()
        print(f"[{self._robot_name}|{self.graph_encoding}] "
              f"Graph: {parser.N} nodes, features {self._graph_node_features.shape}")

    # ──────────────────────────────────────────────────────────────────────
    # Kinematic graph (SWAT / context)
    # ──────────────────────────────────────────────────────────────────────

    def _build_graph_structure(self):
        num_limbs  = self.metadata['num_limbs']
        num_joints = self.metadata['num_joints']

        joint_to   = self.sim.model.jnt_bodyid[1:].copy() - 1
        parents_id = self.sim.model.body_parentid.copy()
        joint_from = np.array([parents_id[c + 1] - 1 for c in joint_to])
        joint_to   = joint_to[:num_joints]
        joint_from = joint_from[:num_joints]

        self.edges = np.vstack((joint_to, joint_from)).T.flatten().astype(np.int32)

        parents = [-1] * num_limbs
        for i in range(len(joint_to)):
            ci, pi = int(joint_to[i]), int(joint_from[i])
            if 0 <= ci < num_limbs and pi >= -1:
                parents[ci] = pi

        self.traversals = self._get_traversal(parents)
        self.SWAT_RE = (
            self._get_graph_dict(parents)
            if cfg.MODEL.TRANSFORMER.USE_SWAT_RE
            else np.zeros([cfg.MODEL.MAX_LIMBS, cfg.MODEL.MAX_LIMBS, 3])
        )

    def _get_traversal(self, parents):
        root = parents.index(-1) if -1 in parents else 0
        kids = {i: [] for i in range(len(parents))}
        for i, p in enumerate(parents):
            if p >= 0: kids[p].append(i)
        order = []
        def dfs(n):
            order.append(n)
            for c in sorted(kids[n]): dfs(c)
        dfs(root)
        return np.array(order, dtype=np.int32)

    def _get_graph_dict(self, parents):
        n  = len(parents)
        gd = np.zeros([cfg.MODEL.MAX_LIMBS, cfg.MODEL.MAX_LIMBS, 3])
        kids = {i: [] for i in range(n)}
        for i, p in enumerate(parents):
            if p >= 0: kids[p].append(i)
        for i in range(n):
            for j in range(n):
                if   i == j:          gd[i, j] = [1, 0, 0]
                elif parents[i] == j: gd[i, j] = [0, 1, 0]
                elif j in kids[i]:    gd[i, j] = [0, 0, 1]
        return gd

    # ──────────────────────────────────────────────────────────────────────
    # Context encoding
    # ──────────────────────────────────────────────────────────────────────

    def _compute_normalization_bounds(self):
        num_limbs  = self.metadata['num_limbs']
        num_joints = self.metadata['num_joints']
        body_idxs  = list(range(1, num_limbs + 1))

        def bnd(a): return a.min(axis=0, keepdims=True), a.max(axis=0, keepdims=True)

        lb = {}
        lb['body_pos']   = bnd(self.sim.model.body_pos[body_idxs])
        lb['body_ipos']  = bnd(self.sim.model.body_ipos[body_idxs])
        lb['body_iquat'] = np.array([[-1,-1,-1,-1]]), np.array([[1,1,1,1]])
        lb['geom_quat']  = np.array([[-1,-1,-1,-1]]), np.array([[1,1,1,1]])
        mass = self.sim.model.body_mass[body_idxs]
        lb['body_mass']  = np.array([[mass.min()]]), np.array([[mass.max()]])

        gi = [next((g for g in range(self.sim.model.ngeom)
                    if self.sim.model.geom_bodyid[g] == bi), None)
              for bi in body_idxs]
        gi = [g for g in gi if g is not None]
        if len(gi) == num_limbs:
            lb['body_shape']    = bnd(self.sim.model.geom_size[gi, :2])
            lb['body_friction'] = bnd(self.sim.model.geom_friction[gi, 0:1])
        else:
            lb['body_shape']    = np.array([[0.01,0.01]]), np.array([[0.20,0.20]])
            lb['body_friction'] = np.array([[0.5]]),       np.array([[1.5]])

        jb = {}
        jb['jnt_pos']     = bnd(self.sim.model.jnt_pos[1:num_joints+1])
        jb['joint_range'] = bnd(self.sim.model.jnt_range[1:num_joints+1])
        jb['joint_axis']  = bnd(self.sim.model.jnt_axis[1:num_joints+1])
        jb['gear']        = bnd(self.sim.model.actuator_gear[:num_joints, 0:1])
        jb['armature']    = bnd(self.sim.model.dof_armature[6:6+num_joints, np.newaxis])
        jb['damping']     = bnd(self.sim.model.dof_damping[6:6+num_joints,  np.newaxis])
        return lb, jb

    @staticmethod
    def _norm(arr, lo, hi):
        return np.clip(-1.0 + 2.0*(arr - lo)/(hi - lo + 1e-8), -1.0, 1.0)

    def _compute_context_encoding(self):
        num_limbs  = self.metadata['num_limbs']
        num_joints = self.metadata['num_joints']
        body_idxs  = list(range(1, num_limbs + 1))
        lb, jb     = self._compute_normalization_bounds()

        cl = {
            'body_pos':   self.sim.model.body_pos[body_idxs].copy(),
            'body_ipos':  self.sim.model.body_ipos[body_idxs].copy(),
            'body_iquat': self.sim.model.body_iquat[body_idxs].copy(),
            'body_mass':  self.sim.model.body_mass[body_idxs].copy()[:,np.newaxis],
        }
        gi = [next((g for g in range(self.sim.model.ngeom)
                    if self.sim.model.geom_bodyid[g] == bi), None)
              for bi in body_idxs]
        gi = [g for g in gi if g is not None]
        if len(gi) == num_limbs:
            cl['geom_quat']     = self.sim.model.geom_quat[gi].copy()
            cl['body_shape']    = self.sim.model.geom_size[gi, :2].copy()
            cl['body_friction'] = self.sim.model.geom_friction[gi, 0:1].copy()
        else:
            cl['geom_quat']     = np.tile([1,0,0,0], (num_limbs,1)).astype(np.float32)
            cl['body_shape']    = np.zeros((num_limbs, 2), np.float32)
            cl['body_friction'] = np.ones((num_limbs, 1), np.float32)
        for k in cl:
            if k in lb: cl[k] = self._norm(cl[k], *lb[k])

        cj = {
            'jnt_pos':     self.sim.model.jnt_pos[1:num_joints+1].copy(),
            'joint_range': self.sim.model.jnt_range[1:num_joints+1].copy(),
            'joint_axis':  self.sim.model.jnt_axis[1:num_joints+1].copy(),
            'gear':        self.sim.model.actuator_gear[:num_joints, 0:1].copy(),
            'armature':    self.sim.model.dof_armature[6:6+num_joints].copy()[:,np.newaxis],
            'damping':     self.sim.model.dof_damping[6:6+num_joints].copy()[:,np.newaxis],
        }
        for k in cj:
            if k in jb: cj[k] = self._norm(cj[k], *jb[k])

        lf = self._select_ctx(cl, cfg.MODEL.CONTEXT_OBS_TYPES)
        jf = self._select_ctx(cj, cfg.MODEL.CONTEXT_OBS_TYPES)
        self.context_features = self._combine_ctx(lf, jf)

    def _select_ctx(self, d, keys):
        parts = [d[k] for k in keys if k in d]
        return np.hstack(parts) if parts else np.array([])

    def _combine_ctx(self, lobs, jobs):
        num_limbs  = self.metadata['num_limbs']
        num_joints = self.metadata['num_joints']
        if lobs is None or not len(lobs): return jobs.flatten() if (jobs is not None and len(jobs)) else np.array([])
        if jobs is None or not len(jobs): return lobs.flatten()
        jsz = jobs.shape[1]
        jp  = np.zeros((num_limbs, jsz * 2))
        ja  = np.zeros(num_limbs, dtype=int)
        if len(self.edges):
            for i, li in enumerate(self.edges[::2]):
                if 0 <= li < num_limbs and i < num_joints:
                    s = ja[li]*jsz; e = s+jsz
                    if e <= jsz*2: jp[li, s:e] = jobs[i]; ja[li] += 1
        return np.hstack((lobs, jp)).flatten()

    # ──────────────────────────────────────────────────────────────────────
    # Observation
    # ──────────────────────────────────────────────────────────────────────

    def _get_obs_per_limb(self, b):
        root_x = self.data.get_body_xpos(self._root_body)[0]
        xpos   = self.data.get_body_xpos(b).copy(); xpos[0] -= root_x
        xvelp  = np.clip(self.data.get_body_xvelp(b), -10, 10)
        xvelr  = self.data.get_body_xvelr(b)
        expmap = quat2expmap(self.data.get_body_xquat(b))

        if b == self._root_body:
            angle, joint_range = 0.0, np.array([0.0, 0.0])
        else:
            try:
                bid     = self.sim.model.body_name2id(b)
                jnt_adr = self.sim.model.body_jntadr[bid]
                if jnt_adr == -1:
                    angle, joint_range = 0.0, np.array([0.0, 0.0])
                else:
                    qa          = self.sim.model.jnt_qposadr[jnt_adr]
                    angle       = self.data.qpos[qa]
                    joint_range = self.sim.model.jnt_range[jnt_adr].copy()
                    span        = joint_range[1] - joint_range[0]
                    angle       = (angle - joint_range[0]) / span if span > 1e-8 else 0.0
                    joint_range = joint_range / np.pi
            except Exception:
                angle, joint_range = 0.0, np.array([0.0, 0.0])

        # Gait phase signal — robot must see the clock to learn contact reward.
        # sin/cos encoding avoids discontinuity at phase wraparound.
        # Non-zero only on root body; all other limbs get zeros so per-limb
        # vector length is uniform (required by transformer padding).
        if b == self._root_body:
            phase     = (self._episode_time % GAIT_PERIOD) / GAIT_PERIOD
            sin_phase = np.array([np.sin(2 * np.pi * phase)])
            cos_phase = np.array([np.cos(2 * np.pi * phase)])
        else:
            sin_phase = np.array([0.0])
            cos_phase = np.array([0.0])

        if self.graph_encoding == "none":
            n = b.lower()
            if   "hip_pitch" in n: ltv = np.array((1,0,0,0,0,0))
            elif "hip_roll"  in n: ltv = np.array((0,1,0,0,0,0))
            elif "hip_yaw"   in n: ltv = np.array((0,0,1,0,0,0))
            elif "knee"      in n: ltv = np.array((0,0,0,1,0,0))
            elif "ankle"     in n: ltv = np.array((0,0,0,0,1,0))
            else:                  ltv = np.array((0,0,0,0,0,0))
            return np.concatenate([xpos, xvelp, xvelr, expmap, ltv, [angle], joint_range,
                                   sin_phase, cos_phase])

        return np.concatenate([xpos, xvelp, xvelr, expmap, [angle], joint_range,
                               sin_phase, cos_phase])

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
    # Reward helpers (ported from legged_gym G1)
    # ──────────────────────────────────────────────────────────────────────

    def _foot_contact_forces(self):
        """
        Returns array of shape (num_feet,) with vertical contact force per foot.
        cfrc_ext shape: (nbody, 6) — [0:3]=torque, [3:6]=force in world frame.
        Index 5 = Fz (vertical).
        Returns empty array if foot ids not yet populated (safe for init step).
        """
        if not self._foot_body_ids:
            return np.array([])
        forces = []
        for bid in self._foot_body_ids:
            fz = abs(self.sim.data.cfrc_ext[bid, 5])
            forces.append(fz)
        return np.array(forces)

    def _foot_positions(self):
        """Returns (num_feet, 3) foot positions in world frame."""
        pos = []
        for fname in FOOT_BODIES:
            try:
                pos.append(self.data.get_body_xpos(fname).copy())
            except Exception:
                pos.append(np.zeros(3))
        return np.array(pos)

    def _foot_velocities(self):
        """Returns (num_feet, 3) foot linear velocities."""
        vel = []
        for fname in FOOT_BODIES:
            try:
                vel.append(np.clip(self.data.get_body_xvelp(fname), -10, 10))
            except Exception:
                vel.append(np.zeros(3))
        return np.array(vel)

    def _gait_phase(self, t):
        """
        Returns (is_stance_left, is_stance_right) booleans at time t.
        Left leg: phase = (t % period) / period
        Right leg: offset by GAIT_OFFSET
        Stance when phase < STANCE_THRESH.
        """
        phase_left  = (t % GAIT_PERIOD) / GAIT_PERIOD
        phase_right = (phase_left + GAIT_OFFSET) % 1.0
        return phase_left < STANCE_THRESH, phase_right < STANCE_THRESH

    def _rew_tracking_lin_vel(self, forward_vel):
        """
        Gaussian reward centered on TARGET_FORWARD_VEL.
        exp(-error^2 / sigma) so maximum=1 when vel=target, decays smoothly.
        Equivalent to legged_gym tracking_lin_vel with sigma=0.25.
        """
        error = (forward_vel - TARGET_FORWARD_VEL) ** 2
        return 1.0 * DT * float(np.exp(-error / 0.25))

    def _rew_lin_vel_z(self, root_velz):
        """Penalise vertical velocity of root — prevents hopping/leaning bounce."""
        return -2.0 * DT * float(root_velz ** 2)

    def _rew_ang_vel_xy(self, pitch_rate, roll_rate):
        """Penalise pitch and roll rate — leaning forward generates large pitch_rate."""
        return -0.05 * DT * float(pitch_rate**2 + roll_rate**2)

    def _rew_orientation(self, pitch, roll):
        """
        Penalise tilt using projected gravity proxy.
        projected_gravity_x ≈ sin(pitch), projected_gravity_y ≈ sin(roll).
        """
        return -1.0 * DT * float(np.sin(pitch)**2 + np.sin(roll)**2)

    def _rew_base_height(self, height):
        """Penalise deviation from standing height. Large scale (-10) is intentional."""
        return -10.0 * DT * float((height - self._init_height) ** 2)

    def _rew_dof_vel(self):
        """Penalise excessive joint velocities (smooth motion)."""
        qvel_joints = self.data.qvel[6:]   # skip 6-DOF root vel
        return -1e-3 * DT * float(np.square(qvel_joints).sum())

    def _rew_dof_acc(self, last_qvel):
        """Penalise joint accelerations (smooth motion)."""
        if last_qvel is None:
            return 0.0
        qvel_joints      = self.data.qvel[6:]
        last_qvel_joints = last_qvel[6:]
        acc = (qvel_joints - last_qvel_joints) / self.dt
        return -2.5e-7 * DT * float(np.square(acc).sum())

    def _rew_action_rate(self, action, last_action):
        """Penalise large action changes between steps (smooth policy)."""
        if last_action is None:
            return 0.0
        return -0.01 * DT * float(np.square(action - last_action).sum())

    def _rew_hip_pos(self):
        """
        Penalise hip_roll and hip_yaw deviation from zero.
        Stops lateral sway / sideways lean strategy.
        qpos indices: free joint = qpos[0:7], joints at qpos[7+]
        """
        if not self._hip_roll_yaw_qpos:
            return 0.0
        hip_angles = self.data.qpos[self._hip_roll_yaw_qpos]
        return -1.0 * DT * float(np.square(hip_angles).sum())

    def _rew_contact(self, t):
        """
        Reward feet matching the expected gait phase.
        Translated from G1Robot._reward_contact:
          +1 per foot that matches (stance when should be stance, swing when swing)
        scale=0.18
        """
        contact_forces = self._foot_contact_forces()
        in_contact = contact_forces > 1.0   # bool array (num_feet,)

        is_stance = list(self._gait_phase(t))   # [left_stance, right_stance]
        n_feet = min(len(in_contact), len(is_stance))

        score = 0.0
        for i in range(n_feet):
            # reward = NOT (contact XOR expected_stance)
            # i.e. +1 if foot state matches gait clock
            if not (bool(in_contact[i]) ^ bool(is_stance[i])):
                score += 1.0
        return 0.18 * DT * score

    def _rew_feet_swing_height(self, t):
        """
        Penalise feet not lifting high enough during swing phase.
        Target swing height = 0.08 m above ground.
        Only penalised during swing (not contact).
        Translated from G1Robot._reward_feet_swing_height.
        scale=-20.0 (already included in return value)
        """
        contact_forces = self._foot_contact_forces()
        in_contact     = contact_forces > 1.0
        foot_pos       = self._foot_positions()   # (num_feet, 3)

        penalty = 0.0
        for i in range(min(len(foot_pos), len(in_contact))):
            if not in_contact[i]:   # only during swing
                foot_z     = float(foot_pos[i, 2])
                height_err = (foot_z - 0.08) ** 2
                penalty   += height_err
        return -20.0 * DT * penalty

    def _rew_contact_no_vel(self, t):
        """
        Penalise foot sliding while planted.
        Feet in contact should have zero velocity.
        Translated from G1Robot._reward_contact_no_vel.
        scale=-0.2
        """
        contact_forces = self._foot_contact_forces()
        in_contact     = contact_forces > 1.0
        foot_vels      = self._foot_velocities()   # (num_feet, 3)

        penalty = 0.0
        for i in range(min(len(foot_vels), len(in_contact))):
            if in_contact[i]:
                penalty += float(np.square(foot_vels[i]).sum())
        return -0.2 * DT * penalty

    # ──────────────────────────────────────────────────────────────────────
    # Step
    # ──────────────────────────────────────────────────────────────────────

    def step(self, a):
        # Save state before physics step
        last_qvel   = self.data.qvel.copy() if self._last_qvel is not None or self._episode_step > 0 else None
        last_action = self._last_action.copy() if self._last_action is not None else None

        posbefore = self.sim.data.qpos[0]
        self.do_simulation(a, self.frame_skip)
        posafter = self.sim.data.qpos[0]

        # ── Core state ───────────────────────────────────────────────────
        height      = float(self.data.get_body_xpos(self._root_body)[2])
        quat        = self.data.get_body_xquat(self._root_body)
        pitch       = float(2 * np.arcsin(np.clip(2*(quat[0]*quat[2] - quat[3]*quat[1]), -1, 1)))
        roll        = float(2 * np.arcsin(np.clip(2*(quat[0]*quat[1] + quat[2]*quat[3]), -1, 1)))

        # Root angular velocity (world frame) for ang_vel_xy reward
        root_angvel = self.data.get_body_xvelr(self._root_body)   # (3,) [roll_rate, pitch_rate, yaw_rate]
        pitch_rate  = float(root_angvel[1])
        roll_rate   = float(root_angvel[0])

        # Root vertical linear velocity for lin_vel_z reward
        root_linvel = self.data.get_body_xvelp(self._root_body)   # (3,) world frame
        root_velz   = float(root_linvel[2])

        forward_vel = (posafter - posbefore) / self.dt

        # Episode time for gait phase clock
        self._episode_time += self.dt

        # ── Reward terms ─────────────────────────────────────────────────
        r_tracking   = self._rew_tracking_lin_vel(forward_vel)     # +, max=1.0
        r_lin_vel_z  = self._rew_lin_vel_z(root_velz)              # -, stops bouncing
        r_ang_vel_xy = self._rew_ang_vel_xy(pitch_rate, roll_rate)  # -, stops lean
        r_orientation = self._rew_orientation(pitch, roll)          # -, tilt penalty
        r_height     = self._rew_base_height(height)                # -, height penalty
        r_dof_vel    = self._rew_dof_vel()                          # -, smooth joints
        r_dof_acc    = self._rew_dof_acc(last_qvel)                 # -, smooth joints
        r_action_rate = self._rew_action_rate(a, last_action)       # -, smooth policy
        r_hip        = self._rew_hip_pos()                          # -, no lateral sway
        r_contact    = self._rew_contact(self._episode_time)        # +, gait rhythm
        r_swing      = self._rew_feet_swing_height(self._episode_time)  # -, foot lift
        r_no_vel     = self._rew_contact_no_vel(self._episode_time) # -, no sliding
        r_alive      = 0.15 * DT                                     # small survival bonus

        reward = (r_tracking + r_lin_vel_z + r_ang_vel_xy + r_orientation
                + r_height + r_dof_vel + r_dof_acc + r_action_rate
                + r_hip + r_contact + r_swing + r_no_vel + r_alive)
        # Clip to zero: equivalent to legged_gym only_positive_rewards.
        # Prevents large penalty terms collapsing the policy during
        # early exploration when the robot has not yet learned to stand.
        

        # ── Termination (FIX 4: robot-relative bounds) ───────────────────
        height_ok = (height > 0.50 * self._init_height and
                     height < 1.50 * self._init_height)
        done = not (height_ok and abs(pitch) < 0.8 and abs(roll) < 0.8)

        # ── Bookkeeping ───────────────────────────────────────────────────
        self._last_qvel   = self.data.qvel.copy()
        self._last_action = a.copy()
        self._episode_step   += 1
        self._episode_return += reward

        # FIX 1: name from filename
        info = {
            'name':           self._robot_name,
            'forward_vel':    forward_vel,
            'height':         height,
            'pitch':          pitch,
            'roll':           roll,
            # individual reward components for debugging
            'rew/tracking':   r_tracking,
            'rew/lin_vel_z':  r_lin_vel_z,
            'rew/ang_vel_xy': r_ang_vel_xy,
            'rew/orientation': r_orientation,
            'rew/height':     r_height,
            'rew/hip':        r_hip,
            'rew/contact':    r_contact,
            'rew/swing':      r_swing,
            'rew/no_vel':     r_no_vel,
        }

        if done or self._episode_step >= 1000:
            info['episode'] = {'r': self._episode_return, 'l': self._episode_step}
            self._episode_step   = 0
            self._episode_return = 0.0
            self._episode_time   = 0.0
            self._last_qvel      = None
            self._last_action    = None

        return self._get_obs(), reward, done, info

    # ──────────────────────────────────────────────────────────────────────
    # Reset / Viewer
    # ──────────────────────────────────────────────────────────────────────

    def reset_model(self):
        self._episode_step   = 0
        self._episode_return = 0.0
        self._episode_time   = 0.0
        self._last_qvel      = None
        self._last_action    = None

        qpos = self.init_qpos + self.np_random.uniform(-0.01, 0.01, self.model.nq)
        qvel = self.init_qvel + self.np_random.uniform(-0.01, 0.01, self.model.nv)
        qpos[2]   = self.init_qpos[2] + self.np_random.uniform(-0.02, 0.02)
        qpos[3:7] = qpos[3:7] / np.linalg.norm(qpos[3:7])
        self.set_state(qpos, qvel)
        return self._get_obs()

    def viewer_setup(self):
        self.viewer.cam.trackbodyid = 1
        self.viewer.cam.distance    = self.model.stat.extent * 2.0
        self.viewer.cam.lookat[2]   = self._init_height
        self.viewer.cam.elevation   = -20


def make_env(xml):
    env = ModularEnv(xml)
    print(f"Made env for {env._robot_name} | "
          f"encoding={cfg.MODEL.GRAPH_ENCODING} | "
          f"limbs={env.metadata['num_limbs']} "
          f"joints={env.metadata['num_joints']} | "
          f"root={env._root_body} "
          f"init_height={env._init_height:.3f}m")

    if cfg.MODEL.MLP.CONSISTENT_PADDING:
        env = ConsistentModularObservationPadding(env)
        env = ConsistentModularActionPadding(env)
    else:
        env = ModularObservationPadding(env)
        env = ModularActionPadding(env)

    return env