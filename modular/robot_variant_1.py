"""
G1 12DOF MuJoCo Environment — matched to G1RoughCfg / G1Robot (IsaacGym)

Key implementation decisions:
  - PD position control:  torque = Kp*(target - pos) - Kd*vel
                          actions are OFFSETS from default_joint_angles (scale 0.25)
  - Decimation = 4:       4 sim steps per policy step (matching IsaacGym config)
  - Observation = 47 dim: ang_vel(3) + proj_gravity(3) + commands(3) +
                          dof_pos_offset(12) + dof_vel(12) + actions(12) +
                          sin_phase(1) + cos_phase(1)
  - Gait phase:           period=0.8s, left offset=0, right offset=0.5
  - Termination:          pelvis contact OR |pitch|>1.0 OR |roll|>0.8 OR height OOB
  - Rewards:              exact scales from G1RoughCfg, all multiplied by dt internally
  - Contact detection:    sim.data.cfrc_ext[body_id, 5] (precomputed Fz, no mujoco_py)
                          cfrc_ext shape (nbody, 6): [0:3]=torque [3:6]=force, index 5=Fz
"""

from gym import utils
from gym.envs.mujoco import mujoco_env
import numpy as np
import os
import re

from modular.utils import *
from modular.wrappers import *
from metamorph.config import cfg

from graphs.parser import MujocoGraphParser


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def quat_to_euler(quat):
    """[w, x, y, z] → (roll, pitch, yaw)."""
    w, x, y, z = quat
    roll  = np.arctan2(2*(w*x + y*z), 1 - 2*(x*x + y*y))
    pitch = np.arcsin(np.clip(2*(w*y - z*x), -1, 1))
    yaw   = np.arctan2(2*(w*z + x*y), 1 - 2*(y*y + z*z))
    return roll, pitch, yaw


def quat_rotate_inverse(q, v):
    """
    Rotate vector v by the inverse of quaternion q = [w, x, y, z].
    Returns v expressed in the body frame.
    """
    w, x, y, z = q
    # q_inv = [w, -x, -y, -z]  (unit quat conjugate)
    t  = 2.0 * np.cross(np.array([x, y, z]), v)
    return v - w * t + np.cross(np.array([x, y, z]), t)


# ---------------------------------------------------------------------------
# PD gains — mirroring G1RoughCfg.control
# ---------------------------------------------------------------------------

_KP = {
    'hip_yaw':   100.0,
    'hip_roll':  100.0,
    'hip_pitch': 100.0,
    'knee':      150.0,
    'ankle':      40.0,
}
_KD = {
    'hip_yaw':   2.0,
    'hip_roll':  2.0,
    'hip_pitch': 2.0,
    'knee':      4.0,
    'ankle':     2.0,
}
_ACTION_SCALE   = 0.25
_DECIMATION     = 4       # sim steps per policy step
_GAIT_PERIOD    = 0.8     # seconds
_GAIT_OFFSET    = 0.5     # right leg phase offset

# Joint order for the 12-DOF G1 (matches URDF / IsaacGym dof order)
_JOINT_NAMES = [
    'left_hip_yaw_joint',
    'left_hip_roll_joint',
    'left_hip_pitch_joint',
    'left_knee_joint',
    'left_ankle_pitch_joint',
    'left_ankle_roll_joint',
    'right_hip_yaw_joint',
    'right_hip_roll_joint',
    'right_hip_pitch_joint',
    'right_knee_joint',
    'right_ankle_pitch_joint',
    'right_ankle_roll_joint',
]

_DEFAULT_JOINT_ANGLES = {
    'left_hip_yaw_joint':    0.0,
    'left_hip_roll_joint':   0.0,
    'left_hip_pitch_joint': -0.1,
    'left_knee_joint':       0.3,
    'left_ankle_pitch_joint':-0.2,
    'left_ankle_roll_joint': 0.0,
    'right_hip_yaw_joint':   0.0,
    'right_hip_roll_joint':  0.0,
    'right_hip_pitch_joint':-0.1,
    'right_knee_joint':      0.3,
    'right_ankle_pitch_joint':-0.2,
    'right_ankle_roll_joint':0.0,
}

# Reward scales from G1RoughCfg (will be multiplied by dt in _compute_rewards)
_REWARD_SCALES = {
    'tracking_lin_vel':  1.0,
    'tracking_ang_vel':  0.5,
    'lin_vel_z':        -2.0,
    'ang_vel_xy':       -0.05,
    'orientation':      -1.0,
    'base_height':     -10.0,
    'dof_acc':         -2.5e-7,
    'dof_vel':         -1e-3,
    'action_rate':     -0.01,
    'dof_pos_limits':  -5.0,
    'alive':            0.15,
    'hip_pos':         -1.0,
    'contact_no_vel':  -0.2,
    'feet_swing_height':-20.0,
    'contact':          0.18,
}

# Soft DOF pos limit fraction (G1RoughCfg.rewards.soft_dof_pos_limit = 0.9)
_SOFT_DOF_LIMIT = 0.9
_BASE_HEIGHT_TARGET = 0.78
_TRACKING_SIGMA = 0.25   # exp(-err / sigma) tracking kernel


# ---------------------------------------------------------------------------
# Main environment
# ---------------------------------------------------------------------------

class ModularEnv(mujoco_env.MujocoEnv, utils.EzPickle):

    def __init__(self, xml):
        self.xml = xml

        base = os.path.basename(xml)
        base = os.path.splitext(base)[0]
        self._robot_name = re.sub(r'_stripped$', '', base)

        self.graph_encoding = cfg.MODEL.GRAPH_ENCODING

        self.full_limb_names = [
            'pelvis',
            'left_hip_pitch_link', 'left_hip_roll_link', 'left_hip_yaw_link',
            'left_knee_link',
            'left_ankle_pitch_link', 'left_ankle_roll_link',
            'right_hip_pitch_link', 'right_hip_roll_link', 'right_hip_yaw_link',
            'right_knee_link',
            'right_ankle_pitch_link', 'right_ankle_roll_link',
        ]

        # Graph placeholders
        self.edges        = np.array([])
        self.traversals   = np.array([])
        self.SWAT_RE      = np.zeros([cfg.MODEL.MAX_LIMBS, cfg.MODEL.MAX_LIMBS, 3])
        self.context_features = np.array([])
        self._graph_node_features = None
        self._graph_A_norm        = None
        self._graph_built         = False

        # ── Safe defaults before MujocoEnv.__init__ calls reset/step ────────
        # IMPORTANT: MujocoEnv.__init__ calls reset() which calls step() with a
        # zero action. Every attribute touched by step(), _compute_torques(),
        # _compute_rewards(), _check_termination(), and _get_obs() must exist
        # here with a safe no-op value before that call happens.

        self._root_body      = "pelvis"
        self._init_height    = 0.80
        self._episode_step   = 0
        self._episode_return = 0.0
        self._graph_built    = False

        # Commands (vel_x, vel_y, yaw_vel)
        self._commands = np.array([0.5, 0.0, 0.0])

        # PD gains — safe 12-joint defaults; rebuilt from model after init
        self._kp = np.ones(12) * 100.0
        self._kd = np.ones(12) * 2.0

        # Default joint positions for PD target — zeros means "don't move"
        # during the init step; rebuilt correctly after MujocoEnv.__init__
        self._default_dof_pos = np.zeros(12)

        # Soft DOF limit arrays — no penalty during init step
        self._dof_pos_limits_lower = np.full(12, -np.pi)
        self._dof_pos_limits_upper = np.full(12,  np.pi)

        # Last state buffers — zeros are safe for the init step
        self._last_action = np.zeros(12)
        self._last_qpos   = np.zeros(19)   # 7 free + 12 joints
        self._last_qvel   = np.zeros(18)   # 6 free + 12 joints

        # Contact body IDs — safe sentinels so cfrc_ext lookups don't crash.
        # MuJoCo body 0 is the world body; its cfrc_ext is always zero,
        # so using 0 here means all contact checks return False during init.
        self.left_foot_body_id  = 0
        self.right_foot_body_id = 0
        self.pelvis_body_id     = 0
        self.penalized_body_ids = []

        # MujocoEnv init (frame_skip=1 — we handle decimation manually in step)
        mujoco_env.MujocoEnv.__init__(self, xml, 1)
        utils.EzPickle.__init__(self)

        # ── Post-init setup ──────────────────────────────────────────────────
        self.metadata['num_limbs']  = len(self.model.body_names) - 1
        self.metadata['num_joints'] = self.sim.model.nu
        self.agent_limb_names       = self.model.body_names[1:]
        self._root_body             = self.model.body_names[1]
        self._init_height           = float(self.init_qpos[2])

        self._last_action = np.zeros(self.model.nu)
        self._last_qpos   = self.init_qpos.copy()
        self._last_qvel   = np.zeros(self.model.nv)

        # Build default angles and PD gains in joint order
        self._build_pd_gains()
        self._default_dof_pos = self._make_default_dof_pos()

        # Joint soft limits
        self._build_dof_limits()

        # Build graph / context
        self._build_graph_structure()
        self._compute_context_encoding()

        if self.graph_encoding != "none":
            self._init_graph_data()

        self._graph_built = True

        # Foot / pelvis body IDs for contact detection
        self._init_contact_bodies()

    # ─────────────────────────────────────────────────────────────────────────
    # Initialization helpers
    # ─────────────────────────────────────────────────────────────────────────

    def _build_pd_gains(self):
        """Build per-joint Kp / Kd arrays in MuJoCo actuator order."""
        kp = np.zeros(self.model.nu)
        kd = np.zeros(self.model.nu)
        for i, jname in enumerate(self.model.joint_names[1:]):   # skip free joint
            if i >= self.model.nu:
                break
            matched = False
            for key in _KP:
                if key in jname:
                    kp[i] = _KP[key]
                    kd[i] = _KD[key]
                    matched = True
                    break
            if not matched:
                kp[i] = 0.0
                kd[i] = 0.0
        self._kp = kp
        self._kd = kd

    def _make_default_dof_pos(self):
        """Default joint positions array in MuJoCo order."""
        pos = np.zeros(self.model.nu)
        for i, jname in enumerate(self.model.joint_names[1:]):
            if i >= self.model.nu:
                break
            if jname in _DEFAULT_JOINT_ANGLES:
                pos[i] = _DEFAULT_JOINT_ANGLES[jname]
        return pos

    def _build_dof_limits(self):
        """Store joint range limits for soft limit penalty."""
        # jnt_range shape: (nj, 2), skip index 0 (free joint)
        self._dof_pos_limits_lower = np.zeros(self.model.nu)
        self._dof_pos_limits_upper = np.zeros(self.model.nu)
        for i in range(self.model.nu):
            jnt_idx = i + 1   # +1 because joint 0 is free joint
            lo = self.sim.model.jnt_range[jnt_idx, 0]
            hi = self.sim.model.jnt_range[jnt_idx, 1]
            m  = (lo + hi) * 0.5
            r  = (hi - lo)
            self._dof_pos_limits_lower[i] = m - 0.5 * r * _SOFT_DOF_LIMIT
            self._dof_pos_limits_upper[i] = m + 0.5 * r * _SOFT_DOF_LIMIT

    def _init_contact_bodies(self):
        """Find foot and pelvis body IDs for contact queries."""
        self.left_foot_body_id  = None
        self.right_foot_body_id = None
        self.pelvis_body_id     = None
        self.penalized_body_ids = []   # hip, knee

        for bid, bname in enumerate(self.model.body_names):
            n = bname.lower()
            if 'pelvis' in n:
                self.pelvis_body_id = bid
            if 'left_ankle_roll' in n:
                self.left_foot_body_id = bid
            if 'right_ankle_roll' in n:
                self.right_foot_body_id = bid
            if 'hip' in n or 'knee' in n:
                self.penalized_body_ids.append(bid)

        # Fallbacks
        nb = len(self.model.body_names)
        if self.left_foot_body_id  is None: self.left_foot_body_id  = nb - 2
        if self.right_foot_body_id is None: self.right_foot_body_id = nb - 1
        if self.pelvis_body_id     is None: self.pelvis_body_id     = 1

    # ─────────────────────────────────────────────────────────────────────────
    # PD torque computation
    # ─────────────────────────────────────────────────────────────────────────

    def _compute_torques(self, action):
        """
        PD position control — matches IsaacGym 'P' control_type.
        action  : (nu,) position targets offset from default  [dimensionless]
        returns : (nu,) torques clipped to model effort limits
        """
        nu = len(action)
        action_scaled = np.clip(action, -100.0, 100.0) * _ACTION_SCALE

        # Guard: _default_dof_pos may be pre-init placeholder of wrong size
        default = self._default_dof_pos
        if len(default) != nu:
            default = np.zeros(nu)

        kp = self._kp if len(self._kp) == nu else np.ones(nu) * 100.0
        kd = self._kd if len(self._kd) == nu else np.ones(nu) * 2.0

        target_pos  = default + action_scaled
        current_pos = self.sim.data.qpos[7:7 + nu]
        current_vel = self.sim.data.qvel[6:6 + nu]

        torques = kp * (target_pos - current_pos) - kd * current_vel

        # Clip to URDF effort limits when available
        if hasattr(self.sim.model, 'actuator_forcerange'):
            lo = self.sim.model.actuator_forcerange[:nu, 0]
            hi = self.sim.model.actuator_forcerange[:nu, 1]
            torques = np.clip(torques, lo, hi)
        else:
            torques = np.clip(torques, -300.0, 300.0)

        return torques

    # ─────────────────────────────────────────────────────────────────────────
    # Contact helpers  (cfrc_ext — no mujoco_py dependency)
    # ─────────────────────────────────────────────────────────────────────────
    # sim.data.cfrc_ext shape: (nbody, 6)
    #   [0:3] = torque,  [3:6] = force  — all in world frame.
    # Index 5 = Fz (vertical contact force), same quantity IsaacGym reads via
    # net_contact_force_tensor[:, feet_indices, 2].

    def _contact_fz(self, body_id):
        """Vertical contact force on a body [N], from precomputed cfrc_ext."""
        return abs(float(self.sim.data.cfrc_ext[body_id, 5]))

    def _body_in_contact(self, body_id, threshold=1.0):
        """True if vertical contact force exceeds threshold [N]."""
        return self._contact_fz(body_id) > threshold

    def _get_foot_velocity(self, body_name):
        """3-D linear velocity of a named body."""
        try:
            return self.sim.data.get_body_xvelp(body_name).copy()
        except Exception:
            return np.zeros(3)

    def _get_foot_pos_z(self, body_name):
        """World-frame Z position of a named body."""
        try:
            return float(self.sim.data.get_body_xpos(body_name)[2])
        except Exception:
            return 0.0

    # ─────────────────────────────────────────────────────────────────────────
    # Gait phase
    # ─────────────────────────────────────────────────────────────────────────

    def _gait_phase(self):
        """
        Returns (phase_left, phase_right) both in [0, 1).
        Left leg is in stance when phase < 0.55, swing otherwise.
        Right leg is offset by 0.5 (anti-phase walk).
        """
        t = self._episode_step * self.dt   # MujocoEnv.dt = frame_skip * model.opt.timestep
        phase_left  = (t % _GAIT_PERIOD) / _GAIT_PERIOD
        phase_right = (phase_left + _GAIT_OFFSET) % 1.0
        return phase_left, phase_right

    # ─────────────────────────────────────────────────────────────────────────
    # Reward computation  (matches G1Robot + G1RoughCfg exactly)
    # ─────────────────────────────────────────────────────────────────────────

    def _compute_rewards(self, action):
        """
        Reward terms mirroring G1Robot._reward_* with G1RoughCfg scales.
        All terms that are not 'alive' or 'contact' are multiplied by dt.
        """
        dt = _DECIMATION * self.model.opt.timestep   # policy timestep

        # ── State ─────────────────────────────────────────────────────────
        qpos = self.sim.data.qpos
        qvel = self.sim.data.qvel

        height = float(qpos[2])
        quat   = qpos[3:7]          # [w, x, y, z]
        roll, pitch, yaw = quat_to_euler(quat)

        # Body-frame velocities (quat_rotate_inverse)
        lin_vel_world = qvel[0:3].copy()
        ang_vel_world = qvel[3:6].copy()
        base_lin_vel  = quat_rotate_inverse(quat, lin_vel_world)   # body frame
        base_ang_vel  = quat_rotate_inverse(quat, ang_vel_world)

        # Projected gravity (body frame) — matches IsaacGym projected_gravity
        gravity_world = np.array([0.0, 0.0, -1.0])
        proj_gravity  = quat_rotate_inverse(quat, gravity_world)

        # DOF state (joint space, no free-body DOFs)
        dof_pos = qpos[7:].copy()    # (12,)
        dof_vel = qvel[6:].copy()    # (12,)

        # ── Foot contact state ────────────────────────────────────────────
        left_bname  = self.model.body_names[self.left_foot_body_id]
        right_bname = self.model.body_names[self.right_foot_body_id]

        left_contact  = self._body_in_contact(self.left_foot_body_id,  threshold=1.0)
        right_contact = self._body_in_contact(self.right_foot_body_id, threshold=1.0)

        left_foot_vel  = self._get_foot_velocity(left_bname)
        right_foot_vel = self._get_foot_velocity(right_bname)
        left_foot_z    = self._get_foot_pos_z(left_bname)
        right_foot_z   = self._get_foot_pos_z(right_bname)

        phase_left, phase_right = self._gait_phase()

        # ── Reward terms ──────────────────────────────────────────────────

        # 1. tracking_lin_vel  (exp kernel, xy only)  — matches _reward_tracking_lin_vel
        lin_vel_error = np.sum(np.square(self._commands[:2] - base_lin_vel[:2]))
        r_tracking_lin_vel = np.exp(-lin_vel_error / _TRACKING_SIGMA)

        # 2. tracking_ang_vel  (yaw)                  — matches _reward_tracking_ang_vel
        ang_vel_error = float((self._commands[2] - base_ang_vel[2]) ** 2)
        r_tracking_ang_vel = np.exp(-ang_vel_error / _TRACKING_SIGMA)

        # 3. lin_vel_z                                — matches _reward_lin_vel_z
        r_lin_vel_z = float(base_lin_vel[2] ** 2)

        # 4. ang_vel_xy                               — matches _reward_ang_vel_xy
        r_ang_vel_xy = float(np.sum(base_ang_vel[:2] ** 2))

        # 5. orientation  (projected gravity xy)      — matches _reward_orientation
        r_orientation = float(np.sum(proj_gravity[:2] ** 2))

        # 6. base_height                              — matches _reward_base_height
        r_base_height = float((height - _BASE_HEIGHT_TARGET) ** 2)

        # 7. dof_acc                                  — matches _reward_dof_acc
        # Guard against size mismatch on the very first step during __init__
        last_dof_vel = self._last_qvel[6:] if len(self._last_qvel) > 6 else np.zeros_like(dof_vel)
        if len(last_dof_vel) != len(dof_vel):
            last_dof_vel = np.zeros_like(dof_vel)
        dof_acc = np.sum(np.square((dof_vel - last_dof_vel) / dt))
        r_dof_acc = float(dof_acc)

        # 8. dof_vel                                  — matches _reward_dof_vel
        r_dof_vel = float(np.sum(dof_vel ** 2))

        # 9. action_rate                              — matches _reward_action_rate
        last_action = self._last_action if len(self._last_action) == len(action) else np.zeros_like(action)
        r_action_rate = float(np.sum(np.square(action - last_action)))

        # 10. dof_pos_limits  (soft limit penalty)    — matches _reward_dof_pos_limits
        lo = self._dof_pos_limits_lower if len(self._dof_pos_limits_lower) == len(dof_pos) else np.full_like(dof_pos, -np.pi)
        hi = self._dof_pos_limits_upper if len(self._dof_pos_limits_upper) == len(dof_pos) else np.full_like(dof_pos,  np.pi)
        out_lo = np.maximum(0.0, lo - dof_pos)
        out_hi = np.maximum(0.0, dof_pos - hi)
        r_dof_pos_limits = float(np.sum(out_lo + out_hi))

        # 11. alive                                   — matches _reward_alive
        r_alive = 1.0

        # 12. hip_pos  (penalise hip roll / yaw)      — matches _reward_hip_pos
        # Joint indices for left_hip_roll(1), left_hip_yaw(0), right_hip_roll(7), right_hip_yaw(6)
        hip_indices = [0, 1, 6, 7]   # yaw, roll × 2 sides
        r_hip_pos = float(np.sum(dof_pos[hip_indices] ** 2))

        # 13. contact  (gait-phase-aware)             — matches _reward_contact
        left_is_stance  = float(phase_left  < 0.55)
        right_is_stance = float(phase_right < 0.55)
        left_correct    = float(left_contact  == (left_is_stance  > 0.5))
        right_correct   = float(right_contact == (right_is_stance > 0.5))
        r_contact = left_correct + right_correct   # 0..2

        # 14. contact_no_vel  (penalise sliding feet) — matches _reward_contact_no_vel
        left_pen  = np.sum(left_foot_vel  ** 2) * float(left_contact)
        right_pen = np.sum(right_foot_vel ** 2) * float(right_contact)
        r_contact_no_vel = float(left_pen + right_pen)

        # 15. feet_swing_height                       — matches _reward_feet_swing_height
        # squared error to 0.08 m target, active only during swing
        l_sw = float((left_foot_z  - 0.08) ** 2) * (1.0 - float(left_contact))
        r_sw = float((right_foot_z - 0.08) ** 2) * (1.0 - float(right_contact))
        r_feet_swing_height = l_sw + r_sw

        # ── Apply scales × dt  (mirrors _prepare_reward_function dt multiply) ──
        s = _REWARD_SCALES
        total = (
            s['tracking_lin_vel']  * r_tracking_lin_vel  * dt +
            s['tracking_ang_vel']  * r_tracking_ang_vel  * dt +
            s['lin_vel_z']         * r_lin_vel_z          * dt +
            s['ang_vel_xy']        * r_ang_vel_xy         * dt +
            s['orientation']       * r_orientation        * dt +
            s['base_height']       * r_base_height        * dt +
            s['dof_acc']           * r_dof_acc            * dt +
            s['dof_vel']           * r_dof_vel            * dt +
            s['action_rate']       * r_action_rate        * dt +
            s['dof_pos_limits']    * r_dof_pos_limits     * dt +
            s['alive']             * r_alive              * dt +
            s['hip_pos']           * r_hip_pos            * dt +
            s['contact']           * r_contact            * dt +
            s['contact_no_vel']    * r_contact_no_vel     * dt +
            s['feet_swing_height'] * r_feet_swing_height  * dt
        )

        info = {
            'r_tracking_lin_vel':  s['tracking_lin_vel']  * r_tracking_lin_vel  * dt,
            'r_tracking_ang_vel':  s['tracking_ang_vel']  * r_tracking_ang_vel  * dt,
            'r_lin_vel_z':         s['lin_vel_z']         * r_lin_vel_z          * dt,
            'r_ang_vel_xy':        s['ang_vel_xy']        * r_ang_vel_xy         * dt,
            'r_orientation':       s['orientation']       * r_orientation        * dt,
            'r_base_height':       s['base_height']       * r_base_height        * dt,
            'r_dof_acc':           s['dof_acc']           * r_dof_acc            * dt,
            'r_dof_vel':           s['dof_vel']           * r_dof_vel            * dt,
            'r_action_rate':       s['action_rate']       * r_action_rate        * dt,
            'r_dof_pos_limits':    s['dof_pos_limits']    * r_dof_pos_limits     * dt,
            'r_alive':             s['alive']             * r_alive              * dt,
            'r_hip_pos':           s['hip_pos']           * r_hip_pos            * dt,
            'r_contact':           s['contact']           * r_contact            * dt,
            'r_contact_no_vel':    s['contact_no_vel']    * r_contact_no_vel     * dt,
            'r_feet_swing_height': s['feet_swing_height'] * r_feet_swing_height  * dt,
            'total':               total,
            # diagnostics
            'height':       height,
            'pitch':        pitch,
            'roll':         roll,
            'base_lin_vel': base_lin_vel.tolist(),
            'base_ang_vel': base_ang_vel.tolist(),
            'phase_left':   phase_left,
            'phase_right':  phase_right,
        }
        return total, info

    # ─────────────────────────────────────────────────────────────────────────
    # Termination
    # ─────────────────────────────────────────────────────────────────────────

    def _check_termination(self):
        """
        Mirrors IsaacGym check_termination:
          - contact force on pelvis > 1 N  (terminate_after_contacts_on=['pelvis'])
          - |pitch| > 1.0 rad
          - |roll|  > 0.8 rad
          - height out of [0.4, 1.15]  (extra safety)
        cfrc_ext[bid, 5] is the precomputed vertical contact force — no iteration needed.
        """
        qpos = self.sim.data.qpos
        height = float(qpos[2])
        quat   = qpos[3:7]
        roll, pitch, _ = quat_to_euler(quat)

        pelvis_contact = self._contact_fz(self.pelvis_body_id) > 1.0
        height_ok  = 0.4 < height < 1.15
        pitch_ok   = abs(pitch) <= 1.0
        roll_ok    = abs(roll)  <= 0.8

        return pelvis_contact or not (height_ok and pitch_ok and roll_ok)

    # ─────────────────────────────────────────────────────────────────────────
    # Observations — per-limb node-centric format for transformer
    # ─────────────────────────────────────────────────────────────────────────
    # The transformer expects one token per limb of fixed size limb_obs_size.
    # ModularObservationPadding computes limb_obs_size = len(proprioceptive) // num_limbs
    # so proprioceptive must be exactly num_limbs × limb_obs_size with no remainder.
    #
    # Per-limb token layout:
    #   graph_encoding == "none":  xpos(3)+xvelp(3)+xvelr(3)+expmap(3)+ltv(6)+angle(1)+jrange(2)+global(11) = 32
    #   graph_encoding != "none":  xpos(3)+xvelp(3)+xvelr(3)+expmap(3)+angle(1)+jrange(2)+global(11)        = 26
    #
    # global(11) on root body only, zeros on all other limbs:
    #   ang_vel*0.25 (3) + proj_gravity (3) + cmd_scaled (3) + sin_phase (1) + cos_phase (1)

    def _get_obs_per_limb(self, b):
        """Per-limb observation token. Size is uniform across all limbs."""
        qpos = self.sim.data.qpos
        qvel = self.sim.data.qvel
        quat = qpos[3:7]

        # Body kinematics
        root_x_pos = self.data.get_body_xpos(self._root_body)[0]
        xpos  = self.data.get_body_xpos(b).copy()
        xpos[0] -= root_x_pos
        xvelp = np.clip(self.data.get_body_xvelp(b), -10, 10)
        xvelr = self.data.get_body_xvelr(b)
        expmap = quat2expmap(self.data.get_body_xquat(b))

        # Joint angle (local to this limb)
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

        # Global state on root body only — zeros on all other limbs.
        # Keeps token size uniform so the wrapper computes limb_obs_size correctly.
        if b == self._root_body:
            ang_vel  = quat_rotate_inverse(quat, qvel[3:6].copy()) * 0.25  # (3,)
            proj_grav = quat_rotate_inverse(quat, np.array([0.0, 0.0, -1.0]))  # (3,)
            cmd      = self._commands * np.array([2.0, 2.0, 0.25])             # (3,)
            phase_left, _ = self._gait_phase()
            phase    = np.array([
                np.sin(2.0 * np.pi * phase_left),
                np.cos(2.0 * np.pi * phase_left),
            ])                                                                  # (2,)
        else:
            ang_vel   = np.zeros(3)
            proj_grav = np.zeros(3)
            cmd       = np.zeros(3)
            phase     = np.zeros(2)

        global_state = np.concatenate([ang_vel, proj_grav, cmd, phase])  # (11,)

        if self.graph_encoding == "none":
            n = b.lower()
            if   "hip_pitch" in n: ltv = np.array((1,0,0,0,0,0))
            elif "hip_roll"  in n: ltv = np.array((0,1,0,0,0,0))
            elif "hip_yaw"   in n: ltv = np.array((0,0,1,0,0,0))
            elif "knee"      in n: ltv = np.array((0,0,0,1,0,0))
            elif "ankle"     in n: ltv = np.array((0,0,0,0,1,0))
            else:                  ltv = np.array((0,0,0,0,0,0))
            return np.concatenate([xpos, xvelp, xvelr, expmap, ltv,
                                   [angle], joint_range, global_state])  # 32

        return np.concatenate([xpos, xvelp, xvelr, expmap,
                               [angle], joint_range, global_state])      # 26

    def _get_obs(self):
        """
        Proprioceptive obs: num_limbs × limb_obs_size (26 or 32 per limb).
        Concatenated flat so ModularObservationPadding can compute
        limb_obs_size = len(proprioceptive) // num_limbs without remainder.
        Global state (ang_vel, proj_gravity, commands, gait phase) is packed
        into the root body token; all other tokens carry zeros for those dims.
        """
        full_obs = np.concatenate(
            [self._get_obs_per_limb(b) for b in self.model.body_names[1:]]
        ).ravel()

        if not self._graph_built:
            return full_obs

        result = {
            'proprioceptive': full_obs,
            'context':        self.context_features,
            'edges':          self.edges,
            'traversals':     self.traversals,
            'SWAT_RE':        self.SWAT_RE,
        }
        if self.graph_encoding != "none":
            result['graph_node_features'] = self._graph_node_features
            result['graph_A_norm']        = self._graph_A_norm
        return result

    # ─────────────────────────────────────────────────────────────────────────
    # Step / Reset
    # ─────────────────────────────────────────────────────────────────────────

    def step(self, action):
        """
        Policy step:
          1. Compute PD torques from position-offset actions
          2. Step simulation _DECIMATION times at raw sim dt
          3. Compute rewards, check termination, return
        """
        self._last_qpos = self.sim.data.qpos.copy()
        self._last_qvel = self.sim.data.qvel.copy()

        torques = self._compute_torques(action)

        # Decimation loop — 4 sim steps per policy step
        for _ in range(_DECIMATION):
            self.sim.data.ctrl[:] = torques
            self.sim.step()

        reward, reward_info = self._compute_rewards(action)
        obs  = self._get_obs()
        done = self._check_termination()

        self._episode_step   += 1
        self._episode_return += reward
        self._last_action     = action.copy()

        info = {'name': self._robot_name}
        info.update(reward_info)

        max_steps = 1000
        if done or self._episode_step >= max_steps:
            info['episode'] = {
                'r': self._episode_return,
                'l': self._episode_step,
            }
            if done:
                self._episode_step   = 0
                self._episode_return = 0.0

        return obs, reward, done, info

    def reset_model(self):
        """
        Reset to standing pose with small perturbations.
        Mirrors IsaacGym _reset_dofs (0.5–1.5x default) + _reset_root_states.
        """
        self._episode_step   = 0
        self._episode_return = 0.0

        # Resample commands uniformly — mirrors _resample_commands
        self._commands = np.array([
            self.np_random.uniform(0.0, 1.0),    # lin_vel_x  (forward)
            self.np_random.uniform(-0.5, 0.5),   # lin_vel_y
            self.np_random.uniform(-0.5, 0.5),   # yaw_vel
        ])
        # Zero out tiny commands (matches |cmd| > 0.2 threshold)
        if np.linalg.norm(self._commands[:2]) < 0.2:
            self._commands[:2] = 0.0

        qpos = self.init_qpos.copy()
        qvel = np.zeros(self.model.nv)

        # Default joint angles × uniform(0.5, 1.5) — mirrors IsaacGym _reset_dofs
        for i, jname in enumerate(self.model.joint_names[1:]):
            if i >= self.model.nu:
                break
            default = _DEFAULT_JOINT_ANGLES.get(jname, 0.0)
            scale   = self.np_random.uniform(0.5, 1.5)
            qpos[7 + i] = default * scale

        # Root position with small noise
        qpos[0] = 0.0
        qpos[1] = 0.0
        qpos[2] = self._init_height + self.np_random.uniform(-0.02, 0.02)

        # Identity quaternion + small tilt perturbation
        qpos[3:7] = np.array([1.0, 0.0, 0.0, 0.0])
        qpos[3:7] += self.np_random.uniform(-0.05, 0.05, size=4)
        qpos[3:7] /= np.linalg.norm(qpos[3:7])

        # Small joint velocity noise (mirrors IsaacGym base vel randomisation)
        qvel[6:] = self.np_random.uniform(-0.5, 0.5, size=self.model.nv - 6)
        qvel[:6] = self.np_random.uniform(-0.5, 0.5, size=6)

        self.set_state(qpos, qvel)

        self._last_qpos   = self.sim.data.qpos.copy()
        self._last_qvel   = self.sim.data.qvel.copy()
        self._last_action = np.zeros(self.model.nu)

        return self._get_obs()

    # ─────────────────────────────────────────────────────────────────────────
    # Graph / context (unchanged from original — not the bottleneck)
    # ─────────────────────────────────────────────────────────────────────────

    def _init_graph_data(self):
        parser = MujocoGraphParser(self.xml)
        self._graph_node_features = parser.get_features(self.graph_encoding)
        self._graph_A_norm        = parser.normalized_adjacency()
        print(f"[{self.graph_encoding}] Graph: {parser.N} nodes, "
              f"features {self._graph_node_features.shape}, "
              f"A_norm {self._graph_A_norm.shape}")

    def _build_graph_structure(self):
        body_idxs  = list(range(1, self.metadata['num_limbs'] + 1))
        joint_to   = self.sim.model.jnt_bodyid[1:].copy() - 1
        body_parentids = self.sim.model.body_parentid.copy()
        joint_from = np.array([body_parentids[child + 1] - 1 for child in joint_to])
        assert len(joint_to) == self.metadata['num_joints']
        self.edges = np.vstack((joint_to, joint_from)).T.flatten().astype(np.int32)

        parents = [-1] * self.metadata['num_limbs']
        for i in range(len(joint_to)):
            child_idx  = joint_to[i]
            parent_idx = joint_from[i]
            if 0 <= child_idx < len(parents):
                parents[child_idx] = parent_idx

        self.traversals = self._get_traversal(parents)
        if cfg.MODEL.TRANSFORMER.USE_SWAT_RE:
            self.SWAT_RE = self._get_graph_dict(parents)

    def _get_traversal(self, parents):
        root     = parents.index(-1) if -1 in parents else 0
        children = {i: [] for i in range(len(parents))}
        for i, p in enumerate(parents):
            if p >= 0:
                children[p].append(i)
        trav = []
        def dfs(n):
            trav.append(n)
            for c in sorted(children[n]):
                dfs(c)
        dfs(root)
        return np.array(trav, dtype=np.int32)

    def _get_graph_dict(self, parents):
        n = len(parents)
        gd = np.zeros([cfg.MODEL.MAX_LIMBS, cfg.MODEL.MAX_LIMBS, 3])
        children = {i: [] for i in range(n)}
        for i, p in enumerate(parents):
            if p >= 0:
                children[p].append(i)
        for i in range(n):
            for j in range(n):
                if i == j:
                    gd[i, j] = [1, 0, 0]
                elif parents[i] == j:
                    gd[i, j] = [0, 1, 0]
                elif j in children[i]:
                    gd[i, j] = [0, 0, 1]
        return gd

    def _compute_normalization_bounds(self):
        num_limbs  = self.metadata['num_limbs']
        body_idxs  = list(range(1, num_limbs + 1))
        lb, jb = {}, {}

        body_pos = self.sim.model.body_pos[body_idxs, :]
        lb['body_pos']  = (body_pos.min(0, keepdims=True), body_pos.max(0, keepdims=True))
        body_ipos = self.sim.model.body_ipos[body_idxs, :]
        lb['body_ipos'] = (body_ipos.min(0, keepdims=True), body_ipos.max(0, keepdims=True))
        lb['body_iquat']  = (np.array([[-1,-1,-1,-1]]), np.array([[1,1,1,1]]))
        lb['geom_quat']   = (np.array([[-1,-1,-1,-1]]), np.array([[1,1,1,1]]))
        bm = self.sim.model.body_mass[body_idxs]
        lb['body_mass'] = (np.array([[bm.min()]]), np.array([[bm.max()]]))

        geom_idxs = []
        for bi in body_idxs:
            for gi in range(self.sim.model.ngeom):
                if self.sim.model.geom_bodyid[gi] == bi:
                    geom_idxs.append(gi); break
        if len(geom_idxs) == num_limbs:
            gs = self.sim.model.geom_size[geom_idxs, :2]
            lb['body_shape']    = (gs.min(0, keepdims=True), gs.max(0, keepdims=True))
            gf = self.sim.model.geom_friction[geom_idxs, 0:1]
            lb['body_friction'] = (gf.min(0, keepdims=True), gf.max(0, keepdims=True))
        else:
            lb['body_shape']    = (np.array([[0.01,0.01]]), np.array([[0.2,0.2]]))
            lb['body_friction'] = (np.array([[0.5]]),        np.array([[1.5]]))

        jp = self.sim.model.jnt_pos[1:, :]
        jb['jnt_pos']    = (jp.min(0, keepdims=True), jp.max(0, keepdims=True))
        jr = self.sim.model.jnt_range[1:, :]
        jb['joint_range']= (jr.min(0, keepdims=True), jr.max(0, keepdims=True))
        ja = self.sim.model.jnt_axis[1:, :]
        jb['joint_axis'] = (ja.min(0, keepdims=True), ja.max(0, keepdims=True))
        gear = self.sim.model.actuator_gear[:, 0:1]
        jb['gear']       = (gear.min(0, keepdims=True), gear.max(0, keepdims=True))
        arm  = self.sim.model.dof_armature[6:][:, np.newaxis]
        jb['armature']   = (arm.min(0, keepdims=True), arm.max(0, keepdims=True))
        damp = self.sim.model.dof_damping[6:][:, np.newaxis]
        jb['damping']    = (damp.min(0, keepdims=True), damp.max(0, keepdims=True))
        return lb, jb

    def _compute_context_encoding(self):
        num_limbs  = self.metadata['num_limbs']
        body_idxs  = list(range(1, num_limbs + 1))
        lb, jb = self._compute_normalization_bounds()

        def norm(arr, key, bounds):
            lo, hi = bounds[key]
            return np.clip(-1.0 + 2.0*(arr - lo)/(hi - lo + 1e-8), -1.0, 1.0)

        cl = {}
        cl['body_pos']  = norm(self.sim.model.body_pos[body_idxs].copy(),  'body_pos',  lb)
        cl['body_ipos'] = norm(self.sim.model.body_ipos[body_idxs].copy(), 'body_ipos', lb)
        cl['body_iquat']= norm(self.sim.model.body_iquat[body_idxs].copy(),'body_iquat',lb)
        cl['body_mass'] = norm(self.sim.model.body_mass[body_idxs].copy()[:,np.newaxis],'body_mass',lb)

        geom_idxs = []
        for bi in body_idxs:
            for gi in range(self.sim.model.ngeom):
                if self.sim.model.geom_bodyid[gi] == bi:
                    geom_idxs.append(gi); break
        if len(geom_idxs) == num_limbs:
            cl['geom_quat']   = norm(self.sim.model.geom_quat[geom_idxs].copy(),    'geom_quat',   lb)
            cl['body_shape']  = norm(self.sim.model.geom_size[geom_idxs,:2].copy(), 'body_shape',  lb)
            cl['body_friction']= norm(self.sim.model.geom_friction[geom_idxs,0:1].copy(),'body_friction',lb)
        else:
            cl['geom_quat']    = np.tile([1,0,0,0], (num_limbs,1))
            cl['body_shape']   = np.zeros((num_limbs,2))
            cl['body_friction']= np.ones((num_limbs,1))

        cj = {}
        cj['jnt_pos']    = norm(self.sim.model.jnt_pos[1:].copy(),              'jnt_pos',    jb)
        cj['joint_range']= norm(self.sim.model.jnt_range[1:].copy(),            'joint_range',jb)
        cj['joint_axis'] = norm(self.sim.model.jnt_axis[1:].copy(),             'joint_axis', jb)
        cj['gear']       = norm(self.sim.model.actuator_gear[:,0:1].copy(),      'gear',       jb)
        cj['armature']   = norm(self.sim.model.dof_armature[6:][:,np.newaxis].copy(),'armature',jb)
        cj['damping']    = norm(self.sim.model.dof_damping[6:][:,np.newaxis].copy(), 'damping', jb)

        lf = self._select_context_obs(cl, cfg.MODEL.CONTEXT_OBS_TYPES)
        jf = self._select_context_obs(cj, cfg.MODEL.CONTEXT_OBS_TYPES)
        self.context_features = self._combine_limb_joint_context(lf, jf)

    def _select_context_obs(self, d, keys):
        out = [d[k] for k in keys if k in d]
        return np.hstack(out) if out else np.array([])

    def _combine_limb_joint_context(self, limb_obs, joint_obs):
        num_limbs  = self.metadata['num_limbs']
        num_joints = self.metadata['num_joints']
        if limb_obs is None or len(limb_obs) == 0:
            return joint_obs.flatten() if (joint_obs is not None and len(joint_obs)) else np.array([])
        if joint_obs is None or len(joint_obs) == 0:
            return limb_obs.flatten()
        jsz  = joint_obs.shape[1]
        jpad = np.zeros((num_limbs, jsz * 2))
        if len(self.edges) > 0:
            j2l  = self.edges[::2]
            jcnt = np.zeros(num_limbs, dtype=int)
            for i, li in enumerate(j2l):
                if 0 <= li < num_limbs and i < num_joints:
                    s = jcnt[li] * jsz
                    e = s + jsz
                    if e <= jsz * 2:
                        jpad[li, s:e] = joint_obs[i]
                        jcnt[li] += 1
        return np.hstack((limb_obs, jpad)).flatten()

    def viewer_setup(self):
        self.viewer.cam.trackbodyid = 1
        self.viewer.cam.distance    = self.model.stat.extent * 2.0
        self.viewer.cam.lookat[2]   = self._init_height
        self.viewer.cam.elevation   = -20


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

def make_env(xml):
    env = ModularEnv(xml)
    print(f"Made env for {xml} | graph_encoding={cfg.MODEL.GRAPH_ENCODING} | "
          f"obs_dim=47 | action_dim={env.model.nu} | PD control")
    if cfg.MODEL.MLP.CONSISTENT_PADDING:
        env = ConsistentModularObservationPadding(env)
        env = ConsistentModularActionPadding(env)
    else:
        env = ModularObservationPadding(env)
        env = ModularActionPadding(env)
    return env