"""
isaacgym_vec_env.py  —  Drop-in replacement for SubprocVecEnv + VecNormalize + VecPyTorch.

Wraps an Isaac Gym G1Robot env so it looks exactly like what ppo.py expects:
  - obs dict with keys: proprioceptive, context, edges, traversals, SWAT_RE,
                        obs_padding_mask, act_padding_mask,
                        [graph_node_features, graph_A_norm if GCN active]
  - step(actions) → obs_dict, rewards, dones, infos
  - reset()       → obs_dict
  - observation_space, action_space  (gym.spaces, used by ActorCritic __init__)
  - ob_rms  (RunningMeanStd, used by save_model / restore_from_checkpoint)

Physics:  Isaac Gym GPU  (4096 envs, ~100k FPS)
Policy:   your PyTorch transformer + GCN  (unchanged)
PPO loop: your metamorph PPO             (unchanged)

Usage in envs.py — add this branch in make_vec_envs():
    elif cfg.VECENV.TYPE == "IsaacGym":
        from isaacgym_vec_env import make_isaacgym_vec_env
        return make_isaacgym_vec_env(training=training, norm_rew=norm_rew)
"""
import isaacgym
from isaacgym import gymapi, gymtorch

import os
import math
import numpy as np
import torch
from gym import spaces

from metamorph.config import cfg
from metamorph.envs.vec_env.running_mean_std import RunningMeanStd
from graphs.parser import MujocoGraphParser


# ---------------------------------------------------------------------------
# Batched quaternion helpers  (pure PyTorch, no numpy, works on GPU)
# ---------------------------------------------------------------------------

def quat_rotate_inverse_batch(q, v):
    """
    Rotate vectors v by inverse of quaternions q.
    q : (N, 4)  [w, x, y, z]
    v : (N, 3)
    returns (N, 3) in body frame
    """
    w = q[:, 0:1]
    xyz = q[:, 1:]                              # (N, 3)
    t = 2.0 * torch.cross(xyz, v, dim=1)        # (N, 3)
    return v - w * t + torch.cross(xyz, t, dim=1)


def quat_to_euler_batch(q):
    """
    q : (N, 4) [w, x, y, z]
    returns roll, pitch, yaw each (N,)
    """
    w, x, y, z = q[:,0], q[:,1], q[:,2], q[:,3]
    roll  = torch.atan2(2*(w*x + y*z), 1 - 2*(x*x + y*y))
    pitch = torch.asin((2*(w*y - z*x)).clamp(-1, 1))
    yaw   = torch.atan2(2*(w*z + x*y), 1 - 2*(y*y + z*z))
    return roll, pitch, yaw


# ---------------------------------------------------------------------------
# Static graph data  (computed once from MuJoCo XML, reused every step)
# ---------------------------------------------------------------------------

class RobotGraphData:
    """
    Precomputes all static per-robot graph data from a MuJoCo XML.
    This data never changes during training — it describes the robot's
    morphology, not its dynamic state.
    """

    def __init__(self, xml_path, device):
        from modular.g1_12dof import ModularEnv as MujocoEnv

        # Spin up a MuJoCo env briefly just to extract static structure
        tmp = MujocoEnv(xml_path)

        self.num_limbs  = tmp.metadata['num_limbs']   # 13 for G1
        self.num_joints = tmp.metadata['num_joints']  # 12 for G1
        max_limbs       = cfg.MODEL.MAX_LIMBS
        num_pads        = max_limbs - self.num_limbs

        # Graph structure tensors  (1, ...) — will be expanded to (N, ...) per batch
        self.context    = torch.tensor(tmp.context_features, dtype=torch.float32, device=device).unsqueeze(0)
        self.edges      = torch.tensor(tmp.edges,            dtype=torch.float32, device=device).unsqueeze(0)
        self.traversals = torch.tensor(tmp.traversals,       dtype=torch.float32, device=device).unsqueeze(0)
        self.swat_re    = torch.tensor(tmp.SWAT_RE,          dtype=torch.float32, device=device).unsqueeze(0)

        # Padding masks
        obs_mask = [False] * self.num_limbs + [True] * num_pads
        act_mask = [True]  + [False] * (self.num_limbs - 1) + [True] * num_pads
        self.obs_padding_mask = torch.tensor(obs_mask, dtype=torch.bool,   device=device).unsqueeze(0)
        self.act_padding_mask = torch.tensor(act_mask, dtype=torch.bool,   device=device).unsqueeze(0)

        # Default joint angles for dof_pos_offset computation
        self.default_dof_pos = torch.tensor(
            tmp._default_dof_pos, dtype=torch.float32, device=device)  # (12,)

        # Joint names in MuJoCo order — for limb-type one-hot
        self.joint_names = list(tmp.model.joint_names[1:])  # skip free joint
        self.body_names  = list(tmp.model.body_names[1:])   # skip world body

        # Body kinematic offsets from model (for per-limb xpos approximation)
        # Isaac Gym gives us rigid_body_states directly so we don't need these,
        # but we cache them as fallback
        self.body_pos_default = torch.tensor(
            tmp.sim.model.body_pos[1:].copy(), dtype=torch.float32, device=device)  # (13, 3)

        # GCN graph data
        if cfg.MODEL.GRAPH_ENCODING != "none":
            parser = MujocoGraphParser(xml_path)
            self.graph_node_features = torch.tensor(
                parser.get_features(cfg.MODEL.GRAPH_ENCODING),
                dtype=torch.float32, device=device).unsqueeze(0)   # (1, N, feat_dim)
            self.graph_A_norm = torch.tensor(
                parser.normalized_adjacency(),
                dtype=torch.float32, device=device).unsqueeze(0)   # (1, N, N)

            # Pad to MAX_LIMBS
            N, feat_dim = self.graph_node_features.shape[1:]
            if N < cfg.MODEL.MAX_LIMBS:
                pad_feats = torch.zeros(1, cfg.MODEL.MAX_LIMBS - N, feat_dim, device=device)
                self.graph_node_features = torch.cat([self.graph_node_features, pad_feats], dim=1)
                pad_A_row = torch.zeros(1, cfg.MODEL.MAX_LIMBS - N, N, device=device)
                A = self.graph_A_norm
                A = torch.cat([A, pad_A_row], dim=1)
                pad_A_col = torch.zeros(1, cfg.MODEL.MAX_LIMBS, cfg.MODEL.MAX_LIMBS - N, device=device)
                A = torch.cat([A, pad_A_col], dim=2)
                self.graph_A_norm = A
        else:
            self.graph_node_features = None
            self.graph_A_norm        = None

        tmp.close()
        del tmp
        print(f"[RobotGraphData] Loaded graph for {os.path.basename(xml_path)}: "
              f"{self.num_limbs} limbs, {self.num_joints} joints")


# ---------------------------------------------------------------------------
# Per-limb observation builder  (batched PyTorch version of _get_obs_per_limb)
# ---------------------------------------------------------------------------

class ObsBuilder:
    """
    Builds per-limb observation tokens from Isaac Gym raw state tensors.
    Mirrors _get_obs_per_limb in g1_12dof.py but batched over all envs.

    Per-limb token layout (matches MuJoCo env exactly):
      graph_encoding == "none":  xpos(3)+xvelp(3)+xvelr(3)+expmap(3)+ltv(6)+angle(1)+jrange(2)+global(11) = 32
      graph_encoding != "none":  xpos(3)+xvelp(3)+xvelr(3)+expmap(3)+angle(1)+jrange(2)+global(11)        = 26

    global(11) on root body token, zeros on all others:
      ang_vel*0.25(3) + proj_gravity(3) + cmd_scaled(3) + sin_phase(1) + cos_phase(1)
    """

    GAIT_PERIOD  = 0.8
    GAIT_OFFSET  = 0.5

    def __init__(self, graph_data: RobotGraphData, device, joint_ranges: torch.Tensor):
        self.gd          = graph_data
        self.device      = device
        self.joint_ranges = joint_ranges   # (12, 2) lower/upper in radians

        use_ltv = (cfg.MODEL.GRAPH_ENCODING == "none")
        self.limb_obs_size = 32 if use_ltv else 26
        self.use_ltv       = use_ltv

        # Precompute per-joint limb-type one-hot  (12, 6) — for "none" encoding
        if use_ltv:
            ltvs = []
            for jname in graph_data.joint_names:
                n = jname.lower()
                if   "hip_pitch" in n: ltv = [1,0,0,0,0,0]
                elif "hip_roll"  in n: ltv = [0,1,0,0,0,0]
                elif "hip_yaw"   in n: ltv = [0,0,1,0,0,0]
                elif "knee"      in n: ltv = [0,0,0,1,0,0]
                elif "ankle"     in n: ltv = [0,0,0,0,1,0]
                else:                  ltv = [0,0,0,0,0,0]
                ltvs.append(ltv)
            self.ltv = torch.tensor(ltvs, dtype=torch.float32, device=device)  # (12, 6)

        # joint_range normalised to [-1, 1] for obs  (12, 2)
        self.jrange_norm = joint_ranges / math.pi   # (12, 2)

    def build(self,
              root_states,        # (N, 13): pos(3)+quat(4)+linvel(3)+angvel(3)
              dof_pos,            # (N, 12)
              dof_vel,            # (N, 12)
              rigid_body_states,  # (N, num_bodies, 13)
              last_actions,       # (N, 12)
              commands,           # (N, 3)  vx, vy, yaw
              episode_steps,      # (N,)    int, for gait phase
              dt,                 # scalar
              ):
        N = root_states.shape[0]
        num_limbs  = self.gd.num_limbs     # 13
        max_limbs  = cfg.MODEL.MAX_LIMBS   # 13

        # ── Global state (computed once, placed on root token) ───────────────
        quat        = root_states[:, 3:7]   # (N, 4) [w,x,y,z]

        # Body-frame angular velocity
        ang_vel_world = root_states[:, 10:13]                    # (N, 3)
        base_ang_vel  = quat_rotate_inverse_batch(quat, ang_vel_world) * 0.25  # (N, 3)

        # Projected gravity
        grav_world  = torch.tensor([0., 0., -1.], device=self.device).expand(N, 3)
        proj_gravity = quat_rotate_inverse_batch(quat, grav_world)              # (N, 3)

        # Scaled commands
        cmd_scaled = commands * torch.tensor([2.0, 2.0, 0.25], device=self.device)  # (N, 3)

        # Gait phase
        t           = episode_steps.float() * dt                                # (N,)
        phase_left  = (t % self.GAIT_PERIOD) / self.GAIT_PERIOD                # (N,)
        phase_right = (phase_left + self.GAIT_OFFSET) % 1.0
        sin_phase   = torch.sin(2.0 * math.pi * phase_left).unsqueeze(1)       # (N, 1)
        cos_phase   = torch.cos(2.0 * math.pi * phase_left).unsqueeze(1)       # (N, 1)

        global_state = torch.cat([base_ang_vel, proj_gravity,
                                   cmd_scaled, sin_phase, cos_phase], dim=1)   # (N, 11)

        # ── Per-body kinematics from rigid_body_states ───────────────────────
        # rigid_body_states: (N, num_bodies, 13)
        # Isaac Gym body ordering matches URDF: body 0 = pelvis (root), 1..12 = joints
        # xpos:  world-frame position  [:, :, 0:3]
        # xvelp: linear velocity       [:, :, 7:10]
        # xvelr: angular velocity      [:, :, 10:13]
        # xquat: orientation           [:, :, 3:7]  [x,y,z,w] in Isaac Gym → reorder to [w,x,y,z]

        body_pos  = rigid_body_states[:, :num_limbs, 0:3]    # (N, 13, 3)
        body_velp = rigid_body_states[:, :num_limbs, 7:10]   # (N, 13, 3)
        body_velr = rigid_body_states[:, :num_limbs, 10:13]  # (N, 13, 3)
        # Isaac Gym quat is [x,y,z,w], convert to [w,x,y,z]
        ig_quat   = rigid_body_states[:, :num_limbs, 3:7]    # (N, 13, 4)
        body_quat = torch.cat([ig_quat[:,:,3:4], ig_quat[:,:,0:3]], dim=2)  # (N, 13, 4) [w,x,y,z]

        # Root-relative x position (translation invariant)
        root_x    = body_pos[:, 0:1, 0:1]   # (N, 1, 1)
        xpos      = body_pos.clone()
        xpos[:, :, 0:1] -= root_x           # (N, 13, 3)
        xvelp     = body_velp.clamp(-10, 10) # (N, 13, 3)
        xvelr     = body_velr                # (N, 13, 3)

        # Exponential map from quaternion  (N, 13, 3)
        expmap    = self._batch_quat_to_expmap(body_quat)  # (N, 13, 3)

        # ── Per-joint angle and range ────────────────────────────────────────
        # dof_pos: (N, 12) — joint angles for joints 1..12
        # Normalise angle to [0,1] within joint range
        lo  = self.joint_ranges[:, 0]   # (12,)
        hi  = self.joint_ranges[:, 1]   # (12,)
        span = (hi - lo).clamp(min=1e-8)
        angle_norm = (dof_pos - lo) / span   # (N, 12), in [0,1]

        # joint_range normalised to [-1,1] relative to pi
        jrange = self.jrange_norm.unsqueeze(0).expand(N, -1, -1)  # (N, 12, 2)

        # ── Assemble per-limb tokens ─────────────────────────────────────────
        # Token layout per limb:
        #   root (limb 0): xpos(3) xvelp(3) xvelr(3) expmap(3) [ltv(6)] angle(1) jrange(2) global(11)
        #   joint i (limb i+1): same layout, zeros for angle/jrange on root, zeros for global

        tokens = torch.zeros(N, max_limbs, self.limb_obs_size, device=self.device)

        for li in range(num_limbs):
            base = 0
            # xpos
            tokens[:, li, base:base+3] = xpos[:, li, :]
            base += 3
            # xvelp
            tokens[:, li, base:base+3] = xvelp[:, li, :]
            base += 3
            # xvelr
            tokens[:, li, base:base+3] = xvelr[:, li, :]
            base += 3
            # expmap
            tokens[:, li, base:base+3] = expmap[:, li, :]
            base += 3

            if self.use_ltv:
                if li == 0:   # root — no joint, all zeros
                    tokens[:, li, base:base+6] = 0.0
                else:
                    tokens[:, li, base:base+6] = self.ltv[li - 1]
                base += 6

            # angle (0 for root, normalised dof_pos for joints)
            if li == 0:
                tokens[:, li, base] = 0.0
            else:
                tokens[:, li, base] = angle_norm[:, li - 1]
            base += 1

            # joint range (zeros for root)
            if li == 0:
                tokens[:, li, base:base+2] = 0.0
            else:
                tokens[:, li, base:base+2] = jrange[:, li - 1, :]
            base += 2

            # global state — root only
            if li == 0:
                tokens[:, li, base:base+11] = global_state
            # else zeros (already zero-initialised)

        # ── Flatten to (N, max_limbs * limb_obs_size) ───────────────────────
        proprioceptive = tokens.reshape(N, max_limbs * self.limb_obs_size)

        # ── Store phase info for reward computation ──────────────────────────
        self._phase_left  = phase_left   # (N,)
        self._phase_right = phase_right  # (N,)

        return proprioceptive, phase_left, phase_right

    def _batch_quat_to_expmap(self, quat):
        """
        quat: (N, B, 4) [w,x,y,z]
        returns expmap: (N, B, 3)
        Exponential map = axis * angle, where angle = 2*acos(|w|).
        """
        w = quat[..., 0].clamp(-1 + 1e-7, 1 - 1e-7)
        xyz = quat[..., 1:]
        angle    = 2.0 * torch.acos(w.abs())           # (N, B)
        sin_half = torch.sin(angle / 2).clamp(min=1e-8)
        axis     = xyz / sin_half.unsqueeze(-1)
        return axis * angle.unsqueeze(-1)               # (N, B, 3)


# ---------------------------------------------------------------------------
# Reward computation  (batched PyTorch, mirrors _compute_rewards exactly)
# ---------------------------------------------------------------------------

class RewardComputer:
    """
    Batched reward computation matching G1RoughCfg scales and G1Robot terms.
    All terms multiplied by dt as in IsaacGym _prepare_reward_function.
    """

    SCALES = {
        'tracking_lin_vel':   1.0,
        'tracking_ang_vel':   0.5,
        'lin_vel_z':         -2.0,
        'ang_vel_xy':        -0.05,
        'orientation':       -1.0,
        'base_height':      -10.0,
        'dof_acc':          -2.5e-7,
        'dof_vel':          -1e-3,
        'action_rate':      -0.01,
        'dof_pos_limits':   -5.0,
        'alive':             0.15,
        'hip_pos':          -1.0,
        'contact_no_vel':   -0.2,
        'feet_swing_height':-20.0,
        'contact':           0.18,
    }

    BASE_HEIGHT_TARGET = 0.78
    TRACKING_SIGMA     = 0.25
    SOFT_DOF_LIMIT     = 0.9
    STANCE_THRESH      = 0.55

    def __init__(self, device, dof_pos_limits, feet_indices, dt):
        """
        dof_pos_limits: (12, 2) soft limits [lower, upper]
        feet_indices:   (2,)   Isaac Gym rigid body indices for L/R ankle_roll
        dt:             policy timestep = decimation * sim_dt
        """
        self.device     = device
        self.dt         = dt
        self.dof_lo     = dof_pos_limits[:, 0]   # (12,)
        self.dof_hi     = dof_pos_limits[:, 1]   # (12,)
        self.feet_idx   = feet_indices            # (2,) tensor

    def compute(self,
                root_states,       # (N, 13)
                dof_pos,           # (N, 12)
                dof_vel,           # (N, 12)
                last_dof_vel,      # (N, 12)
                actions,           # (N, 12)
                last_actions,      # (N, 12)
                commands,          # (N, 3)
                feet_states,       # (N, 2, 13)  feet rigid body states
                contact_forces,    # (N, num_bodies, 3)  net contact forces
                phase_left,        # (N,)
                phase_right,       # (N,)
                ):
        dt  = self.dt
        s   = self.SCALES
        N   = root_states.shape[0]

        quat          = root_states[:, 3:7]   # [w,x,y,z]
        height        = root_states[:, 2]     # (N,)

        # Body-frame velocities
        lin_vel_world = root_states[:, 7:10]
        ang_vel_world = root_states[:, 10:13]
        base_lin_vel  = quat_rotate_inverse_batch(quat, lin_vel_world)   # (N, 3)
        base_ang_vel  = quat_rotate_inverse_batch(quat, ang_vel_world)   # (N, 3)

        # Projected gravity
        grav = torch.tensor([0., 0., -1.], device=self.device).expand(N, 3)
        proj_gravity = quat_rotate_inverse_batch(quat, grav)             # (N, 3)

        # ── Foot contact (z-force > 1 N) ─────────────────────────────────
        # contact_forces: (N, num_bodies, 3), z-component at index 2
        left_fz  = contact_forces[:, self.feet_idx[0], 2].abs()   # (N,)
        right_fz = contact_forces[:, self.feet_idx[1], 2].abs()
        left_contact  = (left_fz  > 1.0).float()   # (N,)
        right_contact = (right_fz > 1.0).float()

        # Foot positions (z) and velocities
        left_foot_z  = feet_states[:, 0, 2]         # (N,)
        right_foot_z = feet_states[:, 1, 2]
        left_foot_vel  = feet_states[:, 0, 7:10]    # (N, 3)
        right_foot_vel = feet_states[:, 1, 7:10]

        # ── Reward terms ─────────────────────────────────────────────────

        # 1. tracking_lin_vel
        lin_err = ((commands[:, :2] - base_lin_vel[:, :2]) ** 2).sum(dim=1)
        r_tracking_lin_vel = torch.exp(-lin_err / self.TRACKING_SIGMA)

        # 2. tracking_ang_vel
        ang_err = (commands[:, 2] - base_ang_vel[:, 2]) ** 2
        r_tracking_ang_vel = torch.exp(-ang_err / self.TRACKING_SIGMA)

        # 3. lin_vel_z
        r_lin_vel_z = base_lin_vel[:, 2] ** 2

        # 4. ang_vel_xy
        r_ang_vel_xy = (base_ang_vel[:, :2] ** 2).sum(dim=1)

        # 5. orientation
        r_orientation = (proj_gravity[:, :2] ** 2).sum(dim=1)

        # 6. base_height
        r_base_height = (height - self.BASE_HEIGHT_TARGET) ** 2

        # 7. dof_acc
        r_dof_acc = ((dof_vel - last_dof_vel) / dt).pow(2).sum(dim=1)

        # 8. dof_vel
        r_dof_vel = dof_vel.pow(2).sum(dim=1)

        # 9. action_rate
        r_action_rate = (actions - last_actions).pow(2).sum(dim=1)

        # 10. dof_pos_limits
        out_lo = (self.dof_lo - dof_pos).clamp(min=0.0)
        out_hi = (dof_pos - self.dof_hi).clamp(min=0.0)
        r_dof_pos_limits = (out_lo + out_hi).sum(dim=1)

        # 11. alive
        r_alive = torch.ones(N, device=self.device)

        # 12. hip_pos  [yaw(0), roll(1), yaw(6), roll(7)]
        r_hip_pos = dof_pos[:, [0,1,6,7]].pow(2).sum(dim=1)

        # 13. contact  (gait-phase-aware)
        left_stance  = (phase_left  < self.STANCE_THRESH).float()
        right_stance = (phase_right < self.STANCE_THRESH).float()
        left_correct  = (left_contact  == left_stance).float()
        right_correct = (right_contact == right_stance).float()
        r_contact = left_correct + right_correct

        # 14. contact_no_vel
        l_pen = (left_foot_vel  ** 2).sum(dim=1) * left_contact
        r_pen = (right_foot_vel ** 2).sum(dim=1) * right_contact
        r_contact_no_vel = l_pen + r_pen

        # 15. feet_swing_height
        l_sw = (left_foot_z  - 0.08).pow(2) * (1.0 - left_contact)
        r_sw = (right_foot_z - 0.08).pow(2) * (1.0 - right_contact)
        r_feet_swing_height = l_sw + r_sw

        # ── Apply scales × dt ─────────────────────────────────────────────
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

        return total   # (N,)


# ---------------------------------------------------------------------------
# Main bridge class
# ---------------------------------------------------------------------------

class IsaacGymVecEnv:
    """
    Drop-in replacement for SubprocVecEnv + VecNormalize + VecPyTorch.

    ppo.py calls:
        obs  = self.envs.reset()
        obs, rew, done, infos = self.envs.step(act)
        self.envs.observation_space
        self.envs.action_space
        get_ob_rms(self.envs)   # for checkpointing
    All of these are implemented here.
    """

    def __init__(self, ig_env, graph_data: RobotGraphData,
                 device, training=True, norm_rew=True, gamma=0.99):
        self.env        = ig_env
        self.gd         = graph_data
        self.device     = device
        self.training   = training
        self.norm_rew   = norm_rew
        self.num_envs   = ig_env.num_envs
        self.num_actions = ig_env.num_actions   # 12

        # Policy timestep
        self.dt = ig_env.cfg.control.decimation * ig_env.sim_params.dt

        # ── Observation / action spaces ───────────────────────────────────
        limb_obs_size   = graph_data.obs_builder.limb_obs_size
        prop_dim        = cfg.MODEL.MAX_LIMBS * limb_obs_size
        ctx_dim         = len(graph_data.context[0])
        max_limbs       = cfg.MODEL.MAX_LIMBS
        max_joints      = cfg.MODEL.MAX_JOINTS

        obs_spaces = {
            'proprioceptive':   spaces.Box(-np.inf, np.inf, (prop_dim,),          np.float32),
            'context':          spaces.Box(-np.inf, np.inf, (ctx_dim,),            np.float32),
            'obs_padding_mask': spaces.Box(-np.inf, np.inf, (max_limbs,),          np.float32),
            'act_padding_mask': spaces.Box(-np.inf, np.inf, (max_limbs,),          np.float32),
            'edges':            spaces.Box(-np.inf, np.inf, (max_joints * 2,),     np.float32),
            'traversals':       spaces.Box(-np.inf, np.inf, (max_limbs,),          np.float32),
            'SWAT_RE':          spaces.Box(-np.inf, np.inf, (max_limbs, max_limbs, 3), np.float32),
        }
        if cfg.MODEL.GRAPH_ENCODING != "none":
            feat_dim = 7 if cfg.MODEL.GRAPH_ENCODING == "onehot" else 6
            obs_spaces['graph_node_features'] = spaces.Box(
                -np.inf, np.inf, (max_limbs, feat_dim), np.float32)
            obs_spaces['graph_A_norm'] = spaces.Box(
                0.0, 1.0, (max_limbs, max_limbs), np.float32)

        self.observation_space = spaces.Dict(obs_spaces)
        self.action_space = spaces.Box(
            low=-1.0, high=1.0, shape=(max_limbs,), dtype=np.float32)

        # ── Running mean/std for obs normalisation ────────────────────────
        self.ob_rms  = {'proprioceptive': RunningMeanStd(shape=(prop_dim,))}
        self.ret_rms = RunningMeanStd(shape=())
        self.clipob  = 10.0
        self.cliprew = 10.0
        self.gamma   = gamma
        self.ret     = torch.zeros(self.num_envs, device=device)

        # ── Episode tracking ──────────────────────────────────────────────
        self.episode_steps   = torch.zeros(self.num_envs, dtype=torch.long,  device=device)
        self.episode_returns = torch.zeros(self.num_envs, dtype=torch.float32, device=device)
        self.last_actions    = torch.zeros(self.num_envs, self.num_actions,  device=device)
        self.last_dof_vel    = torch.zeros(self.num_envs, self.num_actions,  device=device)
        self.commands        = torch.zeros(self.num_envs, 3, device=device)

        # ── Reward computer ───────────────────────────────────────────────
        dof_limits = self._build_soft_dof_limits()
        feet_idx   = self._find_feet_indices()
        self.reward_computer = RewardComputer(
            device=device,
            dof_pos_limits=dof_limits,
            feet_indices=feet_idx,
            dt=self.dt,
        )

        print(f"[IsaacGymVecEnv] {self.num_envs} envs | "
              f"prop_dim={prop_dim} | limb_obs_size={limb_obs_size} | "
              f"dt={self.dt:.4f}s")

    # ── Public interface ──────────────────────────────────────────────────────

    def reset(self):
        env_ids = torch.arange(self.num_envs, device=self.device)
        self.env.reset_idx(env_ids)
        self._resample_commands(env_ids)
        self.episode_steps[:]   = 0
        self.episode_returns[:] = 0
        self.last_actions[:]    = 0
        self.last_dof_vel[:]    = 0

        self.env.gym.simulate(self.env.sim)
        self.env.gym.fetch_results(self.env.sim, True)
        self.env.gym.refresh_actor_root_state_tensor(self.env.sim)
        self.env.gym.refresh_dof_state_tensor(self.env.sim)
        self.env.gym.refresh_rigid_body_state_tensor(self.env.sim)
        self.env.gym.refresh_net_contact_force_tensor(self.env.sim)

        obs = self._get_obs()
        obs = self._normalize_obs(obs)
        return obs

    def step(self, actions):
        """
        actions: (N, max_limbs) padded action tensor from policy.
        Strips padding, applies PD control via Isaac Gym, returns obs dict.
        """
        # Strip action padding — keep only real joints (indices 1..12, skip root pad)
        act_mask = self.gd.act_padding_mask[0]   # (max_limbs,) bool
        real_actions = actions[:, ~act_mask]      # (N, 12)
        real_actions = real_actions.clamp(-1.0, 1.0)

        # Store for action_rate reward
        self.last_dof_vel = self.env.dof_vel.clone()

        # Isaac Gym step (handles PD control + decimation internally)
        self.env.actions = real_actions
        for _ in range(self.env.cfg.control.decimation):
            self.env.torques = self.env._compute_torques(self.env.actions)
            self.env.gym.set_dof_actuation_force_tensor(
                self.env.sim,
                gymtorch.unwrap_tensor(self.env.torques))
            self.env.gym.simulate(self.env.sim)
            self.env.gym.fetch_results(self.env.sim, True)
            self.env.gym.refresh_dof_state_tensor(self.env.sim)

        # Refresh all state tensors
        self.env.gym.refresh_actor_root_state_tensor(self.env.sim)
        self.env.gym.refresh_net_contact_force_tensor(self.env.sim)
        self.env.gym.refresh_rigid_body_state_tensor(self.env.sim)

        # Compute rewards using our reward computer (not Isaac Gym's)
        rigid_body = self.env.rigid_body_states_view   # (N, num_bodies, 13)
        feet_states = rigid_body[:, self.reward_computer.feet_idx, :]  # (N, 2, 13)

        obs_raw, phase_left, phase_right = self._build_obs_raw()

        rewards = self.reward_computer.compute(
            root_states    = self.env.root_states[:self.num_envs],
            dof_pos        = self.env.dof_pos,
            dof_vel        = self.env.dof_vel,
            last_dof_vel   = self.last_dof_vel,
            actions        = real_actions,
            last_actions   = self.last_actions,
            commands       = self.commands,
            feet_states    = feet_states,
            contact_forces = self.env.contact_forces,
            phase_left     = phase_left,
            phase_right    = phase_right,
        )

        # Termination
        dones = self._check_termination()

        # Episode bookkeeping
        self.episode_steps   += 1
        self.episode_returns += rewards
        self.last_actions     = real_actions.clone()

        # Handle episode resets
        done_ids = dones.nonzero(as_tuple=False).flatten()
        infos    = self._build_infos(dones)
        if len(done_ids) > 0:
            self.env.reset_idx(done_ids)
            self._resample_commands(done_ids)
            self.episode_steps[done_ids]   = 0
            self.episode_returns[done_ids] = 0
            self.last_actions[done_ids]    = 0

        # Build and normalise obs
        obs = self._obs_from_raw(obs_raw)
        obs = self._normalize_obs(obs)

        # Normalise rewards
        if self.norm_rew and self.training:
            rewards = self._normalize_rew(rewards)

        return obs, rewards, dones.float(), infos

    # ── Internal helpers ──────────────────────────────────────────────────────

    def _build_obs_raw(self):
        """Build proprioceptive obs and return phase info."""
        rigid_body = self.env.rigid_body_states_view   # (N, num_bodies, 13)

        prop, phase_left, phase_right = self.gd.obs_builder.build(
            root_states       = self.env.root_states[:self.num_envs],
            dof_pos           = self.env.dof_pos,
            dof_vel           = self.env.dof_vel,
            rigid_body_states = rigid_body,
            last_actions      = self.last_actions,
            commands          = self.commands,
            episode_steps     = self.episode_steps,
            dt                = self.dt,
        )
        return prop, phase_left, phase_right

    def _get_obs(self):
        prop, _, _ = self._build_obs_raw()
        return self._obs_from_raw(prop)

    def _obs_from_raw(self, proprioceptive):
        """Assemble full obs dict from proprioceptive tensor."""
        N = self.num_envs
        obs = {
            'proprioceptive':   proprioceptive,
            'context':          self.gd.context.expand(N, -1),
            'edges':            self.gd.edges.expand(N, -1),
            'traversals':       self.gd.traversals.expand(N, -1),
            'SWAT_RE':          self.gd.swat_re.expand(N, -1, -1, -1),
            'obs_padding_mask': self.gd.obs_padding_mask.expand(N, -1),
            'act_padding_mask': self.gd.act_padding_mask.expand(N, -1),
        }
        if cfg.MODEL.GRAPH_ENCODING != "none":
            obs['graph_node_features'] = self.gd.graph_node_features.expand(N, -1, -1)
            obs['graph_A_norm']        = self.gd.graph_A_norm.expand(N, -1, -1)
        return obs

    def _normalize_obs(self, obs):
        if not self.training:
            return obs
        key = 'proprioceptive'
        prop = obs[key]
        prop_np = prop.cpu().numpy()
        self.ob_rms[key].update(prop_np)
        mean = torch.tensor(self.ob_rms[key].mean, dtype=torch.float32, device=self.device)
        var  = torch.tensor(self.ob_rms[key].var,  dtype=torch.float32, device=self.device)
        obs[key] = ((prop - mean) / (var + 1e-8).sqrt()).clamp(-self.clipob, self.clipob)
        return obs

    def _normalize_rew(self, rewards):
        rew_np = rewards.cpu().numpy()
        self.ret     = self.ret * self.gamma + rewards
        self.ret_rms.update(self.ret.cpu().numpy())
        rewards = rewards / (torch.tensor(self.ret_rms.var, device=self.device).sqrt() + 1e-8)
        return rewards.clamp(-self.cliprew, self.cliprew)

    def _check_termination(self):
        """Mirrors IsaacGym check_termination + our height/angle bounds."""
        root = self.env.root_states[:self.num_envs]
        height = root[:, 2]
        quat   = root[:, 3:7]

        # Convert Isaac Gym [x,y,z,w] to [w,x,y,z]
        q_wxyz = torch.cat([quat[:, 3:4], quat[:, 0:3]], dim=1)
        roll, pitch, _ = quat_to_euler_batch(q_wxyz)

        # Pelvis (root) contact
        pelvis_fz = self.env.contact_forces[:, 0, 2].abs()
        pelvis_contact = pelvis_fz > 1.0

        height_fail = (height < 0.4) | (height > 1.15)
        pitch_fail  = pitch.abs() > 1.0
        roll_fail   = roll.abs()  > 0.8

        # Episode timeout
        timeout = self.episode_steps >= 1000

        return (pelvis_contact | height_fail | pitch_fail | roll_fail | timeout)

    def _resample_commands(self, env_ids):
        """Resample velocity commands for specified envs."""
        n = len(env_ids)
        self.commands[env_ids, 0] = torch.FloatTensor(n).uniform_(0.0, 1.0).to(self.device)
        self.commands[env_ids, 1] = torch.FloatTensor(n).uniform_(-0.5, 0.5).to(self.device)
        self.commands[env_ids, 2] = torch.FloatTensor(n).uniform_(-0.5, 0.5).to(self.device)
        # Zero out small xy commands
        small = self.commands[env_ids, :2].norm(dim=1) < 0.2
        self.commands[env_ids[small], :2] = 0.0

    def _build_infos(self, dones):
        """Build info list matching what ppo.py expects."""
        infos = []
        done_np = dones.cpu().numpy()
        for i in range(self.num_envs):
            info = {'name': 'g1'}
            if done_np[i]:
                info['episode'] = {
                    'r': float(self.episode_returns[i].item()),
                    'l': int(self.episode_steps[i].item()),
                }
            infos.append(info)
        return infos

    def _build_soft_dof_limits(self):
        """Build soft joint position limits from Isaac Gym DOF properties."""
        dof_props = self.env.gym.get_actor_dof_properties(
            self.env.envs[0], self.env.actor_handles[0])
        limits = torch.zeros(self.num_actions, 2, device=self.device)
        for i in range(self.num_actions):
            lo = float(dof_props['lower'][i])
            hi = float(dof_props['upper'][i])
            m  = (lo + hi) * 0.5
            r  = (hi - lo)
            limits[i, 0] = m - 0.5 * r * RewardComputer.SOFT_DOF_LIMIT
            limits[i, 1] = m + 0.5 * r * RewardComputer.SOFT_DOF_LIMIT
        return limits

    def _find_feet_indices(self):
        """Find rigid body indices for left and right ankle_roll bodies."""
        body_names = self.env.gym.get_actor_rigid_body_names(
            self.env.envs[0], self.env.actor_handles[0])
        left_idx  = next(i for i, n in enumerate(body_names) if 'left_ankle_roll'  in n.lower())
        right_idx = next(i for i, n in enumerate(body_names) if 'right_ankle_roll' in n.lower())
        return torch.tensor([left_idx, right_idx], dtype=torch.long, device=self.device)

    # ── Compatibility shims for ppo.py ────────────────────────────────────────

    def get_unimal_idx(self):
        return [0] * self.num_envs

    def close(self):
        pass

    @property
    def venv(self):
        return self   # makes get_vec_normalize() work


# ---------------------------------------------------------------------------
# Factory  — call this from envs.py
# ---------------------------------------------------------------------------

def make_isaacgym_vec_env(xml_paths, training=True, norm_rew=True):
    """
    xml_paths: list of MuJoCo XML paths for your robot variants.
               Used only to extract static graph structure.
               Physics runs from the Isaac Gym URDF.

    Returns an IsaacGymVecEnv that ppo.py can use directly.
    """
    from isaacgym import gymtorch, gymapi, gymutil
    from legged_gym.envs.g1.g1_env import G1Robot
    from legged_gym.envs.g1.g1_config import G1RoughCfg
    from legged_gym.utils.task_registry import task_registry

    device_str = cfg.DEVICE
    device     = torch.device(device_str)

    # ── Initialise Isaac Gym ──────────────────────────────────────────────
    sim_params = gymapi.SimParams()
    sim_params.up_axis = gymapi.UP_AXIS_Z
    sim_params.gravity = gymapi.Vec3(0.0, 0.0, -9.81)
    sim_params.dt      = 0.005
    sim_params.substeps = 1
    sim_params.use_gpu_pipeline = True
    sim_params.physx.use_gpu  = True
    sim_params.physx.num_threads = 10
    sim_params.physx.solver_type = 1
    sim_params.physx.num_position_iterations = 4
    sim_params.physx.num_velocity_iterations = 0
    sim_params.physx.contact_offset    = 0.01
    sim_params.physx.rest_offset       = 0.0
    sim_params.physx.bounce_threshold_velocity = 0.5
    sim_params.physx.max_depenetration_velocity = 1.0

    env_cfg = G1RoughCfg()
    env_cfg.env.num_envs = cfg.PPO.NUM_ENVS

    ig_env = G1Robot(
        cfg           = env_cfg,
        sim_params    = sim_params,
        physics_engine= gymapi.SIM_PHYSX,
        sim_device    = device_str,
        headless      = True,
    )

    # ── Build static graph data from first XML ────────────────────────────
    # All your variants have the same morphology so one XML is enough.
    # If variants differ, build one RobotGraphData per variant and batch them.
    xml_path  = xml_paths[0] if isinstance(xml_paths, (list, tuple)) else xml_paths

    # Build joint ranges from Isaac Gym DOF props (more accurate than XML parse)
    dof_props = ig_env.gym.get_actor_dof_properties(ig_env.envs[0], ig_env.actor_handles[0])
    joint_ranges = torch.tensor(
        [[float(dof_props['lower'][i]), float(dof_props['upper'][i])]
         for i in range(ig_env.num_dof)],
        dtype=torch.float32, device=device)   # (12, 2)

    graph_data = RobotGraphData(xml_path, device)
    graph_data.obs_builder = ObsBuilder(graph_data, device, joint_ranges)

    vec_env = IsaacGymVecEnv(
        ig_env     = ig_env,
        graph_data = graph_data,
        device     = device,
        training   = training,
        norm_rew   = norm_rew,
        gamma      = cfg.PPO.GAMMA,
    )

    return vec_env