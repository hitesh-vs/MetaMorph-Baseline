"""
modular/g1_12dof.py  —  Phase 3 (walking)

Changes from Phase 2:
  - Phase 3 reward: tracking + dof_vel + action_rate + contact + feet_swing + contact_no_vel enabled
                    r_alive reduced 2.0 -> 0.5
  - Per-component reward accumulated over episode and logged in info['episode']
  - FOOT_BODIES replaced by topology-derived _find_foot_bodies()

Curriculum history:
  Phase 1: alive + orientation(0.2x) + height(2.0x)
  Phase 2: alive=2.0 + orientation + height + ang_vel_xy + lin_vel_z + hip_pos
  Phase 3: all terms active, alive=0.5

Reward component map (legged_gym scale -> our per-step value after xDT=0.02):
  tracking_lin_vel  -> 1.0*DT  * exp(-||fwd_vel - target||^2 / 0.25)
  lin_vel_z         -> -2.0*DT  * root_velz^2
  ang_vel_xy        -> -0.05*DT * (pitch_rate^2 + roll_rate^2)
  orientation       -> -1.0*DT  * (sin(pitch)^2 + sin(roll)^2)
  base_height       -> -10.0*DT * (height - init_height)^2
  dof_vel           -> -1e-3*DT * sum(qvel[6:]^2)
  dof_acc           -> -2.5e-7*DT * sum(accel^2)
  action_rate       -> -0.01*DT * sum((a - last_a)^2)
  hip_pos           -> -1.0*DT  * sum(hip_roll^2 + hip_yaw^2)
  contact           -> +0.18*DT * gait_phase_match (per foot)
  feet_swing        -> -20.0*DT * (foot_z - 0.08)^2 * swinging
  contact_no_vel    -> -0.2*DT  * foot_vel^2 * in_contact
  alive             -> +0.5 per step (flat, not DT-scaled)
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


# dt scaling: matches legged_gym scale*dt inside _prepare_reward_function
DT = 0.02

GAIT_PERIOD   = 0.8
GAIT_OFFSET   = 0.5
STANCE_THRESH = 0.55

# G1 hip roll/yaw qpos indices (free joint = qpos[0:7])
HIP_ROLL_YAW_QPOS = (8, 9, 14, 15)

TARGET_FORWARD_VEL = 1.0


class ModularEnv(mujoco_env.MujocoEnv, utils.EzPickle):

    def __init__(self, xml):
        self.xml = xml

        self._robot_name = os.path.splitext(os.path.basename(xml))[0]
        if self._robot_name.endswith('_stripped'):
            self._robot_name = self._robot_name[:-9]

        self.graph_encoding = cfg.MODEL.GRAPH_ENCODING

        self.edges            = np.array([])
        self.traversals       = np.array([])
        self.SWAT_RE          = np.zeros([cfg.MODEL.MAX_LIMBS, cfg.MODEL.MAX_LIMBS, 3])
        self.context_features = np.array([])
        self._graph_node_features = None
        self._graph_A_norm        = None
        self._graph_built         = False

        self._root_body   = self._parse_root_body(xml)
        self._init_height = self._parse_init_height(xml)

        self._last_qvel      = None
        self._last_action    = None
        self._episode_step   = 0
        self._episode_return = 0.0
        self._episode_time   = 0.0

        self._ep_r_tracking    = 0.0
        self._ep_r_lin_vel_z   = 0.0
        self._ep_r_ang_vel_xy  = 0.0
        self._ep_r_orientation = 0.0
        self._ep_r_height      = 0.0
        self._ep_r_dof_vel     = 0.0
        self._ep_r_dof_acc     = 0.0
        self._ep_r_action_rate = 0.0
        self._ep_r_hip         = 0.0
        self._ep_r_contact     = 0.0
        self._ep_r_swing       = 0.0
        self._ep_r_no_vel      = 0.0
        self._ep_r_alive       = 0.0
        self._ep_forward_vel   = 0.0

        self._foot_body_ids     = []
        self._foot_names        = []
        self._hip_roll_yaw_qpos = []

        mujoco_env.MujocoEnv.__init__(self, xml, 4)
        utils.EzPickle.__init__(self)

        num_limbs  = len(self.model.body_names) - 1
        num_joints = self.sim.model.nu
        self.metadata['num_limbs']  = num_limbs
        self.metadata['num_joints'] = num_joints

        self._root_body   = self.model.body_names[1]
        self._init_height = float(self.init_qpos[2])

        self.full_limb_names  = list(self.model.body_names[1:])
        self.agent_limb_names = list(self.model.body_names[1:])

        self._foot_names    = self._find_foot_bodies()
        self._foot_body_ids = [self.sim.model.body_name2id(f) for f in self._foot_names]
        self._hip_roll_yaw_qpos = list(HIP_ROLL_YAW_QPOS)

        self._build_graph_structure()
        self._compute_context_encoding()

        if self.graph_encoding != "none":
            self._init_graph_data()

        self._graph_built = True

    # ------------------------------------------------------------------
    # Topology-derived foot detection
    # ------------------------------------------------------------------

    def _find_foot_bodies(self):
        """Leaf bodies with contact geoms — no name parsing needed."""
        num_bodies = len(self.model.body_names)
        has_children = set()
        for i in range(1, num_bodies):
            has_children.add(self.sim.model.body_parentid[i])
        feet = []
        for i in range(1, num_bodies):
            if i in has_children:
                continue
            for g in range(self.sim.model.ngeom):
                if (self.sim.model.geom_bodyid[g] == i and
                        self.sim.model.geom_contype[g] > 0):
                    feet.append(self.model.body_names[i])
                    break
        return tuple(feet)

    # ------------------------------------------------------------------
    # Pre-init XML helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _parse_root_body(xml_path):
        try:
            root = ET.parse(xml_path).getroot()
            wb = root.find('worldbody')
            if wb is not None:
                b = wb.find('body')
                if b is not None:
                    return b.attrib.get('name', 'pelvis')
        except Exception:
            pass
        return 'pelvis'

    @staticmethod
    def _parse_init_height(xml_path):
        try:
            root = ET.parse(xml_path).getroot()
            wb = root.find('worldbody')
            if wb is not None:
                b = wb.find('body')
                if b is not None:
                    pos_str = b.attrib.get('pos', '0 0 0.8')
                    z = float(pos_str.strip().split()[2])
                    return z if z > 0.01 else 0.8
        except Exception:
            pass
        return 0.8

    # ------------------------------------------------------------------
    # Graph
    # ------------------------------------------------------------------

    def _init_graph_data(self):
        parser = MujocoGraphParser(self.xml)
        self._graph_node_features = parser.get_features(self.graph_encoding)
        self._graph_A_norm        = parser.normalized_adjacency()
        print(f"[{self._robot_name}|{self.graph_encoding}] "
              f"Graph: {parser.N} nodes, features {self._graph_node_features.shape}")

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
        gd = np.zeros([cfg.MODEL.MAX_LIMBS, cfg.MODEL.MAX_LIMBS, 3])
        kids = {i: [] for i in range(len(parents))}
        for i, p in enumerate(parents):
            if p >= 0: kids[p].append(i)
        for i in range(len(parents)):
            for j in range(len(parents)):
                if   i == j:          gd[i, j] = [1, 0, 0]
                elif parents[i] == j: gd[i, j] = [0, 1, 0]
                elif j in kids[i]:    gd[i, j] = [0, 0, 1]
        return gd

    # ------------------------------------------------------------------
    # Context encoding
    # ------------------------------------------------------------------

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
                    if self.sim.model.geom_bodyid[g] == bi), None) for bi in body_idxs]
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
                    if self.sim.model.geom_bodyid[g] == bi), None) for bi in body_idxs]
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

    # ------------------------------------------------------------------
    # Observation
    # ------------------------------------------------------------------

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

        # Gait phase on root only; zeros elsewhere (keeps per-limb dim uniform)
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
            return np.concatenate([xpos, xvelp, xvelr, expmap, [angle], joint_range,
                                   sin_phase, cos_phase]) # Removed the ohe just to run the baseline without it 
                                                          # (add ltv later, after expmap)
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
    
    # ------------------------------------------------------------------
    # Labels for t-SNE plot
    # ------------------------------------------------------------------

    def get_limb_labels(self):
        depth = {}
        for i in range(1, len(self.model.body_names)):
            pid = self.sim.model.body_parentid[i]
            depth[i] = depth.get(pid, 0) + 1

        labels = []
        for i, bname in enumerate(self.model.body_names[1:], start=1):
            n = bname.lower()
            side  = ("left"   if ("left"  in n or "_l_" in n or n.startswith("l_")) else
                    "right"  if ("right" in n or "_r_" in n or n.startswith("r_")) else
                    "center")
            jtype = ("hip_pitch" if "hip_pitch" in n else
                    "hip_roll"  if "hip_roll"  in n else
                    "hip_yaw"   if "hip_yaw"   in n else
                    "knee"      if "knee"      in n else
                    "ankle"     if "ankle"     in n else
                    "root"      if ("pelvis"   in n or "torso" in n) else
                    "other")
            labels.append({
                "robot":    self._robot_name,
                "body":     bname,
                "semantic": f"{side}_{jtype}",
                "depth":    depth[i],
            })
        return labels

    # ------------------------------------------------------------------
    # Reward helpers
    # ------------------------------------------------------------------

    def _foot_contact_forces(self):
        if not self._foot_body_ids:
            return np.array([])
        return np.array([abs(self.sim.data.cfrc_ext[bid, 5])
                         for bid in self._foot_body_ids])

    def _foot_positions(self):
        if not self._foot_names:
            return np.zeros((0, 3))
        pos = []
        for fname in self._foot_names:
            try:    pos.append(self.data.get_body_xpos(fname).copy())
            except: pos.append(np.zeros(3))
        return np.array(pos)

    def _foot_velocities(self):
        if not self._foot_names:
            return np.zeros((0, 3))
        vel = []
        for fname in self._foot_names:
            try:    vel.append(np.clip(self.data.get_body_xvelp(fname), -10, 10))
            except: vel.append(np.zeros(3))
        return np.array(vel)

    def _gait_phase(self, t):
        phase_left  = (t % GAIT_PERIOD) / GAIT_PERIOD
        phase_right = (phase_left + GAIT_OFFSET) % 1.0
        return phase_left < STANCE_THRESH, phase_right < STANCE_THRESH

    def _rew_tracking_lin_vel(self, fwd):
        return 1.0 * DT * float(np.exp(-((fwd - TARGET_FORWARD_VEL)**2) / 0.25))

    def _rew_lin_vel_z(self, vz):
        return -2.0 * DT * float(vz**2)

    def _rew_ang_vel_xy(self, pr, rr):
        return -0.05 * DT * float(pr**2 + rr**2)

    def _rew_orientation(self, pitch, roll):
        return -1.0 * DT * float(np.sin(pitch)**2 + np.sin(roll)**2)

    def _rew_base_height(self, h):
        return -10.0 * DT * float((h - self._init_height)**2)

    def _rew_dof_vel(self):
        return -1e-3 * DT * float(np.square(self.data.qvel[6:]).sum())

    def _rew_dof_acc(self, last_qvel):
        if last_qvel is None: return 0.0
        acc = (self.data.qvel[6:] - last_qvel[6:]) / self.dt
        return -2.5e-7 * DT * float(np.square(acc).sum())

    def _rew_action_rate(self, a, last_a):
        if last_a is None: return 0.0
        return -0.01 * DT * float(np.square(a - last_a).sum())

    def _rew_hip_pos(self):
        if not self._hip_roll_yaw_qpos: return 0.0
        return -1.0 * DT * float(np.square(self.data.qpos[self._hip_roll_yaw_qpos]).sum())

    def _rew_contact(self, t):
        cf = self._foot_contact_forces()
        if len(cf) == 0: return 0.0
        in_contact = cf > 1.0
        is_stance  = list(self._gait_phase(t))
        score = sum(1.0 for i in range(min(len(in_contact), len(is_stance)))
                    if not (bool(in_contact[i]) ^ bool(is_stance[i])))
        return 0.18 * DT * score

    def _rew_feet_swing_height(self, t):
        cf = self._foot_contact_forces()
        if len(cf) == 0: return 0.0
        in_contact = cf > 1.0
        foot_pos   = self._foot_positions()
        penalty = sum((float(foot_pos[i, 2]) - 0.08)**2
                      for i in range(min(len(foot_pos), len(in_contact)))
                      if not in_contact[i])
        return -20.0 * DT * penalty

    def _rew_contact_no_vel(self, t):
        cf = self._foot_contact_forces()
        if len(cf) == 0: return 0.0
        in_contact = cf > 1.0
        foot_vels  = self._foot_velocities()
        penalty = sum(float(np.square(foot_vels[i]).sum())
                      for i in range(min(len(foot_vels), len(in_contact)))
                      if in_contact[i])
        return -0.2 * DT * penalty

    # ------------------------------------------------------------------
    # Episode accumulator reset
    # ------------------------------------------------------------------

    def _reset_ep_accumulators(self):
        (self._ep_r_tracking, self._ep_r_lin_vel_z, self._ep_r_ang_vel_xy,
         self._ep_r_orientation, self._ep_r_height, self._ep_r_dof_vel,
         self._ep_r_dof_acc, self._ep_r_action_rate, self._ep_r_hip,
         self._ep_r_contact, self._ep_r_swing, self._ep_r_no_vel,
         self._ep_r_alive, self._ep_forward_vel) = (0.0,) * 14

    # ------------------------------------------------------------------
    # Step
    # ------------------------------------------------------------------

    def step(self, a):
        last_qvel   = self.data.qvel.copy() if self._episode_step > 0 else None
        last_action = self._last_action.copy() if self._last_action is not None else None

        posbefore = self.sim.data.qpos[0]
        self.do_simulation(a, self.frame_skip)
        posafter  = self.sim.data.qpos[0]

        height      = float(self.data.get_body_xpos(self._root_body)[2])
        quat        = self.data.get_body_xquat(self._root_body)
        pitch       = float(2 * np.arcsin(np.clip(2*(quat[0]*quat[2] - quat[3]*quat[1]), -1, 1)))
        roll        = float(2 * np.arcsin(np.clip(2*(quat[0]*quat[1] + quat[2]*quat[3]), -1, 1)))
        root_angvel = self.data.get_body_xvelr(self._root_body)
        pitch_rate  = float(root_angvel[1])
        roll_rate   = float(root_angvel[0])
        root_velz   = float(self.data.get_body_xvelp(self._root_body)[2])
        forward_vel = (posafter - posbefore) / self.dt

        self._episode_time += self.dt

        # Phase 3: all terms active
        r_tracking    = self._rew_tracking_lin_vel(forward_vel)
        r_lin_vel_z   = self._rew_lin_vel_z(root_velz)
        r_ang_vel_xy  = self._rew_ang_vel_xy(pitch_rate, roll_rate)
        r_orientation = self._rew_orientation(pitch, roll)
        r_height      = self._rew_base_height(height)
        r_dof_vel     = self._rew_dof_vel()
        r_dof_acc     = self._rew_dof_acc(last_qvel)
        r_action_rate = self._rew_action_rate(a, last_action)
        r_hip         = self._rew_hip_pos()
        r_contact     = self._rew_contact(self._episode_time)
        r_swing       = self._rew_feet_swing_height(self._episode_time)
        r_no_vel      = self._rew_contact_no_vel(self._episode_time)
        r_alive       = 0.5   # flat, reduced from Phase 2's 2.0

        reward = (r_tracking + r_lin_vel_z + r_ang_vel_xy + r_orientation
                + r_height + r_dof_vel + r_dof_acc + r_action_rate
                + r_hip + r_contact + r_swing + r_no_vel + r_alive)

        # Accumulate per-component totals for episode logging
        self._ep_r_tracking    += r_tracking
        self._ep_r_lin_vel_z   += r_lin_vel_z
        self._ep_r_ang_vel_xy  += r_ang_vel_xy
        self._ep_r_orientation += r_orientation
        self._ep_r_height      += r_height
        self._ep_r_dof_vel     += r_dof_vel
        self._ep_r_dof_acc     += r_dof_acc
        self._ep_r_action_rate += r_action_rate
        self._ep_r_hip         += r_hip
        self._ep_r_contact     += r_contact
        self._ep_r_swing       += r_swing
        self._ep_r_no_vel      += r_no_vel
        self._ep_r_alive       += r_alive
        self._ep_forward_vel   += forward_vel

        height_ok = (height > 0.50 * self._init_height and
                     height < 1.50 * self._init_height)
        done = not (height_ok and abs(pitch) < 0.8 and abs(roll) < 0.8)

        self._last_qvel      = self.data.qvel.copy()
        self._last_action    = a.copy()
        self._episode_step   += 1
        self._episode_return += reward

        info = {
            'name':        self._robot_name,
            'forward_vel': forward_vel,
            'height':      height,
            'pitch':       pitch,
            'roll':        roll,
        }

        if done or self._episode_step >= 1000:
            steps = max(self._episode_step, 1)
            info['episode'] = {
                'r':                self._episode_return,
                'l':                self._episode_step,
                'rew/tracking':     self._ep_r_tracking,
                'rew/lin_vel_z':    self._ep_r_lin_vel_z,
                'rew/ang_vel_xy':   self._ep_r_ang_vel_xy,
                'rew/orientation':  self._ep_r_orientation,
                'rew/height':       self._ep_r_height,
                'rew/dof_vel':      self._ep_r_dof_vel,
                'rew/dof_acc':      self._ep_r_dof_acc,
                'rew/action_rate':  self._ep_r_action_rate,
                'rew/hip':          self._ep_r_hip,
                'rew/contact':      self._ep_r_contact,
                'rew/swing':        self._ep_r_swing,
                'rew/no_vel':       self._ep_r_no_vel,
                'rew/alive':        self._ep_r_alive,
                'mean_forward_vel': self._ep_forward_vel / steps,
            }
            self._episode_step   = 0
            self._episode_return = 0.0
            self._episode_time   = 0.0
            self._last_qvel      = None
            self._last_action    = None
            self._reset_ep_accumulators()

        return self._get_obs(), reward, done, info

    # ------------------------------------------------------------------
    # Reset / Viewer
    # ------------------------------------------------------------------

    def reset_model(self):
        self._episode_step   = 0
        self._episode_return = 0.0
        self._episode_time   = 0.0
        self._last_qvel      = None
        self._last_action    = None
        self._reset_ep_accumulators()

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
    print(f"Made env for {env._robot_name} | encoding={cfg.MODEL.GRAPH_ENCODING} | "
          f"limbs={env.metadata['num_limbs']} joints={env.metadata['num_joints']} | "
          f"root={env._root_body} init_height={env._init_height:.3f}m | "
          f"feet={env._foot_names}")
    if cfg.MODEL.MLP.CONSISTENT_PADDING:
        env = ConsistentModularObservationPadding(env)
        env = ConsistentModularActionPadding(env)
    else:
        env = ModularObservationPadding(env)
        env = ModularActionPadding(env)
    return env