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
        mujoco_env.MujocoEnv.__init__(self, xml, 4)
        utils.EzPickle.__init__(self)

        # Unitree G1 12-DOF: 13 bodies, 12 joints
        self.metadata['num_limbs'] = 13
        self.metadata['num_joints'] = 12

        self.agent_limb_names = self.model.body_names[1:]
        
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

    def _get_obs(self):
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

        full_obs = np.concatenate(
            [_get_obs_per_limb(b) for b in self.model.body_names[1:]]
        )
        return full_obs.ravel()

    def step(self, a):
        posbefore = self.sim.data.qpos[0]
        self.do_simulation(a, self.frame_skip)
        
        posafter = self.sim.data.qpos[0]
        height = self.sim.data.qpos[2]
        
        quat = self.sim.data.qpos[3:7]
        pitch = 2 * np.arcsin(np.clip(2 * (quat[0] * quat[2] - quat[3] * quat[1]), -1, 1))
        roll = 2 * np.arcsin(np.clip(2 * (quat[0] * quat[1] + quat[2] * quat[3]), -1, 1))
        
        alive_bonus = 1.0
        forward_reward = (posafter - posbefore) / self.dt
        ctrl_cost = 1e-3 * np.square(a).sum()
        
        reward = forward_reward + alive_bonus - ctrl_cost
        
        done = not (
            height > 0.4 and
            height < 1.2 and
            abs(pitch) < 0.8 and
            abs(roll) < 0.8
        )
        
        info = {
            'name': 'unitree_g1_12dof',
            'forward_reward': forward_reward,
            'ctrl_cost': ctrl_cost,
            'height': height,
            'pitch': pitch,
            'roll': roll,
        }
        
        return self._get_obs(), reward, done, info

    def reset_model(self):
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
        return self._get_obs()

    def viewer_setup(self):
        self.viewer.cam.trackbodyid = 1
        self.viewer.cam.distance = self.model.stat.extent * 2.0
        self.viewer.cam.lookat[2] = 0.8
        self.viewer.cam.elevation = -20


def make_env(xml):
    """Create environment - xml is the path to XML file"""
    env = ModularEnv(xml)
    
    if cfg.MODEL.MLP.CONSISTENT_PADDING:
        env = ConsistentModularObservationPadding(env)
        env = ConsistentModularActionPadding(env)
    else:
        env = ModularObservationPadding(env)
        env = ModularActionPadding(env)
    
    return env