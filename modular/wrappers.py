import os
import random

import gym
import numpy as np
from gym import spaces
from gym import utils

from gym.spaces import Box
from gym.spaces import Dict

from metamorph.config import cfg


class ModularObservationPadding(gym.ObservationWrapper):

    def __init__(self, env):
        super().__init__(env)

        self.max_limbs = cfg.MODEL.MAX_LIMBS
        self.max_joints = cfg.MODEL.MAX_JOINTS

        num_limbs = self.metadata["num_limbs"]
        self.num_limb_pads = self.max_limbs - num_limbs

        # Since _get_obs now returns a dict, we need to handle this properly
        # Get a sample observation to determine sizes
        sample_obs = env.reset()
        
        if isinstance(sample_obs, dict):
            # Already dictionary format from ModularEnv
            self.limb_obs_size = len(sample_obs['proprioceptive']) // num_limbs
            self.context_size = len(sample_obs['context']) // num_limbs
        else:
            # Fallback for flat array
            self.limb_obs_size = sample_obs.shape[0] // num_limbs
            self.context_size = self.limb_obs_size

        inf = np.float32(np.inf)
        self.observation_space = dict()
        
        # Proprioceptive observation space
        prop_shape = (self.limb_obs_size * self.max_limbs,)
        self.observation_space['proprioceptive'] = Box(-inf, inf, prop_shape, np.float32)
        
        # Context observation space
        context_shape = (self.context_size * self.max_limbs,)
        self.observation_space['context'] = Box(-inf, inf, context_shape, np.float32)
        
        # Padding masks
        self.observation_space['obs_padding_mask'] = Box(-inf, inf, (self.max_limbs,), np.float32)
        self.observation_space['act_padding_mask'] = Box(-inf, inf, (self.max_limbs,), np.float32)
        
        # Graph structure
        self.observation_space['edges'] = Box(-inf, inf, (self.max_joints * 2,), np.float32)
        self.observation_space['traversals'] = Box(-inf, inf, (self.max_limbs,), np.float32)
        self.observation_space['SWAT_RE'] = Box(-inf, inf, (self.max_limbs, self.max_limbs, 3), np.float32)
        
        self.observation_space = Dict(self.observation_space)

        # Create padding masks
        obs_padding_mask = [False] * num_limbs + [True] * self.num_limb_pads
        self.obs_padding_mask = np.asarray(obs_padding_mask)

        act_padding_mask = [True] + [False] * (num_limbs - 1) + [True] * self.num_limb_pads
        self.act_padding_mask = np.asarray(act_padding_mask)

    def observation(self, obs):
        """Pad observations to max size"""
        
        if isinstance(obs, dict):
            # obs is already a dictionary from ModularEnv
            proprioceptive = obs['proprioceptive']
            context = obs['context']
            edges = obs['edges']
            traversals = obs.get('traversals', list(range(self.metadata['num_limbs'])))
            SWAT_RE = obs.get('SWAT_RE', np.zeros([cfg.MODEL.MAX_LIMBS, cfg.MODEL.MAX_LIMBS, 3]))
        else:
            # Fallback for flat array (shouldn't happen with new ModularEnv)
            proprioceptive = obs
            context = obs
            edges = np.zeros(self.max_joints * 2)
            traversals = list(range(self.metadata['num_limbs']))
            SWAT_RE = np.zeros([cfg.MODEL.MAX_LIMBS, cfg.MODEL.MAX_LIMBS, 3])
        
        # Pad proprioceptive observations
        prop_padding = np.zeros(self.limb_obs_size * self.num_limb_pads)
        proprioceptive_padded = np.concatenate([proprioceptive, prop_padding]).ravel()
        
        # Pad context observations
        context_padding = np.zeros(self.context_size * self.num_limb_pads)
        context_padded = np.concatenate([context, context_padding]).ravel()
        
        # Pad edges to max_joints * 2
        edges_padded = np.zeros(self.max_joints * 2)
        edges_padded[:len(edges)] = edges
        
        # Pad traversals to max_limbs
        traversals_padded = np.zeros(self.max_limbs)
        traversals_array = np.array(traversals)
        traversals_padded[:len(traversals_array)] = traversals_array
        
        obs_dict = {
            "proprioceptive": proprioceptive_padded,
            "context": context_padded,
            "obs_padding_mask": self.obs_padding_mask,
            "act_padding_mask": self.act_padding_mask,
            "edges": edges_padded,
            "traversals": traversals_padded,
            "SWAT_RE": SWAT_RE,
        }
        
        return obs_dict

    def reset(self, **kwargs):
        observation = self.env.reset(**kwargs)
        return self.observation(observation)


class ConsistentModularObservationPadding(ModularObservationPadding):

    def __init__(self, env):
        super().__init__(env)

        agent_limb_names = env.agent_limb_names
        if 'all_train' in cfg.ENV.WALKER_DIR:
            full_limb_names = [
                'torso', 'thigh', 'leg', 'lower_leg', 'foot', 
                'left1', 'left2', 'left3', 'right1', 'right2', 'right3', 
                'right_thigh', 'right_shin', 'left_thigh', 'left_shin', 
                'right_upper_arm', 'right_lower_arm', 'left_upper_arm', 'left_lower_arm', 
            ]
        else:
            full_limb_names = env.full_limb_names
        
        self.limb_index = [full_limb_names.index(name) for name in agent_limb_names]

        self.obs_padding_mask = np.asarray([True] * self.max_limbs)
        self.obs_padding_mask[self.limb_index] = False

        self.act_padding_mask = np.asarray([True] * self.max_limbs)
        self.act_padding_mask[self.limb_index] = False
        # torso/pelvis has no action
        self.act_padding_mask[0] = True

    def observation(self, obs):
        """Pad observations with consistent indexing across different morphologies"""
        
        if isinstance(obs, dict):
            proprioceptive = obs['proprioceptive']
            context = obs['context']
            edges = obs['edges']
            traversals = obs.get('traversals', list(range(self.metadata['num_limbs'])))
            SWAT_RE = obs.get('SWAT_RE', np.zeros([cfg.MODEL.MAX_LIMBS, cfg.MODEL.MAX_LIMBS, 3]))
        else:
            # Fallback
            proprioceptive = obs
            context = obs
            edges = np.zeros(self.max_joints * 2)
            traversals = list(range(self.metadata['num_limbs']))
            SWAT_RE = np.zeros([cfg.MODEL.MAX_LIMBS, cfg.MODEL.MAX_LIMBS, 3])
        
        # Reshape observations to per-limb format
        proprioceptive_per_limb = proprioceptive.reshape(-1, self.limb_obs_size)
        context_per_limb = context.reshape(-1, self.context_size)
        
        # Create padded arrays with consistent indexing
        proprioceptive_padded = np.zeros([self.max_limbs, self.limb_obs_size])
        proprioceptive_padded[self.limb_index] = proprioceptive_per_limb
        
        context_padded = np.zeros([self.max_limbs, self.context_size])
        context_padded[self.limb_index] = context_per_limb
        
        # Flatten back
        proprioceptive_padded = proprioceptive_padded.ravel()
        context_padded = context_padded.ravel()
        
        # Pad edges
        edges_padded = np.zeros(self.max_joints * 2)
        edges_padded[:len(edges)] = edges
        
        # Pad traversals
        traversals_padded = np.zeros(self.max_limbs)
        traversals_array = np.array(traversals)
        traversals_padded[:len(traversals_array)] = traversals_array

        obs_dict = {
            "proprioceptive": proprioceptive_padded,
            "context": context_padded,
            "obs_padding_mask": self.obs_padding_mask,
            "act_padding_mask": self.act_padding_mask,
            "edges": edges_padded,
            "traversals": traversals_padded,
            "SWAT_RE": SWAT_RE,
        }

        return obs_dict


class ModularActionPadding(gym.ActionWrapper):

    def __init__(self, env):
        super().__init__(env)
        self.max_limbs = cfg.MODEL.MAX_LIMBS
        self.max_joints = cfg.MODEL.MAX_LIMBS
        self.num_limb_pads = self.max_limbs - self.metadata["num_limbs"]
        self._update_action_space()
        
        # Create action padding mask
        act_padding_mask = [True] + [False] * self.metadata["num_joints"] + [True] * self.num_limb_pads
        self.act_padding_mask = np.asarray(act_padding_mask)

    def _update_action_space(self):
        num_pads = self.max_limbs - self.metadata["num_limbs"]
        low, high = self.action_space.low, self.action_space.high
        low = np.concatenate([[-1.], low, [-1.] * num_pads]).astype(np.float32)
        high = np.concatenate([[1.], high, [1.] * num_pads]).astype(np.float32)
        self.action_space = spaces.Box(low=low, high=high, dtype=np.float32)

    def action(self, action):
        """Remove padding from actions before passing to environment"""
        new_action = action[~self.act_padding_mask]
        return new_action


class ConsistentModularActionPadding(gym.ActionWrapper):

    def __init__(self, env):
        super().__init__(env)
        self.max_limbs = cfg.MODEL.MAX_LIMBS
        self.max_joints = cfg.MODEL.MAX_LIMBS

        agent_limb_names = env.agent_limb_names
        if 'all_train' in cfg.ENV.WALKER_DIR:
            full_limb_names = [
                'torso', 'thigh', 'leg', 'lower_leg', 'foot', 
                'left1', 'left2', 'left3', 'right1', 'right2', 'right3', 
                'right_thigh', 'right_shin', 'left_thigh', 'left_shin', 
                'right_upper_arm', 'right_lower_arm', 'left_upper_arm', 'left_lower_arm', 
            ]
        else:
            full_limb_names = env.full_limb_names
        
        self.joint_index = [full_limb_names.index(name) for name in agent_limb_names]
        # torso/pelvis has no joint action
        if 0 in self.joint_index:
            self.joint_index.remove(0)

        self._update_action_space()

        self.act_padding_mask = np.asarray([True] * self.max_limbs)
        self.act_padding_mask[self.joint_index] = False

    def _update_action_space(self):
        low = -1. * np.ones(self.max_limbs, dtype=np.float32)
        high = 1. * np.ones(self.max_limbs, dtype=np.float32)
        self.action_space = spaces.Box(low=low, high=high, dtype=np.float32)

    def action(self, action):
        """Remove padding from actions before passing to environment"""
        new_action = action[~self.act_padding_mask]
        return new_action