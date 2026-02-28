import os
import random

import gym
import numpy as np
from gym import spaces
from gym import utils

from gym.spaces import Box
from gym.spaces import Dict

from metamorph.config import cfg


def _pad_graph_obs(obs: dict, max_limbs: int) -> dict:
    """
    Pads graph_node_features (N, feat_dim) → (MAX_LIMBS, feat_dim)
    and graph_A_norm         (N, N)        → (MAX_LIMBS, MAX_LIMBS).

    Called at the end of every observation() method when GRAPH_ENCODING != "none".
    Padding with zeros is safe: padded rows in A_norm have no connections,
    so GCN output for those nodes is zero, and they're masked by obs_padding_mask.
    """
    if cfg.MODEL.GRAPH_ENCODING == "none":
        return obs
    if "graph_node_features" not in obs:
        return obs

    X = obs["graph_node_features"]   # (N, feat_dim)
    A = obs["graph_A_norm"]          # (N, N)
    N, feat_dim = X.shape

    if N < max_limbs:
        X_pad = np.zeros((max_limbs, feat_dim), dtype=np.float32)
        X_pad[:N, :] = X

        A_pad = np.zeros((max_limbs, max_limbs), dtype=np.float32)
        A_pad[:N, :N] = A

        obs["graph_node_features"] = X_pad
        obs["graph_A_norm"]        = A_pad
    # If N == max_limbs already, nothing to do

    return obs


def _graph_obs_spaces(max_limbs: int) -> dict:
    """
    Returns the two extra observation space entries for graph keys.
    Called in __init__ when GRAPH_ENCODING != "none".
    """
    feat_dim = 7 if cfg.MODEL.GRAPH_ENCODING == "onehot" else 6  # onehot=7, topological=6
    inf = np.float32(np.inf)
    return {
        "graph_node_features": Box(-inf, inf, (max_limbs, feat_dim), np.float32),
        "graph_A_norm":        Box(0.0,  1.0,  (max_limbs, max_limbs), np.float32),
    }


class ModularObservationPadding(gym.ObservationWrapper):

    def __init__(self, env):
        super().__init__(env)

        self.max_limbs = cfg.MODEL.MAX_LIMBS
        self.max_joints = cfg.MODEL.MAX_JOINTS

        num_limbs = self.metadata["num_limbs"]
        self.num_limb_pads = self.max_limbs - num_limbs

        sample_obs = env.reset()
        
        if isinstance(sample_obs, dict):
            self.limb_obs_size = len(sample_obs['proprioceptive']) // num_limbs
            self.context_size  = len(sample_obs['context']) // num_limbs
        else:
            self.limb_obs_size = sample_obs.shape[0] // num_limbs
            self.context_size  = self.limb_obs_size

        inf = np.float32(np.inf)
        obs_spaces = dict()
        
        obs_spaces['proprioceptive']     = Box(-inf, inf, (self.limb_obs_size * self.max_limbs,), np.float32)
        obs_spaces['context']            = Box(-inf, inf, (self.context_size  * self.max_limbs,), np.float32)
        obs_spaces['obs_padding_mask']   = Box(-inf, inf, (self.max_limbs,), np.float32)
        obs_spaces['act_padding_mask']   = Box(-inf, inf, (self.max_limbs,), np.float32)
        obs_spaces['edges']              = Box(-inf, inf, (self.max_joints * 2,), np.float32)
        obs_spaces['traversals']         = Box(-inf, inf, (self.max_limbs,), np.float32)
        obs_spaces['SWAT_RE']            = Box(-inf, inf, (self.max_limbs, self.max_limbs, 3), np.float32)

        # ── Graph obs spaces (only when GCN is active) ───────────────────
        if cfg.MODEL.GRAPH_ENCODING != "none":
            obs_spaces.update(_graph_obs_spaces(self.max_limbs))

        self.observation_space = Dict(obs_spaces)

        # Padding masks
        obs_padding_mask = [False] * num_limbs + [True] * self.num_limb_pads
        self.obs_padding_mask = np.asarray(obs_padding_mask)

        act_padding_mask = [True] + [False] * (num_limbs - 1) + [True] * self.num_limb_pads
        self.act_padding_mask = np.asarray(act_padding_mask)

    def observation(self, obs):
        if isinstance(obs, dict):
            proprioceptive = obs['proprioceptive']
            context        = obs['context']
            edges          = obs['edges']
            traversals     = obs.get('traversals', list(range(self.metadata['num_limbs'])))
            SWAT_RE        = obs.get('SWAT_RE', np.zeros([cfg.MODEL.MAX_LIMBS, cfg.MODEL.MAX_LIMBS, 3]))
        else:
            proprioceptive = obs
            context        = obs
            edges          = np.zeros(self.max_joints * 2)
            traversals     = list(range(self.metadata['num_limbs']))
            SWAT_RE        = np.zeros([cfg.MODEL.MAX_LIMBS, cfg.MODEL.MAX_LIMBS, 3])

        # Pad proprioceptive
        prop_padding         = np.zeros(self.limb_obs_size * self.num_limb_pads)
        proprioceptive_padded = np.concatenate([proprioceptive, prop_padding]).ravel()

        # Pad context
        context_padding  = np.zeros(self.context_size * self.num_limb_pads)
        context_padded   = np.concatenate([context, context_padding]).ravel()

        # Pad edges
        edges_padded = np.zeros(self.max_joints * 2)
        edges_padded[:len(edges)] = edges

        # Pad traversals
        traversals_padded = np.zeros(self.max_limbs)
        traversals_array  = np.array(traversals)
        traversals_padded[:len(traversals_array)] = traversals_array

        obs_dict = {
            "proprioceptive":  proprioceptive_padded,
            "context":         context_padded,
            "obs_padding_mask": self.obs_padding_mask,
            "act_padding_mask": self.act_padding_mask,
            "edges":           edges_padded,
            "traversals":      traversals_padded,
            "SWAT_RE":         SWAT_RE,
        }

        # ── Pass through graph keys and pad them ─────────────────────────
        if cfg.MODEL.GRAPH_ENCODING != "none" and isinstance(obs, dict):
            obs_dict["graph_node_features"] = obs.get("graph_node_features",
                np.zeros((self.metadata['num_limbs'],
                          7 if cfg.MODEL.GRAPH_ENCODING == "onehot" else 6), np.float32))
            obs_dict["graph_A_norm"] = obs.get("graph_A_norm",
                np.zeros((self.metadata['num_limbs'], self.metadata['num_limbs']), np.float32))
            obs_dict = _pad_graph_obs(obs_dict, self.max_limbs)

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
        self.act_padding_mask[0] = True

    def observation(self, obs):
        if isinstance(obs, dict):
            proprioceptive = obs['proprioceptive']
            context        = obs['context']
            edges          = obs['edges']
            traversals     = obs.get('traversals', list(range(self.metadata['num_limbs'])))
            SWAT_RE        = obs.get('SWAT_RE', np.zeros([cfg.MODEL.MAX_LIMBS, cfg.MODEL.MAX_LIMBS, 3]))
        else:
            proprioceptive = obs
            context        = obs
            edges          = np.zeros(self.max_joints * 2)
            traversals     = list(range(self.metadata['num_limbs']))
            SWAT_RE        = np.zeros([cfg.MODEL.MAX_LIMBS, cfg.MODEL.MAX_LIMBS, 3])

        # Reshape to per-limb, place at consistent indices
        proprioceptive_per_limb = proprioceptive.reshape(-1, self.limb_obs_size)
        context_per_limb        = context.reshape(-1, self.context_size)

        proprioceptive_padded = np.zeros([self.max_limbs, self.limb_obs_size])
        proprioceptive_padded[self.limb_index] = proprioceptive_per_limb

        context_padded = np.zeros([self.max_limbs, self.context_size])
        context_padded[self.limb_index] = context_per_limb

        proprioceptive_padded = proprioceptive_padded.ravel()
        context_padded        = context_padded.ravel()

        # Pad edges
        edges_padded = np.zeros(self.max_joints * 2)
        edges_padded[:len(edges)] = edges

        # Pad traversals
        traversals_padded = np.zeros(self.max_limbs)
        traversals_array  = np.array(traversals)
        traversals_padded[:len(traversals_array)] = traversals_array

        obs_dict = {
            "proprioceptive":   proprioceptive_padded,
            "context":          context_padded,
            "obs_padding_mask": self.obs_padding_mask,
            "act_padding_mask": self.act_padding_mask,
            "edges":            edges_padded,
            "traversals":       traversals_padded,
            "SWAT_RE":          SWAT_RE,
        }

        # ── Pass through graph keys with consistent-index placement ──────
        # Note: for consistent padding, graph features stay at their natural
        # node indices (0..N-1). They are NOT reindexed to limb_index because
        # the GCN operates on the robot's own graph, not the global vocab.
        # The obs_padding_mask already tells the transformer which slots to ignore.
        if cfg.MODEL.GRAPH_ENCODING != "none" and isinstance(obs, dict):
            obs_dict["graph_node_features"] = obs.get("graph_node_features",
                np.zeros((self.metadata['num_limbs'],
                          7 if cfg.MODEL.GRAPH_ENCODING == "onehot" else 6), np.float32))
            obs_dict["graph_A_norm"] = obs.get("graph_A_norm",
                np.zeros((self.metadata['num_limbs'], self.metadata['num_limbs']), np.float32))
            obs_dict = _pad_graph_obs(obs_dict, self.max_limbs)

        return obs_dict


class ModularActionPadding(gym.ActionWrapper):

    def __init__(self, env):
        super().__init__(env)
        self.max_limbs  = cfg.MODEL.MAX_LIMBS
        self.max_joints = cfg.MODEL.MAX_LIMBS
        self.num_limb_pads = self.max_limbs - self.metadata["num_limbs"]
        self._update_action_space()

        act_padding_mask = [True] + [False] * self.metadata["num_joints"] + [True] * self.num_limb_pads
        self.act_padding_mask = np.asarray(act_padding_mask)

    def _update_action_space(self):
        num_pads = self.max_limbs - self.metadata["num_limbs"]
        low, high = self.action_space.low, self.action_space.high
        low  = np.concatenate([[-1.], low,  [-1.] * num_pads]).astype(np.float32)
        high = np.concatenate([[ 1.], high, [ 1.] * num_pads]).astype(np.float32)
        self.action_space = spaces.Box(low=low, high=high, dtype=np.float32)

    def action(self, action):
        return action[~self.act_padding_mask]


class ConsistentModularActionPadding(gym.ActionWrapper):

    def __init__(self, env):
        super().__init__(env)
        self.max_limbs  = cfg.MODEL.MAX_LIMBS
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
        if 0 in self.joint_index:
            self.joint_index.remove(0)

        self._update_action_space()

        self.act_padding_mask = np.asarray([True] * self.max_limbs)
        self.act_padding_mask[self.joint_index] = False

    def _update_action_space(self):
        low  = -1. * np.ones(self.max_limbs, dtype=np.float32)
        high =  1. * np.ones(self.max_limbs, dtype=np.float32)
        self.action_space = spaces.Box(low=low, high=high, dtype=np.float32)

    def action(self, action):
        return action[~self.act_padding_mask]