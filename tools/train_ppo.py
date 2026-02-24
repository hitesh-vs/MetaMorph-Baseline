import argparse
import os
import sys

import torch

from gym.envs.registration import register

from metamorph.algos.ppo.ppo import PPO
from metamorph.config import cfg
from metamorph.config import dump_cfg
from metamorph.utils import file as fu
from metamorph.utils import sample as su
from metamorph.utils import sweep as swu


def set_cfg_options():
    calculate_max_iters()
    maybe_infer_walkers()
    calculate_max_limbs_joints()


def calculate_max_limbs_joints():
    """
    Calculate MAX_LIMBS and MAX_JOINTS based on robot type.
    
    LIMITATION FOR REAL URDFs: This function uses hardcoded string matching
    on cfg.ENV.WALKER_DIR to determine padding sizes. This breaks when:
    1. New robot URDFs don't match the hardcoded patterns
    2. Robot names use different conventions
    3. Multiple robot types are mixed
    
    RESEARCH NOTE: This is a key limitation you're addressing with learned
    functional groupings - no more manual specification needed.
    """
    if cfg.ENV_NAME == "Unimal-v0":

        num_joints, num_limbs = [], []

        metadata_paths = []
        for agent in cfg.ENV.WALKERS:
            metadata_paths.append(os.path.join(
                cfg.ENV.WALKER_DIR, "metadata", "{}.json".format(agent)
            ))

        for metadata_path in metadata_paths:
            metadata = fu.load_json(metadata_path)
            num_joints.append(metadata["dof"])
            num_limbs.append(metadata["num_limbs"] + 1)

        # Add extra 1 for max_joints; needed for adding edge padding
        cfg.MODEL.MAX_JOINTS = max(num_joints) + 1
        cfg.MODEL.MAX_LIMBS = max(num_limbs) + 1
        cfg.MODEL.MAX_JOINTS = 16
        cfg.MODEL.MAX_LIMBS = 12
        print (cfg.MODEL.MAX_JOINTS, cfg.MODEL.MAX_LIMBS)
    
    elif cfg.ENV_NAME == 'Modular-v0':
        # HARDCODED ROBOT CONFIGURATIONS
        # This is a key limitation: each new real URDF requires manual configuration
        
        if 'hopper' in cfg.ENV.WALKER_DIR:
            cfg.MODEL.MAX_LIMBS = 5
            cfg.MODEL.MAX_JOINTS = 5
            
        if 'walker' in cfg.ENV.WALKER_DIR:
            cfg.MODEL.MAX_LIMBS = 7
            cfg.MODEL.MAX_JOINTS = 7
            
        if 'humanoid' in cfg.ENV.WALKER_DIR:
            cfg.MODEL.MAX_LIMBS = 9
            cfg.MODEL.MAX_JOINTS = 9
        
        # REAL URDF SUPPORT: Unitree G1 (12-DOF humanoid)
        # Manual configuration required - demonstrates need for learned groupings
        if 'unitree_g1' in cfg.ENV.WALKER_DIR or 'g1_12dof' in cfg.ENV.WALKER_DIR:
            cfg.MODEL.MAX_LIMBS = 13  # pelvis + 12 actuated limbs
            cfg.MODEL.MAX_JOINTS = 12  # 12 actuated joints (free joint not counted)
            print(f"[Real URDF] Configured for Unitree G1: MAX_LIMBS=13, MAX_JOINTS=12")
        
        # Multi-robot training (requires consistent padding)
        if 'all' in cfg.ENV.WALKER_DIR:
            if cfg.MODEL.MLP.CONSISTENT_PADDING:
                cfg.MODEL.MAX_LIMBS = 19
                cfg.MODEL.MAX_JOINTS = 19
            else:
                cfg.MODEL.MAX_LIMBS = 9
                cfg.MODEL.MAX_JOINTS = 9
        
        # Debug output to verify configuration
        print(f"[Config] ENV.WALKER_DIR: {cfg.ENV.WALKER_DIR}")
        print(f"[Config] MAX_LIMBS: {cfg.MODEL.MAX_LIMBS}, MAX_JOINTS: {cfg.MODEL.MAX_JOINTS}")
        print(f"[Config] WALKERS: {cfg.ENV.WALKERS}")


def calculate_max_iters():
    # Iter here refers to 1 cycle of experience collection and policy update.
    cfg.PPO.MAX_ITERS = (
        int(cfg.PPO.MAX_STATE_ACTION_PAIRS) // cfg.PPO.TIMESTEPS // cfg.PPO.NUM_ENVS
    )
    cfg.PPO.EARLY_EXIT_MAX_ITERS = (
        int(cfg.PPO.EARLY_EXIT_STATE_ACTION_PAIRS) // cfg.PPO.TIMESTEPS // cfg.PPO.NUM_ENVS
    )


def maybe_infer_walkers():
    if cfg.ENV_NAME not in ["Unimal-v0", "Modular-v0"]:
        return

    # Only infer the walkers if this option was not specified
    if len(cfg.ENV.WALKERS):
        return

    cfg.ENV.WALKERS = [
        xml_file.split(".")[0]
        for xml_file in os.listdir(os.path.join(cfg.ENV.WALKER_DIR, "xml"))
    ]

    if cfg.ENV_NAME == 'Modular-v0':
        register_modular_envs()


def register_modular_envs():
    """register the MuJoCo envs with Gym and return the per-agent observation size and max action value (for modular policy training)"""
    # register each env
    for agent in cfg.ENV.WALKERS:
        xml = os.path.join(cfg.ENV.WALKER_DIR, 'xml', agent + '.xml')
        params = {"xml": os.path.abspath(xml)}
        try:
            register(
                id=f"{agent}-v0",
                max_episode_steps=1000,
                entry_point=f"modular.{agent}:make_env",
                kwargs=params,
            )
            print(f"[Gym Registration] Registered {agent}-v0")
        except Exception as e:
            print(f"[Gym Registration] Failed to register {agent}-v0: {e}")
            continue


def get_hparams():
    hparam_path = os.path.join(cfg.OUT_DIR, "hparam.json")
    # For local sweep return
    if not os.path.exists(hparam_path):
        return {}

    hparams = {}
    varying_args = fu.load_json(hparam_path)
    flatten_cfg = swu.flatten(cfg)

    for k in varying_args:
        hparams[k] = flatten_cfg[k]

    return hparams


def cleanup_tensorboard():
    tb_dir = os.path.join(cfg.OUT_DIR, "tensorboard")

    # Assume there is only one sub_dir and break when it's found
    for content in os.listdir(tb_dir):
        content = os.path.join(tb_dir, content)
        if os.path.isdir(content):
            break

    # Return if no dir found
    if not os.path.isdir(content):
        return

    # Move all the event files from sub_dir to tb_idr
    for event_file in os.listdir(content):
        src = os.path.join(content, event_file)
        dst = os.path.join(tb_dir, event_file)
        fu.move_file(src, dst)

    # Delete the sub_dir
    os.rmdir(content)


def parse_args():
    """Parses the arguments."""
    parser = argparse.ArgumentParser(description="Train a RL agent")
    parser.add_argument(
        "--cfg", dest="cfg_file", help="Config file", required=True, type=str
    )
    parser.add_argument(
        "--no_context_in_state", action="store_true"
    )
    parser.add_argument(
        "opts",
        help="See morphology/core/config.py for all options",
        default=None,
        nargs=argparse.REMAINDER,
    )
    if len(sys.argv) == 1:
        parser.print_help()
        sys.exit(1)
    return parser.parse_args()


def ppo_train():
    su.set_seed(cfg.RNG_SEED)
    # Configure the CUDNN backend
    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = cfg.CUDNN.BENCHMARK
        torch.backends.cudnn.deterministic = cfg.CUDNN.DETERMINISTIC

    torch.set_num_threads(1)
    
    print("="*80)
    print("STARTING PPO TRAINING")
    print("="*80)
    
    PPOTrainer = PPO()
    PPOTrainer.train()
    hparams = get_hparams()
    PPOTrainer.save_rewards(hparams=hparams)
    PPOTrainer.save_model(-1)
    cleanup_tensorboard()


def main():
    # Parse cmd line args
    args = parse_args()

    # Load config options
    cfg.merge_from_file(args.cfg_file)
    cfg.merge_from_list(args.opts)

    if args.no_context_in_state:
        obs_type = [
            "body_xpos", "body_xvelp", "body_xvelr", "body_xquat", # limb
            "qpos", "qvel", # joint
        ]
        ob_opts = ["MODEL.PROPRIOCEPTIVE_OBS_TYPES", obs_type]
        cfg.merge_from_list(ob_opts)

    # Set cfg options which are inferred
    set_cfg_options()
    os.makedirs(cfg.OUT_DIR, exist_ok=True)

    # Save the config
    dump_cfg()
    ppo_train()


if __name__ == "__main__":
    main()