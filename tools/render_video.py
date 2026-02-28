import argparse
import os
import torch
import numpy as np
import pickle

from metamorph.config import cfg
from metamorph.algos.ppo.ppo import PPO
from metamorph.algos.ppo.envs import make_vec_envs, set_ob_rms, get_ob_rms
from metamorph.algos.ppo.model import Agent

from tools.train_ppo import set_cfg_options, register_modular_envs


def _get_sim(env):
    """
    Unwraps whatever wrapper stack the env has to get the mujoco_py sim object.
    Works regardless of how many wrappers are stacked.
    """
    # DummyVecEnv: env.envs[0] is the outermost wrapper
    inner = env.envs[0]
    # Unwrap until we find something with .sim
    while not hasattr(inner, 'sim'):
        if hasattr(inner, 'env'):
            inner = inner.env
        else:
            return None
    return inner.sim


def save_episode_trajectories(
    agent_name,
    model_path,
    agent_path,
    policy_folder,
    num_episodes=10,
    output_file=None,
    terminate_on_fall=True,
    deterministic=False,
):
    '''
    agent_name:     the name of the robot to collect trajectories for (e.g. "g1_12dof")
    model_path:     path to the .pt model file
    agent_path:     path to the folder containing xml/ and metadata/ for the robot
    policy_folder:  path to the folder containing config.yaml
    num_episodes:   how many episodes to collect
    output_file:    path to save the .pkl output (defaults to {agent_name}_trajectories.pkl)
    terminate_on_fall: whether to early stop if agent falls
    deterministic:  whether to use mean action or sample from distribution
    '''

    # ── 1. Config setup ───────────────────────────────────────────────────────
    cfg.merge_from_file(f'{policy_folder}/config.yaml')
    cfg.PPO.CHECKPOINT_PATH = model_path
    cfg.ENV.WALKERS = [agent_name]
    cfg.ENV.WALKER_DIR = agent_path
    cfg.OUT_DIR = './eval'
    cfg.TERMINATE_ON_FALL = terminate_on_fall
    cfg.DETERMINISTIC = deterministic
    cfg.PPO.NUM_ENVS = 1
    cfg.VECENV.TYPE = "DummyVecEnv"
    set_cfg_options()
    register_modular_envs()

    # ── 2. Load policy ────────────────────────────────────────────────────────
    print("Loading model...")
    ppo_trainer = PPO()
    policy = ppo_trainer.agent
    policy.ac.eval()

    # ── 3. Create environment ─────────────────────────────────────────────────
    print("Creating environment...")
    env = make_vec_envs(
        xml_file=agent_name,
        training=False,
        norm_rew=False,
        render_policy=True,
    )
    set_ob_rms(env, get_ob_rms(ppo_trainer.envs))

    # ── 4. Collect trajectories ───────────────────────────────────────────────
    episodes_data = {
        'agent_name': agent_name,
        'episodes':   [],
        'ob_rms':     get_ob_rms(ppo_trainer.envs),
    }

    print(f"\nCollecting trajectories for agent: {agent_name}")
    print("=" * 60)

    for ep in range(num_episodes):
        print(f"\nEpisode {ep + 1}/{num_episodes}")
        print("-" * 40)

        obs = env.reset()
        episode_states  = []
        episode_reward  = 0.0
        episode_length  = 0

        for t in range(2000):
            # Capture MuJoCo sim state
            sim = _get_sim(env)
            if sim is not None:
                episode_states.append({
                    'qpos': sim.data.qpos.copy(),
                    'qvel': sim.data.qvel.copy(),
                })
            else:
                print(f"  Warning: could not access sim at step {t}")

            with torch.no_grad():
                _, action, _, _, _ = policy.act(
                    obs,
                    return_attention=False,
                    compute_val=False,
                )

            obs, reward, done, infos = env.step(action)

            if done[0]:
                episode_reward = infos[0].get('episode', {}).get('r', episode_reward)
                episode_length = infos[0].get('episode', {}).get('l', t + 1)
                break
            else:
                episode_reward += float(reward[0])
                episode_length += 1

            if episode_length % 100 == 0:
                print(f"  Step {episode_length}, reward so far: {episode_reward:.2f}")

        episodes_data['episodes'].append({
            'states': episode_states,
            'reward': episode_reward,
            'length': episode_length,
        })

        print(f"  Collected {len(episode_states)} states | "
              f"Reward: {episode_reward:.2f} | Length: {episode_length}")

    env.close()

    # ── 5. Save ───────────────────────────────────────────────────────────────
    if output_file is None:
        output_file = f'{agent_name}_trajectories.pkl'

    print(f"\nSaving trajectories to {output_file}...")
    with open(output_file, 'wb') as f:
        pickle.dump(episodes_data, f)

    file_size_mb = os.path.getsize(output_file) / (1024 * 1024)
    rewards = [ep['reward'] for ep in episodes_data['episodes']]

    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"File:            {output_file} ({file_size_mb:.2f} MB)")
    print(f"Total episodes:  {num_episodes}")
    print(f"Mean Reward:     {np.mean(rewards):7.2f} ± {np.std(rewards):6.2f}")
    print(f"Min Reward:      {np.min(rewards):7.2f}")
    print(f"Max Reward:      {np.max(rewards):7.2f}")
    print("=" * 60)
    print("\nDownload the .pkl and run replay_local.py to generate video.")

    return output_file


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Collect trajectories from a trained model")
    parser.add_argument("--agent",             required=True,  type=str)
    parser.add_argument("--policy_path",       required=True,  type=str, help="Folder with config.yaml and .pt")
    parser.add_argument("--agent_path",        required=True,  type=str, help="Folder with xml/ and metadata/")
    parser.add_argument("--policy_name",       default="Modular-v0", type=str)
    parser.add_argument("--num_episodes",      default=10,     type=int)
    parser.add_argument("--output_file",       default=None,   type=str)
    parser.add_argument("--terminate_on_fall", action="store_true")
    parser.add_argument("--deterministic",     action="store_true")
    args = parser.parse_args()

    model_path = os.path.join(args.policy_path, args.policy_name + '.pt')

    output = save_episode_trajectories(
        agent_name=args.agent,
        model_path=model_path,
        agent_path=args.agent_path,
        policy_folder=args.policy_path,
        num_episodes=args.num_episodes,
        output_file=args.output_file,
        terminate_on_fall=args.terminate_on_fall,
        deterministic=args.deterministic,
    )

    print(f"\n✓ Done! Download {output} to your local machine.")