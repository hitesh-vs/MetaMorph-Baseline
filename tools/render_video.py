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
    agent_name:     the name of the robot to collect trajectories for (e.g. "humanoid_2d_9_full")
    model_path:     path to the .pt model file
    agent_path:     path to the folder containing xml/ and metadata/ for the robot
    policy_folder:  path to the folder containing config.yaml (usually same as model_path dir)
    num_episodes:   how many episodes to collect
    output_file:    path to save the .pkl output (defaults to {agent_name}_trajectories.pkl)
    terminate_on_fall: whether to early stop if agent falls
    deterministic:  whether to use mean action or sample from distribution
    '''

    # -------------------------------------------------------------------------
    # 1. Config setup — mirrors evaluate.py exactly
    # -------------------------------------------------------------------------
    cfg.merge_from_file(f'{policy_folder}/config.yaml')
    cfg.PPO.CHECKPOINT_PATH = model_path
    cfg.ENV.WALKERS = [agent_name]
    cfg.ENV.WALKER_DIR = agent_path
    cfg.OUT_DIR = './eval'
    cfg.TERMINATE_ON_FALL = terminate_on_fall
    cfg.DETERMINISTIC = deterministic
    cfg.PPO.NUM_ENVS = 1           # Single env for trajectory collection
    cfg.VECENV.TYPE = "DummyVecEnv"
    set_cfg_options()              # Registers envs, sets MAX_LIMBS/JOINTS etc.
    register_modular_envs()

    # -------------------------------------------------------------------------
    # 2. Load policy — via PPO() just like evaluate.py, not manual torch.load
    # -------------------------------------------------------------------------
    print("Loading model...")
    ppo_trainer = PPO()
    policy = ppo_trainer.agent
    policy.ac.eval()  # Eval mode (disables dropout etc.)

    # -------------------------------------------------------------------------
    # 3. Create environment and set normalization stats from training
    # -------------------------------------------------------------------------
    print("Creating environment...")

    env = make_vec_envs(
        xml_file=agent_name,
        training=False,
        norm_rew=False,
        render_policy=True,
    )
    set_ob_rms(env, get_ob_rms(ppo_trainer.envs))  # Use normalization from training

    # -------------------------------------------------------------------------
    # 4. Collect trajectories
    # -------------------------------------------------------------------------
    episodes_data = {
        'agent_name': agent_name,
        'episodes': [],
        'ob_rms': get_ob_rms(ppo_trainer.envs),  # Save normalization stats for replay
    }

    print(f"\nCollecting trajectories for agent: {agent_name}")
    print("=" * 60)

    for ep in range(num_episodes):
        print(f"\nEpisode {ep + 1}/{num_episodes}")
        print("-" * 40)

        obs = env.reset()
        episode_states = []
        episode_reward = 0.0
        episode_length = 0
        not_done = True

        for t in range(2000):  # Max steps — same cap as evaluate.py
            # Capture MuJoCo sim state before stepping
            try:
                sim = env.envs[0].env.env.sim
                episode_states.append({
                    'qpos': sim.data.qpos.copy(),
                    'qvel': sim.data.qvel.copy(),
                })
            except Exception as e:
                print(f"  Warning: Could not capture state at step {t}: {e}")
                break

            # Get action — use policy.act() exactly like evaluate.py
            with torch.no_grad():
                _, action, _, _, _ = policy.act(
                    obs,
                    return_attention=False,
                    compute_val=False,
                )

            obs, reward, done, infos = env.step(action)

            # Track reward from info dict if available (like evaluate.py does)
            if done[0]:
                episode_reward = infos[0]['episode']['r']
                episode_length = infos[0]['episode']['l']
                not_done = False
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

    # -------------------------------------------------------------------------
    # 5. Save to pkl
    # -------------------------------------------------------------------------
    if output_file is None:
        output_file = f'{agent_name}_trajectories.pkl'

    print(f"\nSaving trajectories to {output_file}...")
    with open(output_file, 'wb') as f:
        pickle.dump(episodes_data, f)

    file_size_mb = os.path.getsize(output_file) / (1024 * 1024)

    # Summary
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
    print("\nDownload this file and run the replay script locally to generate videos.")

    return output_file


if __name__ == "__main__":
    # Example command:
    # python tools/render_video.py \
    #   --agent humanoid_2d_9_full \
    #   --policy_path output/example/1409 \
    #   --agent_path unimals_100/train \
    #   --num_episodes 10

    parser = argparse.ArgumentParser(description="Collect trajectories from a trained model")
    parser.add_argument("--agent",        required=True,  type=str, help="Robot name (e.g. humanoid_2d_9_full)")
    parser.add_argument("--policy_path",  required=True,  type=str, help="Path to folder containing config.yaml and .pt model")
    parser.add_argument("--agent_path",   required=True,  type=str, help="Path to robot folder containing xml/ dir")
    parser.add_argument("--policy_name",  default="Modular-v0", type=str, help="Model filename without .pt")
    parser.add_argument("--num_episodes", default=10,     type=int)
    parser.add_argument("--output_file",  default=None,   type=str)
    parser.add_argument("--terminate_on_fall",  action="store_true")
    parser.add_argument("--deterministic",      action="store_true")
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