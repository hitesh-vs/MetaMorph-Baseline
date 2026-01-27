# import numpy as np
# import torch
# import pickle
# from metamorph.config import cfg
# from metamorph.algos.ppo.envs import make_vec_envs, set_ob_rms
# from metamorph.algos.ppo.model import Agent


# def load_model(checkpoint_path):
#     checkpoint = torch.load(checkpoint_path, map_location='cpu')
#     actor_critic = checkpoint[0]
#     ob_rms = checkpoint[1]

#     device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#     actor_critic.to(device)
#     actor_critic.eval()

#     return actor_critic, ob_rms, device


# def collect_and_save_rollouts(
#     agent_name,
#     checkpoint_path,
#     num_episodes=10,
#     output_file="rollouts.pkl"
# ):
#     """Run policy and save observations (no rendering)."""

#     # Load config
#     cfg.merge_from_file('./output/config.yaml')

#     # Configure env (IMPORTANT: training=False)
#     cfg.ENV.WALKERS = [agent_name]
#     cfg.PPO.NUM_ENVS = 1
#     cfg.VECENV.TYPE = "DummyVecEnv"

#     # Load model
#     actor_critic, ob_rms, device = load_model(checkpoint_path)
#     agent = Agent(actor_critic)

#     # Create env (headless-safe)
#     env = make_vec_envs(
#         training=False,        # DO NOT change this
#         norm_rew=False,
#         save_video=False
#     )
#     set_ob_rms(env, ob_rms)

#     all_episodes = []  ### NEW

#     print(f"\nCollecting rollouts for agent: {agent_name}")
#     print("-" * 50)

#     for ep in range(num_episodes):
#         obs = env.reset()

#         episode_data = {
#             "observations": [],
#             "actions": [],
#             "rewards": []
#         }

#         done = False
#         ep_reward = 0.0

#         while not done:
#             # Save observation (DICT SAFE)
#             episode_data["observations"].append({
#                 k: v.detach().cpu().numpy() if torch.is_tensor(v) else v.copy()
#                 for k, v in obs.items()
#             })

#             with torch.no_grad():
#                 _, pi, _, _ = actor_critic(obs)
#                 action = pi.mean

#             episode_data["actions"].append(
#                 action.detach().cpu().numpy()[0]
#             )

#             obs, reward, done, info = env.step(action)

#             episode_data["rewards"].append(float(reward[0]))

#             if done[0]:
#                 break


#         all_episodes.append(episode_data)

#         print(
#             f"Episode {ep+1:2d}: "
#             f"Reward = {ep_reward:7.2f}, "
#             f"Length = {len(episode_data['rewards'])}"
#         )

#     env.close()

#     # Save to disk
#     rollout_data = {
#         "agent_name": agent_name,
#         "num_episodes": num_episodes,
#         "episodes": all_episodes
#     }

#     with open(output_file, "wb") as f:
#         pickle.dump(rollout_data, f)

#     print("-" * 50)
#     print(f"✓ Rollouts saved to: {output_file}")

#     return rollout_data


# if __name__ == "__main__":
#     collect_and_save_rollouts(
#         agent_name="floor-1409-10-0-01-11-09-15",
#         checkpoint_path="./output/Unimal-v0.pt",
#         num_episodes=20,
#         output_file="./rollouts.pkl"
#     )
import numpy as np
import torch
import pickle
from metamorph.config import cfg
from metamorph.algos.ppo.envs import make_vec_envs, set_ob_rms
from metamorph.algos.ppo.model import Agent

def load_model(checkpoint_path):
    """Load trained model from checkpoint"""
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    actor_critic = checkpoint[0]
    ob_rms = checkpoint[1]
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    actor_critic.to(device)
    actor_critic.eval()
    
    return actor_critic, ob_rms, device

def save_episode_trajectories(agent_name, checkpoint_path, num_episodes=10, output_file=None):
    """Save trajectories for offline rendering"""
    
    # Load config
    cfg.merge_from_file('./output/config.yaml')
    
    # Configure for testing
    cfg.ENV.WALKERS = [agent_name]
    cfg.PPO.NUM_ENVS = 1
    cfg.VECENV.TYPE = "DummyVecEnv"
    
    # Load model
    print("Loading model...")
    actor_critic, ob_rms, device = load_model(checkpoint_path)
    agent = Agent(actor_critic)
    
    # Create environment
    print("Creating environment...")
    env = make_vec_envs(training=False, norm_rew=False, save_video=False)
    set_ob_rms(env, ob_rms)
    
    # Prepare data structure
    episodes_data = {
        'agent_name': agent_name,
        'episodes': [],
        'ob_rms': ob_rms  # Save normalization stats too
    }
    
    print(f"\nCollecting trajectories for agent: {agent_name}")
    print("=" * 60)
    
    for ep in range(num_episodes):
        print(f"\nEpisode {ep + 1}/{num_episodes}")
        print("-" * 60)
        
        obs = env.reset()
        done = False
        episode_states = []
        episode_reward = 0
        episode_length = 0
        
        while not done:
            # Get current state from the base environment
            try:
                sim = env.envs[0].env.env.sim
                state = {
                    'qpos': sim.data.qpos.copy(),
                    'qvel': sim.data.qvel.copy(),
                }
                episode_states.append(state)
            except Exception as e:
                print(f"  Warning: Could not capture state at step {episode_length}: {e}")
                break
            
            # Get action
            with torch.no_grad():
                _, pi, _, _ = actor_critic(obs)
                action = pi.mean
            
            # Step environment
            obs, reward, done, info = env.step(action)
            
            episode_reward += float(reward[0])
            episode_length += 1
            
            # Progress update
            if episode_length % 100 == 0:
                print(f"  Collected {episode_length} states, reward so far: {episode_reward:.2f}")
            
            if done[0]:
                break
        
        episodes_data['episodes'].append({
            'states': episode_states,
            'reward': episode_reward,
            'length': episode_length
        })
        
        print(f"  ✓ Collected {len(episode_states)} states")
        print(f"  Reward: {episode_reward:.2f}, Length: {episode_length}")
    
    env.close()
    
    # Save to file
    if output_file is None:
        output_file = f'{agent_name}_trajectories.pkl'
    
    print(f"\nSaving trajectories to {output_file}...")
    with open(output_file, 'wb') as f:
        pickle.dump(episodes_data, f)
    
    file_size_mb = os.path.getsize(output_file) / (1024 * 1024)
    print(f"✓ Saved {num_episodes} episodes ({file_size_mb:.2f} MB)")
    
    # Print summary
    rewards = [ep['reward'] for ep in episodes_data['episodes']]
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"File: {output_file}")
    print(f"Total episodes: {num_episodes}")
    print(f"Mean Reward:  {np.mean(rewards):7.2f} ± {np.std(rewards):6.2f}")
    print(f"Min Reward:   {np.min(rewards):7.2f}")
    print(f"Max Reward:   {np.max(rewards):7.2f}")
    print("=" * 60)
    print("\nDownload this file and run the replay script locally to generate videos.")
    
    return output_file

if __name__ == "__main__":
    import os
    
    output = save_episode_trajectories(
        agent_name='floor-1409-0-14-01-12-42-39',
        checkpoint_path='./output/Unimal-v0.pt',
        num_episodes=10,
        output_file='trajectories2.pkl'
    )
    
    print(f"\n✓ Done! Download {output} to your local machine.")