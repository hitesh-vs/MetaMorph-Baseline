import numpy as np
import torch
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

def test_policy(agent_name, checkpoint_path, num_episodes=10):
    """Test a policy without video"""
    
    # Load config
    cfg.merge_from_file('./output/config.yaml')
    
    # Configure for testing
    cfg.ENV.WALKERS = [agent_name]
    cfg.PPO.NUM_ENVS = 1
    cfg.VECENV.TYPE = "DummyVecEnv"
    
    # Load model
    actor_critic, ob_rms, device = load_model(checkpoint_path)
    agent = Agent(actor_critic)
    
    # Create environment (no video)
    env = make_vec_envs(training=False, norm_rew=False, save_video=False)
    set_ob_rms(env, ob_rms)
    
    # Run test episodes
    episode_rewards = []
    episode_lengths = []
    
    print(f"\nTesting agent: {agent_name}")
    print("-" * 50)
    
    for ep in range(num_episodes):
        obs = env.reset()
        done = False
        episode_reward = 0
        episode_length = 0
        
        while not done:
            # Get action (deterministic - use mean)
            with torch.no_grad():
                _, pi, _, _ = actor_critic(obs)
                action = pi.mean
            
            obs, reward, done, info = env.step(action)
            
            # Convert to float - FIX HERE
            episode_reward += float(reward[0])
            episode_length += 1
            
            if done[0]:
                break
        
        episode_rewards.append(episode_reward)
        episode_lengths.append(episode_length)
        
        # Now episode_reward is a regular Python float
        print(f"Episode {ep+1:2d}: Reward = {episode_reward:7.2f}, Length = {episode_length:4d}")
    
    env.close()
    
    # Print statistics
    print("-" * 50)
    print(f"Mean Reward:  {np.mean(episode_rewards):7.2f} ± {np.std(episode_rewards):6.2f}")
    print(f"Mean Length:  {np.mean(episode_lengths):7.2f} ± {np.std(episode_lengths):6.2f}")
    print(f"Min Reward:   {np.min(episode_rewards):7.2f}")
    print(f"Max Reward:   {np.max(episode_rewards):7.2f}")
    print("=" * 50)
    
    return {
        'agent': agent_name,
        'mean_reward': np.mean(episode_rewards),
        'std_reward': np.std(episode_rewards),
        'episode_rewards': episode_rewards,
        'episode_lengths': episode_lengths
    }

if __name__ == "__main__":
    # Test a single agent
    results = test_policy(
        agent_name='floor-1409-0-14-01-12-42-39',
        checkpoint_path='./output/Unimal-v0.pt',
        num_episodes=20
    )