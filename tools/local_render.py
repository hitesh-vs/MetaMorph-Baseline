import os
os.environ['MUJOCO_GL'] = 'glfw'  # Use GLFW for local rendering with display
os.environ['CUDA_VISIBLE_DEVICES'] = ''  # Force CPU

import pickle
import numpy as np
import torch
import imageio
from metamorph.config import cfg
from metamorph.algos.ppo.envs import make_vec_envs, set_ob_rms

from tools.train_ppo import register_modular_envs, set_cfg_options

def replay_trajectories_to_video(trajectory_file, output_dir='videos', fps=30):
    """
    Load saved trajectories and render them as videos
    
    Args:
        trajectory_file: Path to the .pkl file with saved trajectories
        output_dir: Directory to save videos
        fps: Frames per second for output videos
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Load trajectories
    print(f"Loading trajectories from {trajectory_file}...")
    with open(trajectory_file, 'rb') as f:
        data = pickle.load(f)
    
    agent_name = data['agent_name']
    episodes = data['episodes']
    ob_rms = data.get('ob_rms', None)
    
    print(f"Loaded {len(episodes)} episodes for agent: {agent_name}")
    
    # Load config
    cfg.merge_from_file('./output/config.yaml')
    cfg.DEVICE = 'cpu'
    cfg.ENV.WALKERS = [agent_name]
    cfg.PPO.NUM_ENVS = 1
    cfg.VECENV.TYPE = "DummyVecEnv"
    
    set_cfg_options()
    register_modular_envs()

    # Create environment for rendering
    print("Creating environment...")
    env = make_vec_envs(training=False, norm_rew=False, save_video=False)
    
    if ob_rms is not None:
        set_ob_rms(env, ob_rms)
    
    print(f"\n{'=' * 60}")
    print(f"Rendering videos...")
    print(f"{'=' * 60}\n")
    
    # Render each episode
    for ep_idx, episode in enumerate(episodes):
        print(f"Episode {ep_idx + 1}/{len(episodes)}")
        print("-" * 60)
        
        states = episode['states']
        reward = episode['reward']
        length = episode['length']
        
        print(f"  States: {len(states)}, Reward: {reward:.2f}, Length: {length}")
        
        frames = []
        
        # Reset environment
        env.reset()
        
        # Get the actual mujoco sim object
        sim = env.envs[0].env.env.sim
        
        # Replay each state
        for state_idx, state in enumerate(states):
            # Set the state
            sim.data.qpos[:] = state['qpos']
            sim.data.qvel[:] = state['qvel']
            
            # Forward the simulation to update visualization
            sim.forward()
            
            # Render frame
            try:
                frame = env.render(mode='rgb_array')
                frame = frame[::-1]
                frames.append(frame)
            except Exception as e:
                print(f"  ✗ Rendering error at frame {state_idx}: {e}")
                break
            
            # Progress update
            if (state_idx + 1) % 100 == 0:
                print(f"  Rendered {state_idx + 1}/{len(states)} frames")
        
        # Save video
        if frames:
            video_filename = f"{agent_name}_ep{ep_idx:02d}_r{reward:.0f}.mp4"
            video_path = os.path.join(output_dir, video_filename)
            
            print(f"  Saving video: {video_filename}")
            print(f"  Frames: {len(frames)}, Shape: {frames[0].shape}")
            
            imageio.mimsave(video_path, frames, fps=fps, quality=8)
            print(f"  ✓ Saved: {video_path}\n")
        else:
            print(f"  ✗ No frames rendered for episode {ep_idx}\n")
    
    env.close()
    
    print(f"\n{'=' * 60}")
    print(f"✓ All videos saved to {output_dir}/")
    print(f"{'=' * 60}\n")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Replay saved trajectories as videos')
    parser.add_argument('trajectory_file', type=str, help='Path to .pkl trajectory file')
    parser.add_argument('--output-dir', type=str, default='videos', help='Output directory for videos')
    parser.add_argument('--fps', type=int, default=30, help='Frames per second')
    
    args = parser.parse_args()
    
    replay_trajectories_to_video(
        args.trajectory_file,
        output_dir=args.output_dir,
        fps=args.fps
    )