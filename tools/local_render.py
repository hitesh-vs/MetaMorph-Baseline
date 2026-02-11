import os
import pickle
import numpy as np
import torch
import imageio
import argparse

# Set environment variables before imports that might initialize MuJoCo
os.environ['MUJOCO_GL'] = 'glfw'  # Use GLFW for local rendering with display
os.environ['CUDA_VISIBLE_DEVICES'] = ''  # Force CPU

from metamorph.config import cfg
from metamorph.algos.ppo.envs import make_vec_envs, set_ob_rms
from tools.train_ppo import register_modular_envs, set_cfg_options

def replay_trajectories_to_video(trajectory_file, output_dir='videos', fps=30):
    """
    Load saved trajectories and render them as videos
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
    
    # Finalize config for rendering
    cfg.ENV.WALKERS = [agent_name]
    cfg.PPO.NUM_ENVS = 1
    cfg.VECENV.TYPE = "DummyVecEnv"
    cfg.DEVICE = 'cpu'
    
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
        # Note: structure might vary slightly depending on your specific gym wrapper stack
        sim = env.envs[0].env.env.sim
        
        # Replay each state
        for state_idx, state in enumerate(states):
            # Set the state
            sim.data.qpos[:] = state['qpos']
            sim.data.qvel[:] = state['qvel']
            
            # Forward the simulation to update positions/visuals
            sim.forward()
            
            # Render frame
            try:
                frame = env.render(mode='rgb_array')
                # Flip frame if it comes out upside down (common in MuJoCo/OpenGL)
                frame = frame[::-1]
                frames.append(frame)
            except Exception as e:
                print(f"  ✗ Rendering error at frame {state_idx}: {e}")
                break
            
            if (state_idx + 1) % 100 == 0:
                print(f"  Rendered {state_idx + 1}/{len(states)} frames")
        
        # Save video
        if frames:
            video_filename = f"{agent_name}_ep{ep_idx:02d}_r{reward:.0f}.mp4"
            video_path = os.path.join(output_dir, video_filename)
            
            print(f"  Saving video: {video_filename}")
            imageio.mimsave(video_path, frames, fps=fps, quality=8)
            print(f"  ✓ Saved: {video_path}\n")
        else:
            print(f"  ✗ No frames rendered for episode {ep_idx}\n")
    
    env.close()
    print(f"\n{'=' * 60}")
    print(f"✓ All videos saved to {output_dir}/")
    print(f"{'=' * 60}\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Replay saved trajectories as videos')
    parser.add_argument('trajectory_file', type=str, help='Path to .pkl trajectory file')
    parser.add_argument('--walker-dir', type=str, help='Path to the directory containing walker XML files')
    parser.add_argument('--config', type=str, default='./output/config.yaml', help='Path to config.yaml')
    parser.add_argument('--output-dir', type=str, default='videos', help='Output directory for videos')
    parser.add_argument('--fps', type=int, default=30, help='Frames per second')
    
    args = parser.parse_args()
    
    # 1. Load base configuration
    if os.path.exists(args.config):
        cfg.merge_from_file(args.config)
    else:
        print(f"Warning: Config file {args.config} not found. Using defaults.")

    # 2. Apply CLI Overrides
    if args.walker_dir:
        print(f"Overriding WALKER_DIR to: {args.walker_dir}")
        cfg.ENV.WALKER_DIR = args.walker_dir
    
    # 3. Execute
    replay_trajectories_to_video(
        args.trajectory_file,
        output_dir=args.output_dir,
        fps=args.fps
    )