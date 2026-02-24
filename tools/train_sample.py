"""
Test script to verify ModularEnv modifications
Run this before training to ensure observations are correct
"""

import numpy as np
import sys
import os

# Add paths as needed
# sys.path.insert(0, '/path/to/your/metamorph')

from metamorph.config import cfg

# Mock config if not loading from file
class MockConfig:
    MODEL = type('obj', (object,), {
        'MAX_LIMBS': 13,  # Updated for Unitree G1 (13 bodies)
        'MAX_JOINTS': 13,  # Match MAX_LIMBS for consistent padding
        'CONTEXT_OBS_TYPES': ['body_pos', 'body_ipos', 'body_iquat', 'geom_quat', 'body_mass', 'body_shape'],
        'PROPRIOCEPTIVE_OBS_TYPES': ['body_xpos', 'body_xvelp', 'body_xvelr', 'body_xquat', 'qpos', 'qvel'],
        'TRANSFORMER': type('obj', (object,), {'USE_SWAT_RE': False})(),
        'MLP': type('obj', (object,), {'CONSISTENT_PADDING': True})(),
    })()
    ENV = type('obj', (object,), {
        'WALKER_DIR': 'unitree_g1',
    })()

# If cfg is not properly initialized, use mock
try:
    _ = cfg.MODEL.MAX_LIMBS
    # If MAX_LIMBS is too small for the robot, override it
    if cfg.MODEL.MAX_LIMBS < 13:
        print(f"WARNING: cfg.MODEL.MAX_LIMBS={cfg.MODEL.MAX_LIMBS} is too small for G1 robot (13 limbs)")
        print(f"         Overriding to MAX_LIMBS=13, MAX_JOINTS=13")
        cfg.MODEL.MAX_LIMBS = 13
        cfg.MODEL.MAX_JOINTS = 13
except:
    cfg = MockConfig()


def test_modular_env(xml_path):
    """Test the modified ModularEnv"""
    
    print("=" * 80)
    print("TESTING MODULAR ENV MODIFICATIONS")
    print("=" * 80)
    
    # Import after config is set
    from modular.g1_12dof import make_env
    
    print(f"\n1. Creating environment from: {xml_path}")
    env = make_env(xml_path)
    
    print(f"\n2. Environment metadata:")
    print(f"   - num_limbs: {env.unwrapped.metadata['num_limbs']}")
    print(f"   - num_joints: {env.unwrapped.metadata['num_joints']}")
    print(f"   - MAX_LIMBS (padded): {cfg.MODEL.MAX_LIMBS}")
    print(f"   - MAX_JOINTS (padded): {cfg.MODEL.MAX_JOINTS}")
    
    print(f"\n3. Resetting environment...")
    obs = env.reset()
    
    print(f"\n4. Observation structure:")
    print(f"   Type: {type(obs)}")
    if isinstance(obs, dict):
        print(f"   Keys: {list(obs.keys())}")
        for key, value in obs.items():
            if isinstance(value, np.ndarray):
                print(f"   - {key}: shape {value.shape}, dtype {value.dtype}")
            else:
                print(f"   - {key}: {type(value)}")
    else:
        print(f"   ERROR: Expected dict, got {type(obs)}")
        return False
    
    print(f"\n5. Checking critical components:")
    
    # Check edges
    edges = obs['edges']
    edges_nonzero = np.count_nonzero(edges)
    print(f"   - Edges: {edges_nonzero}/{len(edges)} non-zero values")
    if edges_nonzero == 0:
        print(f"     ⚠️  WARNING: All edges are zero! Graph structure is missing!")
    else:
        print(f"     ✓ Graph structure present")
        print(f"     First 10 edges: {edges[:10]}")
    
    # Check context
    context = obs['context']
    context_nonzero = np.count_nonzero(context)
    print(f"   - Context: {context_nonzero}/{len(context)} non-zero values")
    if context_nonzero == 0:
        print(f"     ⚠️  WARNING: All context is zero! Morphology encoding is missing!")
    else:
        print(f"     ✓ Context encoding present")
        print(f"     Context stats: min={context.min():.3f}, max={context.max():.3f}, mean={context.mean():.3f}")
    
    # Check if context is just duplicated proprioceptive
    prop = obs['proprioceptive']
    if np.allclose(context[:len(prop)], prop):
        print(f"     ⚠️  WARNING: Context appears to be duplicate of proprioceptive!")
    else:
        print(f"     ✓ Context is distinct from proprioceptive")
    
    # Check SWAT_RE
    swat_re = obs['SWAT_RE']
    print(f"   - SWAT_RE: shape {swat_re.shape}, {np.count_nonzero(swat_re)} non-zero")
    
    # Check padding masks
    obs_mask = obs['obs_padding_mask']
    act_mask = obs['act_padding_mask']
    print(f"   - Obs padding mask: {np.sum(obs_mask)}/{len(obs_mask)} masked")
    print(f"   - Act padding mask: {np.sum(act_mask)}/{len(act_mask)} masked")
    
    print(f"\n6. Testing action space:")
    print(f"   - Action space: {env.action_space}")
    print(f"   - Action shape: {env.action_space.shape}")
    
    print(f"\n7. Testing step function...")
    action = env.action_space.sample()
    obs_next, reward, done, info = env.step(action)
    
    print(f"   - Step completed successfully")
    print(f"   - Reward: {reward:.3f}")
    print(f"   - Done: {done}")
    print(f"   - Info keys: {list(info.keys())}")
    
    print(f"\n8. Testing multiple steps...")
    total_reward = 0
    for i in range(10):
        action = env.action_space.sample()
        obs, reward, done, info = env.step(action)
        total_reward += reward
        if done:
            print(f"   - Episode ended at step {i+1}")
            obs = env.reset()
            break
    
    print(f"   - Total reward over 10 steps: {total_reward:.3f}")
    
    print("\n" + "=" * 80)
    print("SUMMARY:")
    
    success = True
    if edges_nonzero == 0:
        print("❌ FAILED: No graph structure (edges are all zero)")
        success = False
    else:
        print("✓ PASSED: Graph structure present")
    
    if context_nonzero == 0:
        print("❌ FAILED: No morphology context (context is all zero)")
        success = False
    elif np.allclose(context[:len(prop)], prop):
        print("❌ FAILED: Context is duplicate of proprioceptive")
        success = False
    else:
        print("✓ PASSED: Morphology context present")
    
    print("\n" + "=" * 80)
    
    env.close()
    return success


def test_reward_scale(xml_path):
    """Test that reward scale is reasonable"""
    print("\n" + "=" * 80)
    print("TESTING REWARD SCALE WITH RANDOM POLICY")
    print("=" * 80)
    
    from modular.g1_12dof import make_env
    
    env = make_env(xml_path)
    
    rewards = []
    episode_lengths = []
    
    print("\nRunning 5 episodes with random actions...")
    for ep in range(5):
        obs = env.reset()
        ep_reward = 0
        ep_length = 0
        
        for step in range(1000):
            action = env.action_space.sample()
            obs, reward, done, info = env.step(action)
            ep_reward += reward
            ep_length += 1
            
            if done:
                break
        
        rewards.append(ep_reward)
        episode_lengths.append(ep_length)
        print(f"  Episode {ep+1}: reward={ep_reward:.1f}, length={ep_length}")
    
    avg_reward = np.mean(rewards)
    avg_length = np.mean(episode_lengths)
    
    print(f"\nAverage reward: {avg_reward:.1f} ± {np.std(rewards):.1f}")
    print(f"Average length: {avg_length:.1f} ± {np.std(episode_lengths):.1f}")
    
    # Check for catastrophic rewards
    if avg_reward < -10000:
        print(f"❌ WARNING: Reward scale is catastrophically negative!")
        print(f"   This suggests a major issue with the reward function or environment.")
        return False
    elif avg_reward < -100:
        print(f"⚠️  WARNING: Rewards are very negative ({avg_reward:.1f})")
        print(f"   Check reward function coefficients (ctrl_cost_weight, etc.)")
    else:
        print(f"✓ Reward scale looks reasonable")
    
    if avg_length < 10:
        print(f"⚠️  WARNING: Episodes are very short ({avg_length:.1f} steps)")
        print(f"   Robot may be falling immediately. Check initial state and termination conditions.")
    
    env.close()
    return True

if __name__ == "__main__":
    # Update this path to your robot XML
    xml_path = "/home/sviswasam/dr/ModuMorph/modular/unitree_g1/xml/g1_12dof.xml"
    
    if len(sys.argv) > 1:
        xml_path = sys.argv[1]
    
    if not os.path.exists(xml_path):
        print(f"Error: XML file not found at {xml_path}")
        print(f"Usage: python test_modular_env.py /path/to/robot.xml")
        sys.exit(1)
    
    # Run tests
    obs_test_passed = test_modular_env(xml_path)
    reward_test_passed = test_reward_scale(xml_path)
    
    print("\n" + "=" * 80)
    if obs_test_passed and reward_test_passed:
        print("✓ ALL TESTS PASSED - Ready for training!")
    else:
        print("❌ TESTS FAILED - Fix issues before training")
    print("=" * 80)