import os
import time
import logging
import numpy as np
from stable_baselines3 import PPO  # Changed from DQN to PPO
import gymnasium as gym
# Import the CarEnv class directly
from carla_env.carla_env_PPO import CarEnv  # Changed to use PPO environment

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def test_model():
    """
    Test a trained model in the CARLA environment.
    
    Args:
        model_path: Path to the saved model
        num_episodes: Number of episodes to test
        render: Whether to render the environment (show preview)
    """
    model_path = r"C:\Storage\8semester\SMAAD\zadanie\smad-zadanie-21\checkpoints\ppo_carla_200000_steps.zip"  # Updated path to PPO model
    num_episodes = 50
    render = True
    logger.info(f"Testing model: {model_path}")
    
    # Create environment directly instead of using gym.make
    # You can use the same env for testing - no need to create a new one
    env = CarEnv()

    # Load the trained PPO model
    logger.info("Loading PPO model...")
    model = PPO.load(model_path)  # Changed from DQN to PPO
    logger.info("Model loaded successfully")
    
    # Testing loop
    total_rewards = []
    success_count = 0
    collision_count = 0
    
    for episode in range(num_episodes):
        logger.info(f"Starting test episode {episode+1}/{num_episodes}")
        
        # Reset environment
        obs = env.reset()
        if isinstance(obs, tuple):  # Handle both old and new gym API
            obs = obs[0]
            
        done = False
        truncated = False
        cumulative_reward = 0
        step_count = 0
        
        # Episode loop
        while not (done or truncated):
            # Get model's action
            action, _ = model.predict(obs, deterministic=True)
            
            # Take step in environment
            step_result = env.step(action)
            
            # Handle different return formats (gym vs gymnasium)
            if len(step_result) == 4:  # Old gym format
                obs, reward, done, info = step_result
                truncated = False
            else:  # New gymnasium format
                obs, reward, done, truncated, info = step_result
            
            cumulative_reward += reward
            step_count += 1
            
            # Optional: Sleep to slow down visualization
            if render and step_count % 10 == 0:
                logger.info(f"Step: {step_count}, Reward: {reward:.2f}, Cumulative: {cumulative_reward:.2f}")
                time.sleep(0.05)
                
        # Episode ended - record results
        total_rewards.append(cumulative_reward)
        
        # Check if episode ended due to collision
        if hasattr(env, 'collision_hist') and len(env.collision_hist) > 0:
            collision_count += 1
            logger.info(f"Episode {episode+1} ended with a collision")
        
        # Check if episode reached destination (simplified success condition)
        elif cumulative_reward > 0 and not truncated:
            success_count += 1
            logger.info(f"Episode {episode+1} successful!")
        
        logger.info(f"Episode {episode+1} finished with reward: {cumulative_reward:.2f} in {step_count} steps")
    
    # Final statistics
    logger.info("=== Testing Summary ===")
    logger.info(f"Average reward: {np.mean(total_rewards):.2f}")
    logger.info(f"Success rate: {success_count/num_episodes*100:.1f}%")
    logger.info(f"Collision rate: {collision_count/num_episodes*100:.1f}%")
    
    # Close environment
    if hasattr(env, 'cleanup'):
        env.cleanup()
    else:
        env.close()

if __name__ == "__main__":
    # Test the model
    test_model()