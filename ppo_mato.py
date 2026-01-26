import os
import torch
import json
import numpy as np
from datetime import datetime
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback, BaseCallback
from stable_baselines3.common.env_util import make_vec_env
from carla_env.carla_env_PPO import CarEnv
import matplotlib.pyplot as plt

# Custom callback to monitor training and terminate early if necessary

class EarlyStoppingCallback(BaseCallback):
    def __init__(self, patience=200, min_evals=50, threshold=-200, verbose=1, plot_path="training_rewards.png"):
        super().__init__(verbose)
        self.patience = patience
        self.min_evals = min_evals
        self.threshold = threshold
        self.best_mean_reward = -np.inf
        self.no_improvement_steps = 0
        self.rewards = []
        self.moving_avg = []
        self.plot_path = plot_path

    def _on_step(self) -> bool:
        if "episode" in self.locals["infos"][0]:
            episode_rewards = [info["episode"]["r"] for info in self.locals["infos"] if "episode" in info]
            self.rewards.extend(episode_rewards)
            self._save_plot()
            
            # Vypočítať kĺzavý priemer
            if len(self.rewards) >= 100:
                self.moving_avg.append(np.mean(self.rewards[-100:]))
                
                if len(self.moving_avg) >= self.min_evals:
                    mean_reward = self.moving_avg[-1]
                    if self.verbose > 0:
                        print(f"Mean reward over last 100 episodes: {mean_reward:.2f}")

                    if mean_reward > self.best_mean_reward:
                        self.best_mean_reward = mean_reward
                        self.no_improvement_steps = 0
                    else:
                        self.no_improvement_steps += 1
                        
                    if self.no_improvement_steps >= self.patience:
                        if self.verbose > 0:
                            print(f"Stopping training because no improvement for {self.patience} evaluations")
                        return False

        return True

    def _save_plot(self) -> None:
        plt.figure(figsize=(12, 8))
        plt.subplot(2, 1, 1)
        plt.plot(self.rewards, alpha=0.3, color='blue', label='Raw rewards')
        plt.xlabel("Episodes")
        plt.ylabel("Reward")
        plt.title("Training Rewards")
        plt.legend()
        
        if len(self.moving_avg) > 0:
            plt.subplot(2, 1, 2)
            plt.plot(self.moving_avg, color='red', label='Moving average (100 episodes)')
            plt.xlabel("Evaluations")
            plt.ylabel("Average Reward")
            plt.title("Training Progress")
            plt.legend()
            
        plt.tight_layout()
        plt.savefig(self.plot_path)
        plt.close()

class PPOStatCallback(BaseCallback):
    def __init__(self, print_freq=1000, verbose=1):
        super().__init__(verbose)
        self.print_freq = print_freq
        self.num_calls = 0

    def _on_step(self) -> bool:
        self.num_calls += 1
        if self.num_calls % self.print_freq == 0 and self.verbose > 0:
            try:
                # Get PPO-specific statistics with defaults
                approx_kl = self.model.logger.name_to_value.get("train/approx_kl", 0.0)
                policy_loss = self.model.logger.name_to_value.get("train/policy_loss", 0.0)
                value_loss = self.model.logger.name_to_value.get("train/value_loss", 0.0)
                entropy = self.model.logger.name_to_value.get("train/entropy", 0.0)
                
                # Format the message with safe values
                message = f"\033[94m[PPOStatCallback] Current stats: "
                
                if approx_kl is not None:
                    message += f"approx_kl={approx_kl:.5f}, "
                else:
                    message += "approx_kl=N/A, "
                    
                if policy_loss is not None:
                    message += f"policy_loss={policy_loss:.5f}, "
                else:
                    message += "policy_loss=N/A, "
                    
                if value_loss is not None:
                    message += f"value_loss={value_loss:.5f}, "
                else:
                    message += "value_loss=N/A, "
                    
                if entropy is not None:
                    message += f"entropy={entropy:.5f}"
                else:
                    message += "entropy=N/A"
                
                message += "\033[0m"
                print(message)
                
            except Exception as e:
                print(f"\033[94m[PPOStatCallback] Could not access PPO stats: {str(e)}\033[0m")
        return True
    
    
class CustomCheckpointCallback(BaseCallback):
    def __init__(self, save_freq, save_path, name_prefix="ppo_model", verbose=1):
        super().__init__(verbose)
        self.save_freq = save_freq
        self.save_path = save_path
        self.name_prefix = name_prefix
        os.makedirs(self.save_path, exist_ok=True)

    def _on_step(self) -> bool:
        if self.n_calls % self.save_freq == 0:
            model_path = os.path.join(self.save_path, f"{self.name_prefix}_{self.num_timesteps}_steps")

            # Save model
            self.model.save(model_path)
            if self.verbose > 0:
                print(f"\033[92m[CustomCheckpointCallback] Saved model to {model_path}.zip\033[0m")

        return True


# Create the environment
# We'll assume CarEnv will be configured with MultiDiscrete action space
# Create the environment
env = make_vec_env(lambda: CarEnv(), n_envs=1)
# exit(0)

# Define model path 47 70 98
model_path = r"C:\Storage\8semester\SMAAD\zadanie\smad-zadanie-21\checkpoints\ppo_carla_580000_steps.zip"


# Define a checkpoint callback to save the model periodically
checkpoint_callback = CustomCheckpointCallback(
    save_freq=10000,
    save_path="./checkpoints/",
    name_prefix="ppo_carla",  # Changed from dqn to ppo
)

early_stopping_callback = EarlyStoppingCallback(patience=200, threshold=-200, verbose=1)

# Replace EpsilonPrinterCallback with our new PPO callback
ppo_stat_callback = PPOStatCallback(print_freq=1000, verbose=1)

# Check if a saved model exists
if os.path.exists(model_path):
    print(f"Loading model from {model_path}")
    model = PPO.load(model_path, env=env, device="cuda" if torch.cuda.is_available() else "cpu")
else:
    print("No saved model found. Creating a new model.")
    model = PPO(
        policy="CnnPolicy",
        env=env,
        learning_rate=0.00001,
        n_steps=2048,
        batch_size=128,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        clip_range_vf=None,
        normalize_advantage=True,
        max_grad_norm=0.5,
        use_sde=True,
        ent_coef=0.05,
        vf_coef=0.5,
        sde_sample_freq=8,
        target_kl=0.01,
        verbose=1,
        device="cuda" if torch.cuda.is_available() else "cpu",
    )

# Train the model for multiple epochs
epochs = 50
timesteps_per_epoch = 100000

for epoch in range(epochs):
    print(f"Starting epoch {epoch + 1}/{epochs}")

    model.learn(
        total_timesteps=timesteps_per_epoch,
        reset_num_timesteps=False,
        callback=[checkpoint_callback, ppo_stat_callback,early_stopping_callback],
    )

# Save the final model
model.save("dqn_carla_final")

# Close the environment
env.close()
