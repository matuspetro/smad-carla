import os
import torch
import numpy as np
from stable_baselines3 import DQN
from stable_baselines3.common.callbacks import CheckpointCallback, BaseCallback
from stable_baselines3.common.env_util import make_vec_env
from carla_env.carla_env_DQN import CarEnv
import matplotlib.pyplot as plt

class EarlyStoppingCallback(BaseCallback):
    def __init__(self, patience=400, min_evals=50, threshold=-200, verbose=1, plot_path="training_rewards.png"):
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

class EpsilonPrinterCallback(BaseCallback):
    def __init__(self, print_freq=1000, verbose=1):
        super().__init__(verbose)
        self.print_freq = print_freq
        self.num_calls = 0

    def _on_step(self) -> bool:
        self.num_calls += 1
        if self.num_calls % self.print_freq == 0:
            current_epsilon = self.model.exploration_rate
            if self.verbose > 0:
                print(f"\033[94m[EpsilonPrinterCallback] Current epsilon: {current_epsilon:.4f}\033[0m")
        return True
    
class CustomCheckpointCallback(BaseCallback):
    def __init__(self, save_freq, save_path, name_prefix="dqn_model", verbose=1):
        super().__init__(verbose)
        self.save_freq = save_freq
        self.save_path = save_path
        self.name_prefix = name_prefix
        os.makedirs(self.save_path, exist_ok=True)

    def _on_step(self) -> bool:
        if self.n_calls % self.save_freq == 0:
            model_path = os.path.join(self.save_path, f"{self.name_prefix}_{self.num_timesteps}_steps")
            replay_buffer_path = model_path + "_replay_buffer.pkl"

            # Save model
            self.model.save(model_path)
            if self.verbose > 0:
                print(f"\033[92m[CustomCheckpointCallback] Saved model to {model_path}.zip\033[0m")

            # Save replay buffer
            # if hasattr(self.model, "save_replay_buffer"):
            #     try:
            #         self.model.save_replay_buffer(replay_buffer_path)
            #         if self.verbose > 0:
            #             print(f"\033[96m[CustomCheckpointCallback] Saved replay buffer to {replay_buffer_path}\033[0m")
            #     except Exception as e:
            #         print(f"\033[91m[CustomCheckpointCallback] Failed to save replay buffer: {e}\033[0m")

        return True


# Create the environment
env = make_vec_env(lambda: CarEnv(), n_envs=1)
# exit(0)

# Define model path 
model_path = r"C:\Storage\8semester\SMAAD\zadanie\smad-zadanie-21\checkpoints\dqn_carla_670000_steps.zip"


# model = DQN(
#     policy="CnnPolicy",
#     env=env,
#     learning_rate=0.00001,
#     buffer_size=20000,
#     learning_starts=1000,
#     batch_size=32,
#     tau=1.0,
#     gamma=0.995,
#     train_freq=4,
#     target_update_interval=500,
#     exploration_fraction=0.4,
#     exploration_initial_eps=0.1,
#     exploration_final_eps=0.00,
#     verbose=1,
#     device="cuda" if torch.cuda.is_available() else "cpu",
#     tensorboard_log="./tensorboard_logs/",
# )

if os.path.exists(model_path):
    print(f"Loading model from: {model_path}")
    old_model = DQN.load(model_path, env=env, device="cuda" if torch.cuda.is_available() else "cpu")
    
    # Load parameters
    # model.policy.load_state_dict(old_model.policy.state_dict())
    # model.load_replay_buffer("./checkpoints/dqn_carla_160000_steps_replay_buffer.pkl")
    model = old_model
else:
    print("No model found, starting training from scratch.")

# Define a checkpoint callback to save the model periodically
checkpoint_callback = CustomCheckpointCallback(
    save_freq=10000,
    save_path="./checkpoints/",
    name_prefix="dqn_carla",
)

early_stopping_callback = EarlyStoppingCallback(patience=5, threshold=-200, verbose=1)

epsilon_printer_callback = EpsilonPrinterCallback(print_freq=1000, verbose=1)

# Train the model for multiple epochs
epochs = 50
timesteps_per_epoch = 100000

for epoch in range(epochs):
    print(f"Starting epoch {epoch + 1}/{epochs}")

    model.learn(
        total_timesteps=timesteps_per_epoch,
        reset_num_timesteps=False,
        callback=[checkpoint_callback, early_stopping_callback, epsilon_printer_callback],
    )

# Save the final model
model.save("dqn_carla_final")

# Close the environment
env.close()
