import os
import sys

# Project root = folder where main.py lives
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))

# Add the *inner* simglucose repo root to sys.path
SUBMODULE_ROOT = os.path.join(PROJECT_ROOT, "simglucose")
sys.path.insert(0, SUBMODULE_ROOT)

import gymnasium as gym
from stable_baselines3 import SAC
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback, CallbackList
import os
import torch
import numpy as np

# Import the environment
from custom_env import CustomT1DEnv

# Simple, direct reward function
def simple_error_reward(bg_hist, **kwargs):
    """
    Simple reward based on distance from target glucose (110 mg/dL).
    Clear learning signal: lower error = higher reward.
    """
    g_next = bg_hist[-1]
    target = 110.0
    
    # Calculate absolute error
    error = abs(g_next - target)
    
    # Normalize to roughly [-1, 0] range
    # Error of 0 → reward = 0 (best)
    # Error of 100 → reward = -1
    # Error of 200+ → reward = -2 or worse
    reward = -error / 100.0
    
    # Extra penalties for dangerous zones
    if g_next < 54:  # Severe hypo
        reward -= 2.0
    elif g_next < 70:  # Mild hypo
        reward -= 0.5
    elif g_next > 250:  # Severe hyper
        reward -= 1.0
    elif g_next > 180:  # Mild hyper
        reward -= 0.3
    
    return reward

def train():
    patient_id = 'adolescent#003'
    
    # Setup directories
    log_dir = "./train/sac_tensorboard/"
    checkpoint_dir = "./train/checkpoints_sac/"
    eval_log_dir = "./train/eval_logs_sac/"
    results_dir = "./train/results/"
    
    for directory in [log_dir, checkpoint_dir, eval_log_dir, results_dir]:
        os.makedirs(directory, exist_ok=True)

    # Environment Configuration
    env_kwargs = {
        'patient_name': patient_id,
        'reward_fun': simple_error_reward,  # Simple error-based reward
        'seed': 42,
        'max_episode_days': 1,
        'allow_early_termination': False  # Full episodes
    }

    # Single environment for SAC (better with replay buffer than parallel envs)
    print("[INFO] Creating training environment...")
    train_env = make_vec_env(
        CustomT1DEnv,
        n_envs=1,
        env_kwargs=env_kwargs
    )

    # Evaluation Environment
    print("[INFO] Creating evaluation environment...")
    eval_env_kwargs = env_kwargs.copy()
    eval_env_kwargs['seed'] = 123
    eval_env = make_vec_env(
        CustomT1DEnv,
        n_envs=1,
        env_kwargs=eval_env_kwargs
    )

    # Callbacks
    checkpoint_callback = CheckpointCallback(
        save_freq=50_000,
        save_path=checkpoint_dir,
        name_prefix=f"sac_{patient_id}",
        save_replay_buffer=True,  # SAC benefit: save replay buffer
        save_vecnormalize=False
    )

    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=checkpoint_dir,
        log_path=eval_log_dir,
        eval_freq=25_000,
        n_eval_episodes=10,
        deterministic=True,
        render=False,
        verbose=1
    )

    callback = CallbackList([checkpoint_callback, eval_callback])

    # Initialize SAC Model
    print(f"\n{'='*70}")
    print(f"[SAC TRAINING] Patient: {patient_id}")
    print(f"Algorithm: SAC (Soft Actor-Critic)")
    print(f"Reward: Simple error-based (-abs(glucose - 110)/100)")
    print(f"Key: Off-policy learning, automatic entropy tuning")
    print(f"{'='*70}\n")
    
    model = SAC(
        "MlpPolicy",
        train_env,
        verbose=1,
        learning_rate=3e-4,
        buffer_size=100_000,  # Replay buffer (keeps past experience)
        learning_starts=1000,  # Start learning after 1k steps
        batch_size=256,
        tau=0.005,  # Soft update coefficient
        gamma=0.99,
        train_freq=1,  # Update after every step
        gradient_steps=1,
        ent_coef='auto',  # Automatic entropy tuning (key SAC feature!)
        target_update_interval=1,
        policy_kwargs=dict(
            net_arch=[256, 256],  # Simpler than PPO's 3 layers
            activation_fn=torch.nn.ReLU
        ),
        tensorboard_log=log_dir,
        device="auto"
    )

    # Training Loop
    total_timesteps = 500_000  # Start with 500k, can increase if working
    
    try:
        model.learn(
            total_timesteps=total_timesteps,
            callback=callback,
            progress_bar=True
        )
    except KeyboardInterrupt:
        print("\n[WARNING] Training interrupted manually.")

    # Save Final Model
    os.makedirs(results_dir, exist_ok=True)
    final_model_name = f"{results_dir}/sac_{patient_id}_final"
    model.save(final_model_name)
    print(f"\n[SUCCESS] Final model saved: {final_model_name}.zip")
    
    # Cleanup: Delete intermediate checkpoints
    print("\n[CLEANUP] Removing checkpoint files...")
    checkpoint_files = [f for f in os.listdir(checkpoint_dir) if f.startswith(f"sac_{patient_id}_")]
    for checkpoint_file in checkpoint_files:
        checkpoint_path = os.path.join(checkpoint_dir, checkpoint_file)
        try:
            os.remove(checkpoint_path)
        except Exception as e:
            print(f"  ✗ Failed to delete {checkpoint_file}: {e}")
    
    if checkpoint_files:
        print(f"[CLEANUP] Removed {len(checkpoint_files)} checkpoint file(s)")
    
    train_env.close()
    eval_env.close()

if __name__ == "__main__":
    train()
