import os
import sys

# Project root = folder where main.py lives
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))

# Add the *inner* simglucose repo root to sys.path
SUBMODULE_ROOT = os.path.join(PROJECT_ROOT, "simglucose")
sys.path.insert(0, SUBMODULE_ROOT)

import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import SubprocVecEnv
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback, CallbackList
import os
import torch
import numpy as np

# Import the environment (paper_reward is now the default inside CustomT1DEnv)
from custom_env import CustomT1DEnv, paper_reward

def train():
    patient_id = 'adolescent#003'
    
    # Setup directories
    log_dir = "./train/ppo_paper_tensorboard/"
    checkpoint_dir = "./train/checkpoints_paper/"
    eval_log_dir = "./train/eval_logs_paper/"
    results_dir = "./train/results/"
    
    for directory in [log_dir, checkpoint_dir, eval_log_dir]:
        os.makedirs(directory, exist_ok=True)

    # Environment Configuration
    env_kwargs = {
        'patient_name': patient_id,
        'reward_fun': paper_reward,  # Explicitly use the paper's reward
        'seed': 42
    }

    # Parallelization
    n_envs = 4
    
    print(f"[INFO] Creating {n_envs} parallel environments...")
    train_env = make_vec_env(
        CustomT1DEnv,
        n_envs=n_envs,
        env_kwargs=env_kwargs,
        vec_env_cls=SubprocVecEnv
    )

    # Evaluation Environment
    print("[INFO] Creating evaluation environment...")
    eval_env_kwargs = env_kwargs.copy()
    eval_env_kwargs['seed'] = 42
    eval_env = make_vec_env(
        CustomT1DEnv,
        n_envs=1,
        env_kwargs=eval_env_kwargs
    )

    # Callbacks
    checkpoint_callback = CheckpointCallback(
        save_freq=100_000 // n_envs,
        save_path=checkpoint_dir,
        name_prefix=f"ppo_{patient_id}_paper",
        save_replay_buffer=False,
        save_vecnormalize=False
    )

    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=checkpoint_dir,
        log_path=eval_log_dir,
        eval_freq=50_000 // n_envs,
        n_eval_episodes=10,
        deterministic=True,
        render=False,
        verbose=1
    )

    callback = CallbackList([checkpoint_callback, eval_callback])

    # Initialize PPO Model
    print(f"\n{'='*70}")
    print(f"[STARTING PAPER REWARD TRAINING] Patient: {patient_id}")
    print(f"Goal: Minimize Risk Index (Penalty: -15 for Hypo)")
    print(f"{'='*70}\n")
    
    model = PPO(
        "MlpPolicy",
        train_env,
        verbose=1,
        learning_rate=3e-4,
        n_steps=2048,
        batch_size=256,
        n_epochs=10,
        gamma=0.995,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.005,
        vf_coef=0.5,
        max_grad_norm=0.5,
        policy_kwargs=dict(
            net_arch=dict(pi=[128, 128, 64], vf=[128, 128, 64]),
            activation_fn=torch.nn.ReLU
        ),
        tensorboard_log=log_dir,
        device="auto"
    )

    # Training Loop
    total_timesteps = 3_000_000
    
    try:
        model.learn(
            total_timesteps=total_timesteps,
            callback=callback,
            progress_bar=True
        )
    except KeyboardInterrupt:
        print("\n[WARNING] Training interrupted manually.")

    # Save Final Model
    final_model_name = f"{results_dir}/ppo_{patient_id}_paper_final"
    model.save(final_model_name)
    print(f"\n[SUCCESS] Final model saved: {final_model_name}.zip")
    
    # Cleanup: Delete all checkpoint files from this run
    print("\n[CLEANUP] Removing checkpoint files...")
    checkpoint_files = [f for f in os.listdir(checkpoint_dir) if f.startswith(f"ppo_{patient_id}_paper_")]
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