"""
Training Setup Module for PPO Blood Glucose Control.

This module provides utility functions for configuring the training
environment and callbacks used by the PPO training script. It handles:
- Directory creation for logs and checkpoints
- Environment configuration and instantiation
- Callback setup for evaluation and model saving

Separating setup from training logic improves code organisation
and makes the training script cleaner.
"""

import os

from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import SubprocVecEnv
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback, CallbackList

from custom_env import CustomT1DEnv
from reward_functions import paper_reward


# =============================================================================
# CONFIGURATION SETUP
# =============================================================================

def setup_train_config(episode_days, total_steps, n_envs, patient_id, log_dir, checkpoint_dir,
                       eval_log_dir, results_dir):
    """
    Configure training parameters and create output directories.
    
    Sets up the environment configuration dictionary and ensures all
    required output directories exist. Also prints a summary of the
    training configuration for user reference.
    
    Args:
        episode_days: Length of each training episode in days
        total_steps: Total timesteps for training
        n_envs: Number of parallel environments
        patient_id: Virtual patient identifier
        log_dir: TensorBoard log directory
        checkpoint_dir: Model checkpoint directory
        eval_log_dir: Evaluation log directory
        results_dir: Final model output directory
        
    Returns:
        dict: Environment keyword arguments for CustomT1DEnv
    """
    # Create all output directories if they don't exist
    for directory in [log_dir, checkpoint_dir, eval_log_dir, results_dir]:
        os.makedirs(directory, exist_ok=True)

    # Environment configuration for CustomT1DEnv
    env_kwargs = {
        'patient_name': patient_id,
        'reward_fun': paper_reward,  # Use paper's reward function
        'seed': 42,                  # Fixed seed for reproducibility
        'episode_days': episode_days
    }

    # Query dynamic values from a temporary environment instance
    # This ensures we get the actual sample time and max steps
    temp_env = CustomT1DEnv(**env_kwargs)
    sample_time = temp_env.sample_time
    max_episode_steps = temp_env.max_episode_steps
    temp_env.close()
    
    # Calculate approximate total episodes
    episodes_total = total_steps // max_episode_steps

    # Print training configuration summary
    print(f"\n{'='*70}")
    print(f"  PPO TRAINING - OPTIMIZED CONFIG")
    print(f"{'='*70}")
    print(f"  Patient: {patient_id}")
    print(f"  Parallel Envs: {n_envs} (SubprocVecEnv)")
    print(f"  Total Steps: {total_steps:,}")
    print(f"  Episode Days: {episode_days}")
    print(f"  Sample Time: {sample_time} min")
    print(f"  Steps/Episode: {max_episode_steps}")
    print(f"  Total Episodes: ~{episodes_total:,}")
    print(f"{'='*70}\n")

    return env_kwargs


# =============================================================================
# ENVIRONMENT AND CALLBACK SETUP
# =============================================================================

def setup_train_env(n_envs, env_kwargs, patient_id, checkpoint_dir, eval_log_dir):
    """
    Create vectorised environments and training callbacks.
    
    Sets up:
    1. Parallel training environments using SubprocVecEnv for speed
    2. Separate evaluation environment with different seed
    3. CheckpointCallback for periodic model saving
    4. EvalCallback for best model selection based on evaluation performance
    
    Args:
        n_envs: Number of parallel training environments
        env_kwargs: Environment configuration dictionary
        patient_id: Patient identifier for checkpoint naming
        checkpoint_dir: Directory for saving checkpoints
        eval_log_dir: Directory for evaluation logs
        
    Returns:
        tuple: (train_env, eval_env, callback) - environments and combined callback
    """
    # =========================================================================
    # TRAINING ENVIRONMENT
    # =========================================================================
    
    # Create parallel environments for faster data collection
    # SubprocVecEnv runs each environment in a separate process
    print(f"[INFO] Creating {n_envs} parallel environments...")
    train_env = make_vec_env(
        CustomT1DEnv,
        n_envs=n_envs,
        env_kwargs=env_kwargs,
        vec_env_cls=SubprocVecEnv
    )

    # =========================================================================
    # EVALUATION ENVIRONMENT
    # =========================================================================
    
    # Separate environment for periodic evaluation with different seed
    # This tests generalisation to unseen random scenarios
    print("[INFO] Creating evaluation environment...")
    eval_env_kwargs = env_kwargs.copy()
    eval_env_kwargs['seed'] = 123  # Different seed for unbiased evaluation
    eval_env = make_vec_env(
        CustomT1DEnv,
        n_envs=1,
        env_kwargs=eval_env_kwargs
    )

    # =========================================================================
    # CALLBACKS
    # =========================================================================
    
    # Checkpoint callback: saves model periodically during training
    # Useful for resuming training or analysing training progression
    checkpoint_callback = CheckpointCallback(
        save_freq=max(100_000 // n_envs, 1),  # Save every ~100k steps
        save_path=checkpoint_dir,
        name_prefix=f"ppo_{patient_id}",
        save_replay_buffer=False,
        save_vecnormalize=False
    )

    # Evaluation callback: runs periodic evaluation and saves best model
    # The best model is determined by mean reward over n_eval_episodes
    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=checkpoint_dir,
        log_path=eval_log_dir,
        eval_freq=max(50_000 // n_envs, 1),   # Evaluate every ~50k steps
        n_eval_episodes=5,                     # Average over 5 episodes
        deterministic=True,                    # Use deterministic actions
        render=False,
        verbose=1
    )

    # Combine callbacks into a single callback list
    callback = CallbackList([checkpoint_callback, eval_callback])

    return train_env, eval_env, callback