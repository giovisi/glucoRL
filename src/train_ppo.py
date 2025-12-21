"""
PPO Training Script for T1D Blood Glucose Control.

This script trains a Proximal Policy Optimisation (PPO) agent to control
insulin delivery for blood glucose regulation in Type 1 Diabetic patients.

The training process:
1. Creates vectorised environments for parallel data collection
2. Configures PPO with hyperparameters tuned for the glucose control task
3. Runs training with periodic evaluation and checkpointing
4. Saves the final model and cleans up intermediate checkpoints

Key features:
- Multi-environment parallel training for efficiency
- TensorBoard logging for training visualisation
- Best model checkpointing via EvalCallback
- Configurable episode length and total training steps
"""

import setup_paths  # Must be first import - configures sys.path for simglucose

import os
from stable_baselines3 import PPO
import torch

from train_setup import setup_train_config, setup_train_env


# =============================================================================
# CONFIGURATION
# =============================================================================

# Patient and directory settings
PATIENT_ID = 'adolescent#003'          # Virtual patient for training

# Output directories for training artefacts
LOG_DIR = "./train/ppo_tensorboard/"   # TensorBoard logs for visualisation
CHECKPOINT_DIR = "./train/checkpoints/"  # Intermediate model checkpoints
EVAL_LOG_DIR = "./train/eval_logs/"    # Evaluation episode logs
RESULTS_DIR = "./train/results/"       # Final trained models


# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

def save_train_results(model, results_dir, patient_id):
    """
    Save the final trained model and clean up checkpoint files.
    
    After training completes, this function:
    1. Saves the final model to the results directory
    2. Removes intermediate checkpoint files to save disk space
    
    Args:
        model: Trained PPO model to save
        results_dir: Directory path for final model
        patient_id: Patient identifier used in filename
    """
    # Save final trained model
    os.makedirs(results_dir, exist_ok=True)
    final_model_name = f"{results_dir}ppo_{patient_id}_final"
    model.save(final_model_name)
    print(f"\n✓ Final model saved: {final_model_name}.zip")
    
    # Cleanup: Remove intermediate checkpoint files to save disk space
    # The best model from EvalCallback is preserved separately
    print("\n[CLEANUP] Removing checkpoint files...")
    checkpoint_files = [f for f in os.listdir(CHECKPOINT_DIR) if f.startswith(f"ppo_{PATIENT_ID}_")]
    for checkpoint_file in checkpoint_files:
        checkpoint_path = os.path.join(CHECKPOINT_DIR, checkpoint_file)
        try:
            os.remove(checkpoint_path)
        except Exception as e:
            print(f"  ✗ Failed to delete {checkpoint_file}: {e}")
    
    if checkpoint_files:
        print(f"[CLEANUP] Removed {len(checkpoint_files)} checkpoint file(s)")


# =============================================================================
# MAIN TRAINING FUNCTION
# =============================================================================
    

# =============================================================================
# MAIN TRAINING FUNCTION
# =============================================================================

def train(n_envs=4, total_steps=2_000_000, episode_days=1):
    """
    Train a PPO agent for blood glucose control.
    
    Sets up the training environment, configures the PPO algorithm with
    hyperparameters optimised for the glucose control task, and runs
    the training loop with evaluation callbacks.
    
    Args:
        n_envs: Number of parallel environments for data collection (default: 4)
        total_steps: Total training timesteps across all environments (default: 2M)
        episode_days: Length of each training episode in days (default: 1)
    """
    # =========================================================================
    # ENVIRONMENT SETUP
    # =========================================================================
    
    # Get environment configuration parameters
    env_kwargs = setup_train_config(
        episode_days, total_steps, n_envs, PATIENT_ID,
        LOG_DIR, CHECKPOINT_DIR, EVAL_LOG_DIR, RESULTS_DIR
    )
    print(env_kwargs)

    # Create vectorised training and evaluation environments with callbacks
    train_env, eval_env, callback = setup_train_env(
        n_envs, env_kwargs, PATIENT_ID, CHECKPOINT_DIR, EVAL_LOG_DIR
    )

    # =========================================================================
    # PPO MODEL CONFIGURATION
    # =========================================================================
    
    # Initialise PPO with hyperparameters tuned for glucose control
    model = PPO(
        "MlpPolicy",               # Multi-layer perceptron policy
        train_env,
        verbose=1,
        
        # Learning parameters
        learning_rate=3e-4,        # Adam optimiser learning rate
        n_steps=2048,              # Steps per environment before update
        batch_size=256,            # Minibatch size for SGD
        n_epochs=10,               # Epochs per policy update
        
        # Discount and advantage estimation
        gamma=0.995,               # High discount factor for long-term rewards
        gae_lambda=0.95,           # GAE lambda for advantage estimation
        
        # PPO-specific parameters
        clip_range=0.2,            # PPO clipping parameter
        ent_coef=0.01,             # Entropy coefficient for exploration
        vf_coef=0.5,               # Value function loss coefficient
        max_grad_norm=0.5,         # Gradient clipping threshold
        normalize_advantage=True,  # Normalise advantages for stability
        
        # Neural network architecture
        policy_kwargs=dict(
            net_arch=dict(
                pi=[128, 128, 64],  # Policy network: 3 layers
                vf=[128, 128, 64]   # Value network: 3 layers
            ),
            activation_fn=torch.nn.ReLU
        ),
        
        # Logging and compute
        tensorboard_log=LOG_DIR,
        device="auto"              # Auto-select CPU/GPU
    )

    # =========================================================================
    # TRAINING LOOP
    # =========================================================================
    
    try:
        model.learn(
            total_timesteps=total_steps,
            callback=callback,      # EvalCallback for checkpointing
            progress_bar=True
        )
    except KeyboardInterrupt:
        print("\n⚠ Training interrupted manually.")

    # =========================================================================
    # SAVE AND CLEANUP
    # =========================================================================
    
    save_train_results(model, RESULTS_DIR, PATIENT_ID)
    
    # Close environments to release resources
    train_env.close()
    eval_env.close()


# =============================================================================
# ENTRY POINT
# =============================================================================

if __name__ == "__main__":
    train(n_envs=4, total_steps=2_000_000, episode_days=1)