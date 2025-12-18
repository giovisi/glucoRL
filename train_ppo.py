import setup_paths  # Must be first import - configures sys.path for simglucose

import os
from stable_baselines3 import PPO
import torch

from train_setup import setup_train_config, setup_train_env

PATIENT_ID = 'adolescent#003'

LOG_DIR = "./train/ppo_tensorboard/"
CHECKPOINT_DIR = "./train/checkpoints/"
EVAL_LOG_DIR = "./train/eval_logs/"
RESULTS_DIR = "./train/results/"

def save_train_results(model, results_dir, patient_id):
    # Save Final Model
    os.makedirs(results_dir, exist_ok=True)
    final_model_name = f"{results_dir}ppo_{patient_id}_final"
    model.save(final_model_name)
    print(f"\n✓ Final model saved: {final_model_name}.zip")
    
    # Cleanup: Delete all checkpoint files from this run
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
    

def train(n_envs=4, total_steps=2_000_000, episode_days=1):
    # Initialize training configuration
    env_kwargs = setup_train_config(episode_days, total_steps, n_envs, PATIENT_ID, LOG_DIR, CHECKPOINT_DIR,
                                   EVAL_LOG_DIR, RESULTS_DIR)

    # Setup Environments and Callbacks
    train_env, eval_env, callback = setup_train_env(n_envs, env_kwargs, PATIENT_ID, CHECKPOINT_DIR, EVAL_LOG_DIR)

    # Initialize PPO Model
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
        ent_coef=0.01,  # Increased from 0.005 for better exploration
        vf_coef=0.5,
        max_grad_norm=0.5,
        normalize_advantage=True,  # Added for better training stability
        policy_kwargs=dict(
            net_arch=dict(pi=[128, 128, 64], vf=[128, 128, 64]),
            activation_fn=torch.nn.ReLU
        ),
        tensorboard_log=LOG_DIR,
        device="auto"
    )

    # Training Loop
    try:
        model.learn(
            total_timesteps=total_steps,
            callback=callback,
            progress_bar=True
        )
    except KeyboardInterrupt:
        print("\n⚠ Training interrupted manually.")

    # Save Training Results
    save_train_results(model, RESULTS_DIR, PATIENT_ID)
    
    train_env.close()
    eval_env.close()


if __name__ == "__main__":
    train(n_envs=4, total_steps=2_000_000, episode_days=1)