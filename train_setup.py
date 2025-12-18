import os

from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import SubprocVecEnv
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback, CallbackList

from custom_env import CustomT1DEnv
from reward_functions import paper_reward

def setup_train_config(episode_days, total_steps, n_envs, patient_id, log_dir, checkpoint_dir,
                       eval_log_dir, results_dir):
    # Directory setup
    for directory in [log_dir, checkpoint_dir, eval_log_dir, results_dir]:
        os.makedirs(directory, exist_ok=True)

    # Environment Configuration
    env_kwargs = {
        'patient_name': patient_id,
        'reward_fun': paper_reward,
        'seed': 42,
        'episode_days': episode_days
    }

    # Read dynamic values from temporary environment
    temp_env = CustomT1DEnv(**env_kwargs)
    sample_time = temp_env.sample_time
    max_episode_steps = temp_env.max_episode_steps
    temp_env.close()
    
    episodes_total = total_steps // max_episode_steps

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


def setup_train_env(n_envs, env_kwargs, patient_id, checkpoint_dir, eval_log_dir):
    # Create environments
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
    eval_env_kwargs['seed'] = 123  # Different seed for evaluation
    eval_env = make_vec_env(
        CustomT1DEnv,
        n_envs=1,
        env_kwargs=eval_env_kwargs
    )

    # Callbacks
    checkpoint_callback = CheckpointCallback(
        save_freq=max(100_000 // n_envs, 1),
        save_path=checkpoint_dir,
        name_prefix=f"ppo_{patient_id}",
        save_replay_buffer=False,
        save_vecnormalize=False
    )

    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=checkpoint_dir,
        log_path=eval_log_dir,
        eval_freq=max(50_000 // n_envs, 1),
        n_eval_episodes=5,
        deterministic=True,
        render=False,
        verbose=1
    )

    callback = CallbackList([checkpoint_callback, eval_callback])

    return train_env, eval_env, callback