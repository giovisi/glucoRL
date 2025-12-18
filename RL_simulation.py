from custom_env import CustomT1DEnv
from stable_baselines3 import PPO
from datetime import timedelta
import os
import pandas as pd

def simulate_RL(trial, scenario, PATIENT_ID, SIM_DAYS, MODEL_SEARCH_PATHS):
    print(f"Simulation starts...")

    # Find model with fallback logic
    model_path = None
    for mpath in MODEL_SEARCH_PATHS:
        test_path = mpath if mpath.endswith('.zip') else f"{mpath}.zip"
        if os.path.exists(test_path):
            model_path = mpath.replace('.zip', '')
            break
    
    if model_path is None:
        print(f"[ERROR] Model not found! Tried:")
        for mpath in MODEL_SEARCH_PATHS:
            print(f"    - {mpath}.zip")
        print("\n  → Train first: python train_ppo.py")
        return
    
    if trial == 0:  # Only print once
        print(f"    Loading model: {model_path}")
    
    env_rl = CustomT1DEnv(patient_name=PATIENT_ID, custom_scenario=scenario, episode_days=SIM_DAYS)
    model = PPO.load(model_path)
    obs, _ = env_rl.reset()
    
    rl_bg_history = []
    rl_cgm_history = []
    rl_time_history = []
    step_count = 0
    max_steps = env_rl.max_episode_steps
    
    if trial == 0:  # Print config once
        print(f"    Sample time: {env_rl.sample_time} min")
        print(f"    Episode days: {env_rl.episode_days}")
        print(f"    Max steps: {max_steps}")
    
    done = False
    while not done and step_count < max_steps:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = env_rl.step(action)
        
        # Record data
        current_time = env_rl.env.time
        current_bg = env_rl.env.patient.observation.Gsub
        current_cgm = env_rl.env.CGM_hist[-1] if env_rl.env.CGM_hist else current_bg
        
        rl_bg_history.append(current_bg)
        rl_cgm_history.append(current_cgm)
        rl_time_history.append(current_time)
        step_count += 1
        
        # Check termination conditions
        if terminated or truncated or current_time >= scenario.start_time + timedelta(days=SIM_DAYS):
            done = True
            if trial == 0:  # Print termination info for first trial
                termination_reason = "Normal completion"
                if 'catastrophic_failure' in info and info['catastrophic_failure']:
                    termination_reason = f"Catastrophic: {info['failure_reason']}"
                elif 'internal_termination' in info:
                    bg = info.get('bg_at_termination', 'N/A')
                    termination_reason = f"Internal: BG={bg:.1f}" if isinstance(bg, float) else f"Internal: {bg}"
                elif truncated:
                    termination_reason = "Max steps reached"
                print(f"    → {termination_reason} ({step_count}/{max_steps} steps)")

    print(f"Simulation completed!")
    results_rl = pd.DataFrame(data={'BG': rl_bg_history, 'CGM': rl_cgm_history}, index=rl_time_history)
    return results_rl