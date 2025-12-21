"""
RL Simulation Module for T1D Blood Glucose Control.

This module provides the simulation function for running a trained
reinforcement learning agent (PPO) on the T1D glucose control task.
It handles model loading, environment setup and data collection during
the simulation episode.
"""

from custom_env import CustomT1DEnv
from stable_baselines3 import PPO
from datetime import timedelta
import os
import pandas as pd


# =============================================================================
# RL SIMULATION FUNCTION
# =============================================================================

def simulate_RL(trial, scenario, PATIENT_ID, SIM_DAYS, MODEL_SEARCH_PATHS):
    """
    Run a simulation episode using a trained RL agent.
    
    Loads a pre-trained PPO model and executes it on the custom T1D
    environment for the specified duration. Records blood glucose
    and CGM readings throughout the episode.
    
    Args:
        trial: Trial number (used for conditional logging)
        scenario: Meal scenario defining carbohydrate intake timing
        PATIENT_ID: Virtual patient identifier from SimGlucose database
        SIM_DAYS: Duration of simulation in days
        MODEL_SEARCH_PATHS: List of paths to search for the trained model
        
    Returns:
        pd.DataFrame: Simulation results with 'BG' and 'CGM' columns,
                      indexed by timestamp
    """
    print(f"Simulation starts...")

    # =========================================================================
    # MODEL LOADING: Search paths in priority order
    # =========================================================================
    model_path = None
    for mpath in MODEL_SEARCH_PATHS:
        # Handle both with and without .zip extension
        test_path = mpath if mpath.endswith('.zip') else f"{mpath}.zip"
        if os.path.exists(test_path):
            model_path = mpath.replace('.zip', '')
            break
    
    # Exit gracefully if no model found
    if model_path is None:
        print(f"[ERROR] Model not found! Tried:")
        for mpath in MODEL_SEARCH_PATHS:
            print(f"    - {mpath}.zip")
        print("\n  → Train first: python train_ppo.py")
        return
    
    # Log model path only on first trial to reduce output clutter
    if trial == 0:
        print(f"    Loading model: {model_path}")

    # =========================================================================
    # ENVIRONMENT SETUP
    # =========================================================================
    # Create custom environment with the specified scenario and patient
    env_rl = CustomT1DEnv(patient_name=PATIENT_ID, custom_scenario=scenario, episode_days=SIM_DAYS)
    
    # Load the pre-trained PPO model
    model = PPO.load(model_path)
    
    # Reset environment and get initial observation
    obs, _ = env_rl.reset()
    
    # =========================================================================
    # DATA COLLECTION SETUP
    # =========================================================================
    # Lists to store simulation history
    rl_bg_history = []    # True blood glucose values
    rl_cgm_history = []   # CGM sensor readings (may have noise/delay)
    rl_time_history = []  # Timestamps for each step
    step_count = 0
    max_steps = env_rl.max_episode_steps
    
    # Log configuration on first trial only
    if trial == 0:
        print(f"    Sample time: {env_rl.sample_time} min")
        print(f"    Episode days: {env_rl.episode_days}")
        print(f"    Max steps: {max_steps}")

    # =========================================================================
    # SIMULATION LOOP
    # =========================================================================
    done = False
    while not done and step_count < max_steps:
        # Get action from trained policy (deterministic for evaluation)
        action, _ = model.predict(obs, deterministic=True)
        
        # Execute action in environment
        obs, reward, terminated, truncated, info = env_rl.step(action)
        
        # Record current state data
        current_time = env_rl.env.time
        current_bg = env_rl.env.patient.observation.Gsub  # True BG
        current_cgm = env_rl.env.CGM_hist[-1] if env_rl.env.CGM_hist else current_bg
        
        # Append to history lists
        rl_bg_history.append(current_bg)
        rl_cgm_history.append(current_cgm)
        rl_time_history.append(current_time)
        step_count += 1
        
        # ---------------------------------------------------------------------
        # TERMINATION CHECK
        # Episode ends if: terminated, truncated or simulation duration reached
        # ---------------------------------------------------------------------
        if terminated or truncated or current_time >= scenario.start_time + timedelta(days=SIM_DAYS):
            done = True
            
            # Log termination reason for first trial only
            if trial == 0:
                termination_reason = "Normal completion"
                if 'catastrophic_failure' in info and info['catastrophic_failure']:
                    termination_reason = f"Catastrophic: {info['failure_reason']}"
                elif 'internal_termination' in info:
                    bg = info.get('bg_at_termination', 'N/A')
                    termination_reason = f"Internal: BG={bg:.1f}" if isinstance(bg, float) else f"Internal: {bg}"
                elif truncated:
                    termination_reason = "Max steps reached"
                print(f"    → {termination_reason} ({step_count}/{max_steps} steps)")

    # =========================================================================
    # RESULTS COMPILATION
    # =========================================================================
    print(f"Simulation completed!")
    
    # Create DataFrame with BG and CGM data indexed by timestamp
    results_rl = pd.DataFrame(data={'BG': rl_bg_history, 'CGM': rl_cgm_history}, index=rl_time_history)
    return results_rl