import os
import sys

# Project root = folder where main.py lives
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))

# Add the *inner* simglucose repo root to sys.path
SUBMODULE_ROOT = os.path.join(PROJECT_ROOT, "simglucose")
sys.path.insert(0, SUBMODULE_ROOT)

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

import helpers
from simglucose.controller.basal_bolus_ctrller import BBController
from simglucose.controller.pid_ctrller import PIDController
from simglucose.patient.t1dpatient import T1DPatient
from simglucose.sensor.cgm import CGMSensor
from simglucose.actuator.pump import InsulinPump
from simglucose.simulation.env import T1DSimEnv
from simglucose.simulation.scenario import CustomScenario
from simglucose.simulation.sim_engine import SimObj, sim
from simglucose.controller.base import Action

from stable_baselines3 import PPO
from custom_env import CustomT1DEnv 

# =============================================================================
# CONFIGURATION
# =============================================================================
SIM_DAYS = 7 
PATIENT_ID = 'adolescent#003'
RL_MODEL_NAME = f"ppo_{PATIENT_ID}_final"  # Points to new paper-based model
RL_MODEL_PATH = f"train/results/{RL_MODEL_NAME}"  # Updated path to results folder
N_TRIALS = 5

# Model search priority list
MODEL_SEARCH_PATHS = [
    "train/checkpoints/best_model",                             # 1. Best from EvalCallback
    f"train/results/ppo_{PATIENT_ID}_final",                    # 2. Final training model
    RL_MODEL_PATH,                                              # 3. Configured path
] 

# ... (Scenario Generation Logic - No Changes) ...
def get_extended_scenario(n_days=7):
    now = datetime.now()
    start_time = datetime.combine(now.date(), datetime.min.time())
    meal_events = []
    for day in range(n_days):
        day_offset = timedelta(days=day)
        breakfast_carbs = np.random.randint(45, 55)
        lunch_carbs = np.random.randint(65, 75)
        dinner_carbs = np.random.randint(55, 65)
        breakfast_time = start_time + day_offset + timedelta(hours=7, minutes=np.random.randint(0, 30))
        lunch_time = start_time + day_offset + timedelta(hours=12, minutes=np.random.randint(0, 30))
        dinner_time = start_time + day_offset + timedelta(hours=19, minutes=np.random.randint(0, 30))
        meal_events.extend([(breakfast_time, breakfast_carbs), (lunch_time, lunch_carbs), (dinner_time, dinner_carbs)])
    return CustomScenario(start_time=start_time, scenario=meal_events)

# ... (Metric and Helper functions - No Changes) ...
def create_standard_env(scenario):
    patient = T1DPatient.withName(PATIENT_ID)
    sensor = CGMSensor.withName('Dexcom', seed=1)
    pump = InsulinPump.withName('Insulet')
    return T1DSimEnv(patient, sensor, pump, scenario)

def calculate_detailed_metrics(df):
    BG = df['BG'].values
    if len(BG) == 0: return None
    tir = ((BG >= 70) & (BG <= 180)).sum() / len(BG) * 100
    tbr = (BG < 70).sum() / len(BG) * 100
    tbr_severe = (BG < 54).sum() / len(BG) * 100
    tar = (BG > 180).sum() / len(BG) * 100
    tar_severe = (BG > 250).sum() / len(BG) * 100
    mean_bg = BG.mean()
    std_bg = BG.std()
    cv_bg = (std_bg / mean_bg) * 100 if mean_bg > 0 else 0
    hbgi = ((np.log(BG[BG > 180]) ** 1.084) - 5.381).sum() / len(BG) if (BG > 180).any() else 0
    lbgi = ((np.log(BG[BG < 70]) ** 1.084) - 5.381).sum() / len(BG) if (BG < 70).any() else 0
    return {'TIR (70-180)': tir, 'TBR (<70)': tbr, 'TBR Severe (<54)': tbr_severe, 'TAR (>180)': tar, 'TAR Severe (>250)': tar_severe, 'Mean BG': mean_bg, 'Std BG': std_bg, 'CV%': cv_bg, 'HBGI': hbgi, 'LBGI': lbgi}

def print_metrics_table(results_dict):
    print("\n" + "="*90 + "\n" + " " * 30 + "COMPARATIVE METRICS" + "\n" + "="*90)
    header = f"{'Metric':<20}"
    for name in results_dict.keys(): header += f"{name:>20}"
    print(header + "\n" + "-"*90)
    all_metrics = {}
    for name, df in results_dict.items(): all_metrics[name] = calculate_detailed_metrics(df)
    metric_names = ['TIR (70-180)', 'TBR (<70)', 'TBR Severe (<54)', 'TAR (>180)', 'TAR Severe (>250)', 'Mean BG', 'CV%']
    for metric in metric_names:
        row = f"{metric:<20}"
        for name in results_dict.keys():
            value = all_metrics[name][metric]
            row += f"{value:>20.2f}" if 'BG' in metric else f"{value:>19.2f}%"
        print(row)
    print("="*90)

# =============================================================================
# ADVANCED PLOTTING FUNCTIONS
# =============================================================================
def plot_glycemic_distribution(results_dict, save_path):
    """
    Generates a Violin Plot to compare glucose density distribution.
    Shows if RL maintains more stable glucose (denser center).
    """
    plt.figure(figsize=(10, 6))
    
    data_to_plot = []
    labels = []
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c']  # Blue, Orange, Green
    
    for name, df in results_dict.items():
        if not df.empty:
            data_to_plot.append(df['BG'].values)
            labels.append(name)
            
    if not data_to_plot:
        return

    parts = plt.violinplot(data_to_plot, showmeans=True, showmedians=True)
    
    # Styling
    for i, pc in enumerate(parts['bodies']):
        pc.set_facecolor(colors[i % len(colors)])
        pc.set_alpha(0.6)
        
    plt.axhline(180, color='red', linestyle='--', alpha=0.5, label='Hyper Limit')
    plt.axhline(70, color='red', linestyle='--', alpha=0.5, label='Hypo Limit')
    plt.fill_between([0.5, len(labels)+0.5], 70, 180, color='green', alpha=0.1, label='Target Range')
    
    plt.xticks(np.arange(1, len(labels) + 1), labels, fontsize=11, fontweight='bold')
    plt.ylabel('Glucose (mg/dL)')
    plt.title('Glucose Distribution Density (Violin Plot)')
    plt.legend(loc='upper right')
    plt.grid(axis='y', linestyle=':', alpha=0.5)
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

def plot_tir_breakdown(results_dict, save_path):
    """
    Generates stacked bar chart with 5 standard clinical zones.
    """
    zones = {
        'Very Low (<54)': (-np.inf, 54),
        'Low (54-70)': (54, 70),
        'Target (70-180)': (70, 180),
        'High (180-250)': (180, 250),
        'Very High (>250)': (250, np.inf)
    }
    
    zone_colors = ['#8b0000', '#ff4444', '#32cd32', '#ffa500', '#8b4500']
    names = []
    zone_data = {k: [] for k in zones.keys()}
    
    for name, df in results_dict.items():
        if df.empty:
            continue
        names.append(name)
        bg = df['BG'].values
        total = len(bg)
        
        for zone_name, (low, high) in zones.items():
            count = ((bg >= low) & (bg < high)).sum()
            pct = (count / total) * 100
            zone_data[zone_name].append(pct)
            
    if not names:
        return

    # Plotting
    fig, ax = plt.subplots(figsize=(10, 6))
    bottom = np.zeros(len(names))
    
    for (zone_label, values), color in zip(zone_data.items(), zone_colors):
        ax.bar(names, values, bottom=bottom, label=zone_label, color=color, alpha=0.8, width=0.6)
        
        # Add percentage labels if space is sufficient
        for i, v in enumerate(values):
            if v > 5:  # Only show if > 5%
                ax.text(i, bottom[i] + v/2, f"{v:.1f}%", ha='center', va='center', 
                        color='white' if color != '#32cd32' else 'black', fontweight='bold')
        bottom += values

    ax.set_ylabel('Percentage of Time (%)')
    ax.set_title('Clinical Glycemic Zones Breakdown')
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

def perform_advanced_analysis(results_dict, folder):
    """Generate advanced visualization plots."""
    print("\n[ANALYSIS] Generating advanced plots...")
    plot_glycemic_distribution(results_dict, os.path.join(folder, "analysis_violin.png"))
    plot_tir_breakdown(results_dict, os.path.join(folder, "analysis_zones.png"))
    print("  ✓ Violin plot saved.")
    print("  ✓ Clinical Zones plot saved.")

# =============================================================================
# MAIN SIMULATION LOOP
# =============================================================================
def main():
    print(f"\n{'='*90}\n" + " " * 25 + f"GLUCOSE CONTROL EVALUATION\n" + f"{'='*90}")
    
    # Aggregate results container
    all_results = {'Basal-Bolus': [], 'PID': [], 'RL Smart': []}

    for trial in range(N_TRIALS):
        print(f"\n[TRIAL {trial+1}/{N_TRIALS}]")
        np.random.seed(trial * 42)
        scenario = get_extended_scenario(n_days=SIM_DAYS)

        # 1. BB
        print("  → Running Basal-Bolus...")
        bb_controller = BBController()
        env_bb = create_standard_env(scenario)
        sim_obj_bb = SimObj(env_bb, bb_controller, timedelta(days=SIM_DAYS), animate=False, path=f'./test/temp/temp_bb_trial{trial}')
        results_bb = sim(sim_obj_bb)
        all_results['Basal-Bolus'].append(results_bb)

        # 2. PID
        print("  → Running PID...")
        pid_controller = PIDController(P=1.0e-4, I=1.0e-7, D=3.9e-3, target=140)
        env_pid = create_standard_env(scenario)
        sim_obj_pid = SimObj(env_pid, pid_controller, timedelta(days=SIM_DAYS), animate=False, path=f'./test/temp/temp_pid_trial{trial}')
        results_pid = sim(sim_obj_pid)
        all_results['PID'].append(results_pid)

        # 3. RL
        print("  → Running RL (Smart PPO)...")
        
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

        results_rl = pd.DataFrame(data={'BG': rl_bg_history, 'CGM': rl_cgm_history}, index=rl_time_history)
        all_results['RL Smart'].append(results_rl)

    print("\n[AGGREGATING RESULTS...]")
    aggregated_results = {}
    for method, trials in all_results.items():
        aggregated_results[method] = pd.concat(trials)

    print_metrics_table(aggregated_results)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    folder = f"test/results/Results_{timestamp}"
    os.makedirs(folder, exist_ok=True)
    print(f"\n[SAVING RESULTS] → {folder}/")
    for name, df in aggregated_results.items():
        safe_name = name.replace(" ", "_").lower()
        file_path = os.path.join(folder, f"{safe_name}.csv")
        df.to_csv(file_path)
        print(f"  ✓ {safe_name}.csv")
    
    helpers.plot_comparison(aggregated_results, patient_id=PATIENT_ID, save_path=os.path.join(folder, "comparison_plot.png"))
    print(f"  ✓ comparison_plot.png")
    
    # Generate advanced analysis plots
    perform_advanced_analysis(aggregated_results, folder)
    
    print("\n" + "="*90 + "\n" + " " * 30 + "EVALUATION COMPLETED" + "\n" + "="*90)

if __name__ == "__main__":
    main()