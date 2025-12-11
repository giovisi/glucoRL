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
import os

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
RL_MODEL_PATH = "ppo_adolescent003_paper_final"  # Points to new paper-based model
N_TRIALS = 5 

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

# ... (Main Loop) ...
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
        sim_obj_bb = SimObj(env_bb, bb_controller, timedelta(days=SIM_DAYS), animate=False, path=f'./temp_bb_trial{trial}')
        results_bb = sim(sim_obj_bb)
        all_results['Basal-Bolus'].append(results_bb)

        # 2. PID
        print("  → Running PID...")
        pid_controller = PIDController(P=1.0e-4, I=1.0e-7, D=3.9e-3, target=140)
        env_pid = create_standard_env(scenario)
        sim_obj_pid = SimObj(env_pid, pid_controller, timedelta(days=SIM_DAYS), animate=False, path=f'./temp_pid_trial{trial}')
        results_pid = sim(sim_obj_pid)
        all_results['PID'].append(results_pid)

        # 3. RL
        print("  → Running RL (Smart PPO)...")
        if not os.path.exists(f"{RL_MODEL_PATH}.zip"):
            print(f"[ERROR] Modello {RL_MODEL_PATH}.zip non trovato!\nEsegui prima: python train_ppo.py")
            return

        env_rl = CustomT1DEnv(patient_name=PATIENT_ID, custom_scenario=scenario)
        model = PPO.load(RL_MODEL_PATH)
        obs, _ = env_rl.reset()
        rl_bg_history = []
        rl_time_history = []
        done = False
        step_count = 0
        max_steps = SIM_DAYS * 288 
        
        while not done and step_count < max_steps:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done, truncated, info = env_rl.step(action)
            current_time = env_rl.env.time
            current_bg = env_rl.env.patient.observation.Gsub
            rl_bg_history.append(current_bg)
            rl_time_history.append(current_time)
            step_count += 1
            if current_time >= scenario.start_time + timedelta(days=SIM_DAYS): done = True

        results_rl = pd.DataFrame(data={'BG': rl_bg_history}, index=rl_time_history)
        all_results['RL Smart'].append(results_rl)

    print("\n[AGGREGATING RESULTS...]")
    aggregated_results = {}
    for method, trials in all_results.items():
        aggregated_results[method] = pd.concat(trials)

    print_metrics_table(aggregated_results)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    folder = f"Results_{timestamp}"
    os.makedirs(folder, exist_ok=True)
    print(f"\n[SAVING RESULTS] → {folder}/")
    for name, df in aggregated_results.items():
        safe_name = name.replace(" ", "_").lower()
        file_path = os.path.join(folder, f"{safe_name}.csv")
        df.to_csv(file_path)
        print(f"  ✓ {safe_name}.csv")
    
    helpers.plot_comparison(aggregated_results, patient_id=PATIENT_ID, save_path=os.path.join(folder, "comparison_plot.png"))
    print(f"  ✓ comparison_plot.png")
    print("\n" + "="*90 + "\n" + " " * 30 + "EVALUATION COMPLETED" + "\n" + "="*90)

if __name__ == "__main__":
    main()