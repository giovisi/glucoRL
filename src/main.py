import setup_paths  # Must be first import - configures sys.path for simglucose

import os
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

from helpers import perform_advanced_analysis, plot_comparison  
from scenario import create_standard_env, get_extended_scenario, print_metrics_table
from RL_simulation import simulate_RL

from simglucose.controller.basal_bolus_ctrller import BBController
from simglucose.controller.pid_ctrller import PIDController
from simglucose.simulation.sim_engine import SimObj, sim


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

def save_test_results(aggregated_results):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    folder = f"test/results/Results_{timestamp}"
    os.makedirs(folder, exist_ok=True)
    print(f"\n[SAVING RESULTS] → {folder}/")

    for name, df in aggregated_results.items():
        safe_name = name.replace(" ", "_").lower()
        file_path = os.path.join(folder, f"{safe_name}.csv")
        df.to_csv(file_path)
        print(f"  ✓ {safe_name}.csv")

    return folder


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
        env_bb = create_standard_env(scenario, PATIENT_ID)
        sim_obj_bb = SimObj(env_bb, bb_controller, timedelta(days=SIM_DAYS), animate=False, path=f'./test/temp/temp_bb_trial{trial}')
        results_bb = sim(sim_obj_bb)
        all_results['Basal-Bolus'].append(results_bb)

        # 2. PID
        print("  → Running PID...")
        pid_controller = PIDController(P=1.0e-4, I=1.0e-7, D=3.9e-3, target=140)
        env_pid = create_standard_env(scenario, PATIENT_ID)
        sim_obj_pid = SimObj(env_pid, pid_controller, timedelta(days=SIM_DAYS), animate=False, path=f'./test/temp/temp_pid_trial{trial}')
        results_pid = sim(sim_obj_pid)
        all_results['PID'].append(results_pid)

        # 3. RL
        print("  → Running RL...")
        results_rl = simulate_RL(trial, scenario, PATIENT_ID=PATIENT_ID, SIM_DAYS=SIM_DAYS, MODEL_SEARCH_PATHS=MODEL_SEARCH_PATHS)
        all_results['RL Smart'].append(results_rl)

    print("\n[AGGREGATING RESULTS...]")
    aggregated_results = {}
    for method, trials in all_results.items():
        aggregated_results[method] = pd.concat(trials)

    print_metrics_table(aggregated_results)

    folder = save_test_results(aggregated_results)
    
    plot_comparison(aggregated_results, patient_id=PATIENT_ID, save_path=os.path.join(folder, "comparison_plot.png"))
    print(f"  ✓ comparison_plot.png")
    
    # Generate advanced analysis plots
    perform_advanced_analysis(aggregated_results, folder)
    
    print("\n" + "="*90 + "\n" + " " * 30 + "EVALUATION COMPLETED" + "\n" + "="*90)

if __name__ == "__main__":
    main()