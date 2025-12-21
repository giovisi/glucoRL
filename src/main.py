"""
Main Evaluation Script for T1D Blood Glucose Control Comparison.

This script runs comparative simulations between three control strategies:
1. Basal-Bolus: Traditional insulin therapy controller
2. PID: Proportional-Integral-Derivative controller
3. RL Smart: Reinforcement learning agent with meal lookahead

The evaluation runs multiple trials with randomised meal scenarios to
ensure statistically meaningful comparisons across controllers.
"""

import setup_paths  # Must be first import - configures sys.path for simglucose

import os
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

# Local project imports
from helpers import perform_advanced_analysis, plot_comparison  
from scenario import create_standard_env, get_extended_scenario, print_metrics_table
from RL_simulation import simulate_RL

# SimGlucose controller and simulation components
from simglucose.controller.basal_bolus_ctrller import BBController
from simglucose.controller.pid_ctrller import PIDController
from simglucose.simulation.sim_engine import SimObj, sim


# =============================================================================
# SIMULATION CONFIGURATION
# =============================================================================


# =============================================================================
# SIMULATION CONFIGURATION
# =============================================================================

# Duration and patient settings
SIM_DAYS = 7                                    # Length of each simulation trial
PATIENT_ID = 'adolescent#003'                   # Virtual patient from SimGlucose database

# RL model configuration
RL_MODEL_NAME = f"ppo_{PATIENT_ID}_final"       # Trained PPO model identifier
RL_MODEL_PATH = f"train/results/{RL_MODEL_NAME}"  # Primary model directory
N_TRIALS = 5                                     # Number of trials for statistical robustness

# Model search priority list - tries each path in order until a valid model is found
MODEL_SEARCH_PATHS = [
    "train/checkpoints/best_model",              # 1. Best model from EvalCallback
    f"train/results/ppo_{PATIENT_ID}_final",     # 2. Final model after training
    RL_MODEL_PATH,                               # 3. Configured model path
]


# =============================================================================
# UTILITY FUNCTIONS
# ============================================================================= 

def save_test_results(aggregated_results):
    """
    Save all aggregated simulation results to timestamped folder.
    
    Creates a new results directory with timestamp and exports each
    controller's results to a separate CSV file.
    
    Args:
        aggregated_results: Dictionary mapping controller names to DataFrames
        
    Returns:
        str: Path to the created results folder
    """
    # Create timestamped output folder
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    folder = f"test/results/Results_{timestamp}"
    os.makedirs(folder, exist_ok=True)
    print(f"\n[SAVING RESULTS] → {folder}/")

    # Export each controller's results to CSV
    for name, df in aggregated_results.items():
        safe_name = name.replace(" ", "_").lower()
        file_path = os.path.join(folder, f"{safe_name}.csv")
        df.to_csv(file_path)
        print(f"  ✓ {safe_name}.csv")

    return folder


# =============================================================================
# MAIN EVALUATION LOOP
# =============================================================================


def main():
    """
    Main evaluation function comparing glucose control strategies.
    
    Runs N_TRIALS simulations for each controller (Basal-Bolus, PID and RL)
    using randomised meal scenarios. Results are aggregated across trials
    and saved with visualisation plots for analysis.
    """
    print(f"\n{'='*90}\n" + " " * 25 + f"GLUCOSE CONTROL EVALUATION\n" + f"{'='*90}")
    
    # Container for collecting results from all trials
    all_results = {'Basal-Bolus': [], 'PID': [], 'RL Smart': []}

    # =========================================================================
    # SIMULATION LOOP: Run each controller for N_TRIALS
    # =========================================================================
    for trial in range(N_TRIALS):
        print(f"\n[TRIAL {trial+1}/{N_TRIALS}]")
        
        # Set random seed for reproducibility (different seed per trial)
        np.random.seed(trial * 42)
        scenario = get_extended_scenario(n_days=SIM_DAYS)

        # ---------------------------------------------------------------------
        # Controller 1: Basal-Bolus (traditional insulin therapy)
        # Standard clinical approach with basal rate and meal boluses
        # ---------------------------------------------------------------------
        print("  → Running Basal-Bolus...")
        bb_controller = BBController()
        env_bb = create_standard_env(scenario, PATIENT_ID)
        sim_obj_bb = SimObj(env_bb, bb_controller, timedelta(days=SIM_DAYS), animate=False, path=f'./test/temp/temp_bb_trial{trial}')
        results_bb = sim(sim_obj_bb)
        all_results['Basal-Bolus'].append(results_bb)

        # ---------------------------------------------------------------------
        # Controller 2: PID (feedback control)
        # Classic control theory approach using proportional-integral-derivative
        # ---------------------------------------------------------------------
        print("  → Running PID...")
        pid_controller = PIDController(P=1.0e-4, I=1.0e-7, D=3.9e-3, target=140)
        env_pid = create_standard_env(scenario, PATIENT_ID)
        sim_obj_pid = SimObj(env_pid, pid_controller, timedelta(days=SIM_DAYS), animate=False, path=f'./test/temp/temp_pid_trial{trial}')
        results_pid = sim(sim_obj_pid)
        all_results['PID'].append(results_pid)

        # ---------------------------------------------------------------------
        # Controller 3: RL Smart (reinforcement learning with meal lookahead)
        # Trained PPO agent that anticipates upcoming meals for proactive dosing
        # ---------------------------------------------------------------------
        print("  → Running RL...")
        results_rl = simulate_RL(trial, scenario, PATIENT_ID=PATIENT_ID, SIM_DAYS=SIM_DAYS, MODEL_SEARCH_PATHS=MODEL_SEARCH_PATHS)
        all_results['RL Smart'].append(results_rl)

    # =========================================================================
    # RESULTS AGGREGATION: Combine all trials per controller
    # =========================================================================
    print("\n[AGGREGATING RESULTS...]")
    aggregated_results = {}
    for method, trials in all_results.items():
        # Concatenate all trial DataFrames for each controller
        aggregated_results[method] = pd.concat(trials)

    # Display summary metrics table
    print_metrics_table(aggregated_results)

    # =========================================================================
    # OUTPUT: Save results and generate visualisations
    # =========================================================================
    folder = save_test_results(aggregated_results)
    
    # Generate comparison plot with all controllers
    plot_comparison(aggregated_results, patient_id=PATIENT_ID, save_path=os.path.join(folder, "comparison_plot.png"))
    print(f"  ✓ comparison_plot.png")
    
    # Generate advanced analysis plots (violin and clinical zones)
    perform_advanced_analysis(aggregated_results, folder)
    
    print("\n" + "="*90 + "\n" + " " * 30 + "EVALUATION COMPLETED" + "\n" + "="*90)


if __name__ == "__main__":
    main()