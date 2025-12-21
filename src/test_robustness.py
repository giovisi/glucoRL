"""
Robustness Testing Module for RL Blood Glucose Control Agent.

This script evaluates the trained RL agent's performance under challenging
scenarios that differ from typical training conditions. It tests the agent's
ability to handle:
- High carbohydrate meals (hyperglycaemia stress)
- Skipped meals (hypoglycaemia stress)
- Late-night meals (nocturnal hypoglycaemia risk)
- Random meal patterns (general robustness)

Outputs include detailed metrics, CSV data and visualisation plots.
"""

import setup_paths  # Must be first import - configures sys.path for simglucose

import os
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from stable_baselines3 import PPO
from custom_env import CustomT1DEnv
from simglucose.simulation.scenario import CustomScenario
import random

# Set working directory to script location for relative path resolution
os.chdir(os.path.dirname(os.path.abspath(__file__)))


# =============================================================================
# CONFIGURATION
# =============================================================================

PATIENT_ID = 'adolescent#003'                    # Virtual patient for testing
MODEL_PATH = "train/checkpoints/best_model.zip"  # Always use the best trained model
OUTPUT_FOLDER = f"test/robustness/Robustness_Test_{datetime.now().strftime('%Y%m%d_%H%M')}"

# =============================================================================
# SCENARIO DEFINITIONS
# =============================================================================

def get_scenarios():
    """
    Define stress test scenarios to evaluate agent robustness.
    
    Creates four challenging scenarios that test different failure modes:
    1. High Carbs: Tests response to hyperglycaemia-inducing large meals
    2. Missed Lunch: Tests if agent avoids over-dosing when meals are skipped
    3. Late Dinner: Tests handling of late-night meals (nocturnal hypo risk)
    4. Random Chaos: Tests general adaptability to unpredictable patterns
    
    Returns:
        dict: Mapping of scenario names to CustomScenario objects
    """
    now = datetime.now()
    start_time = datetime.combine(now.date(), datetime.min.time())  # Midnight
    
    scenarios = {}

    # -------------------------------------------------------------------------
    # Scenario 1: High Carbs (Hyperglycaemia Stress)
    # Large meals that challenge the agent's ability to prevent high BG
    # -------------------------------------------------------------------------
    high_carb_meals = [
        (timedelta(hours=8), 80),    # Large breakfast
        (timedelta(hours=13), 110),  # Very large lunch
        (timedelta(hours=20), 90)    # Heavy dinner
    ]
    scenarios['High_Carbs'] = CustomScenario(
        start_time=start_time,
        scenario=[(start_time + t, c) for t, c in high_carb_meals]
    )

    # -------------------------------------------------------------------------
    # Scenario 2: Missed Lunch (Hypoglycaemia Stress)
    # Tests if agent correctly reduces insulin when meals are skipped
    # -------------------------------------------------------------------------
    missed_lunch_meals = [
        (timedelta(hours=8), 40),
        # No lunch - agent must recognise this and reduce insulin
        (timedelta(hours=20), 60)
    ]
    scenarios['Missed_Lunch'] = CustomScenario(
        start_time=start_time,
        scenario=[(start_time + t, c) for t, c in missed_lunch_meals]
    )

    # -------------------------------------------------------------------------
    # Scenario 3: Late Dinner (Nocturnal Hypoglycaemia Risk)
    # Late meal timing increases risk of overnight low blood glucose
    # -------------------------------------------------------------------------
    late_dinner_meals = [
        (timedelta(hours=8), 40),
        (timedelta(hours=13), 60),
        (timedelta(hours=22, minutes=30), 70)  # Very late dinner
    ]
    scenarios['Late_Dinner'] = CustomScenario(
        start_time=start_time,
        scenario=[(start_time + t, c) for t, c in late_dinner_meals]
    )

    # -------------------------------------------------------------------------
    # Scenario 4: Random Chaos (General Robustness)
    # Unpredictable meal times and amounts to test adaptability
    # -------------------------------------------------------------------------
    rand_meals = []
    for _ in range(4):  # Generate 4 random meals
        t = timedelta(hours=random.randint(6, 22), minutes=random.randint(0, 59))
        c = random.randint(20, 100)
        rand_meals.append((start_time + t, c))
    rand_meals.sort(key=lambda x: x[0])  # Sort chronologically
    scenarios['Random_Chaos'] = CustomScenario(start_time=start_time, scenario=rand_meals)

    return scenarios


# =============================================================================
# METRICS ANALYSIS
# =============================================================================

def analyze_episode(df):
    """
    Calculate detailed clinical glycaemic metrics for an episode.
    
    Computes standard metrics used to evaluate blood glucose control quality:
    - TIR: Time in Range (70-180 mg/dL)
    - TBR: Time Below Range (<70 mg/dL)
    - TAR: Time Above Range (>180 mg/dL)
    - Severe Hypo: Time in severe hypoglycaemia (<54 mg/dL)
    - CV: Coefficient of Variation (glucose stability measure)
    
    Args:
        df: DataFrame with 'BG' column containing blood glucose values
        
    Returns:
        dict: Dictionary of computed metrics, empty if no data
    """
    BG = df['BG'].values
    if len(BG) == 0:
        return {}
    
    total = len(BG)
    
    # Calculate time-in-range percentages
    tir = ((BG >= 70) & (BG <= 180)).sum() / total * 100
    tbr = (BG < 70).sum() / total * 100
    tar = (BG > 180).sum() / total * 100
    severe_hypo = (BG < 54).sum() / total * 100
    
    # Calculate statistical measures
    mean_bg = BG.mean()
    std_bg = BG.std()
    cv = (std_bg / mean_bg) * 100 if mean_bg > 0 else 0  # Coefficient of variation
    
    return {
        'TIR': tir,
        'TBR': tbr,
        'TAR': tar,
        'Severe Hypo': severe_hypo,
        'Mean BG': mean_bg,
        'CV': cv
    }


# =============================================================================
# VISUALISATION
# =============================================================================

def plot_scenarios_grid(results, output_folder):
    """
    Create a 2x2 grid plot showing all test scenarios.
    
    Each subplot displays the blood glucose trajectory for one scenario
    with target range highlighting and key metrics in the title.
    
    Args:
        results: Dictionary mapping scenario names to result DataFrames
        output_folder: Directory path to save the generated plot
    """
    scenarios = list(results.keys())
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    axes = axes.flatten()  # Flatten for easy iteration

    for idx, name in enumerate(scenarios):
        df = results[name]
        ax = axes[idx]
        
        # Plot blood glucose trajectory
        ax.plot(df.index, df['BG'], color='#1f77b4', linewidth=2, label='RL Agent')
        
        # Add clinical threshold lines and target zone
        ax.axhline(180, color='orange', linestyle='--', alpha=0.5)  # Hyper threshold
        ax.axhline(70, color='red', linestyle='--', alpha=0.5)      # Hypo threshold
        ax.fill_between(df.index, 70, 180, color='green', alpha=0.1)  # Target range
        
        # Add title with key metrics
        metrics = analyze_episode(df)
        title = f"{name}\nTIR: {metrics['TIR']:.1f}% | TBR: {metrics['TBR']:.1f}% | Mean: {metrics['Mean BG']:.0f}"
        ax.set_title(title, fontweight='bold', fontsize=10)
        ax.grid(True, alpha=0.3)
        
        # Format x-axis as time of day
        import matplotlib.dates as mdates
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=45)

    plt.tight_layout()
    plt.savefig(os.path.join(output_folder, "robustness_grid.png"))
    plt.close()


# =============================================================================
# MAIN EXECUTION
# =============================================================================

def main():
    """
    Execute the complete robustness test suite.
    
    Loads the trained model, runs all stress test scenarios and generates
    a comprehensive report with metrics and visualisations.
    """
    # Verify model exists before proceeding
    if not os.path.exists(MODEL_PATH):
        print(f"ERROR: Model not found at {MODEL_PATH}")
        print("Ensure training is complete or update the model path.")
        return

    # Create output directory
    os.makedirs(OUTPUT_FOLDER, exist_ok=True)
    print(f"--- ROBUSTNESS TEST STARTED ---")
    print(f"Model: {MODEL_PATH}")
    print(f"Output: {OUTPUT_FOLDER}")

    # Load trained model
    try:
        model = PPO.load(MODEL_PATH)
    except Exception as e:
        print(f"Error loading model: {e}")
        return

    # Get test scenarios and prepare results storage
    scenarios = get_scenarios()
    results = {}
    
    # Initialise report with header
    report_lines = [
        f"ROBUSTNESS TEST REPORT",
        f"Date: {datetime.now()}",
        f"Model: {MODEL_PATH}",
        "-" * 60,
        f"{'SCENARIO':<15} | {'TIR %':<8} | {'TBR %':<8} | {'TAR %':<8} | {'MEAN':<6} | {'CV %':<6}",
        "-" * 60
    ]

    # =========================================================================
    # RUN EACH SCENARIO
    # =========================================================================
    for name, scenario in scenarios.items():
        print(f"Running scenario: {name}...")
        
        # Create environment with current scenario
        env = CustomT1DEnv(patient_name=PATIENT_ID, custom_scenario=scenario, episode_days=1)
        obs, _ = env.reset()
        
        # Data collection lists
        bg_hist = []
        time_hist = []
        
        # Run episode until termination
        done = False
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, _, terminated, truncated, _ = env.step(action)
            
            # Record true blood glucose and timestamp
            bg_hist.append(env.env.patient.observation.Gsub)
            time_hist.append(env.env.time)
            
            if terminated or truncated:
                done = True
                
        # Save scenario results to CSV
        df = pd.DataFrame({'BG': bg_hist}, index=time_hist)
        results[name] = df
        df.to_csv(os.path.join(OUTPUT_FOLDER, f"{name}.csv"))
        
        # Calculate and record metrics
        m = analyze_episode(df)
        line = f"{name:<15} | {m['TIR']:>7.1f}  | {m['TBR']:>7.1f}  | {m['TAR']:>7.1f}  | {m['Mean BG']:>6.0f} | {m['CV']:>6.1f}"
        report_lines.append(line)
        print(f"  -> TIR: {m['TIR']:.1f}%")

    # =========================================================================
    # GENERATE OUTPUTS
    # =========================================================================
    
    # Save text report
    with open(os.path.join(OUTPUT_FOLDER, "report.txt"), "w") as f:
        f.write("\n".join(report_lines))

    # Generate visualisation plots
    print("Generating plots...")
    plot_scenarios_grid(results, OUTPUT_FOLDER)
    
    # Print summary to console
    print(f"\n✅ TEST COMPLETED. Results saved to: {OUTPUT_FOLDER}")
    print("\n--- REPORT PREVIEW ---")
    for line in report_lines:
        print(line)
    print("-" * 60)


if __name__ == "__main__":
    main()