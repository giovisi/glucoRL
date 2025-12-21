"""
Scenario Generation and Metrics Module for T1D Simulations.

This module provides utilities for:
- Generating multi-day meal scenarios with realistic timing and carbohydrate amounts
- Creating standard SimGlucose simulation environments
- Computing detailed glycaemic control metrics
- Displaying comparative metrics tables
"""

import setup_paths  # Must be first import - configures sys.path for simglucose

import numpy as np
from datetime import datetime, timedelta

# SimGlucose components for T1D simulation
from simglucose.patient.t1dpatient import T1DPatient
from simglucose.sensor.cgm import CGMSensor
from simglucose.actuator.pump import InsulinPump
from simglucose.simulation.env import T1DSimEnv
from simglucose.simulation.scenario import CustomScenario


# =============================================================================
# SCENARIO GENERATION
# =============================================================================

def get_extended_scenario(n_days=7):
    """
    Generate a multi-day meal scenario with randomised timing and amounts.
    
    Creates three meals per day (breakfast, lunch and dinner) with:
    - Randomised timing within typical meal windows
    - Randomised carbohydrate amounts within realistic ranges
    
    Args:
        n_days: Number of days to generate meals for (default: 7)
        
    Returns:
        CustomScenario: SimGlucose scenario object containing all meal events
    """
    now = datetime.now()
    start_time = datetime.combine(now.date(), datetime.min.time())  # Midnight
    meal_events = []
    
    for day in range(n_days):
        day_offset = timedelta(days=day)
        
        # Randomise carbohydrate amounts within typical ranges for each meal
        breakfast_carbs = np.random.randint(45, 55)   # Moderate breakfast
        lunch_carbs = np.random.randint(65, 75)       # Larger lunch
        dinner_carbs = np.random.randint(55, 65)      # Medium dinner
        
        # Randomise meal times within typical windows (up to 30 min variation)
        breakfast_time = start_time + day_offset + timedelta(hours=7, minutes=np.random.randint(0, 30))
        lunch_time = start_time + day_offset + timedelta(hours=12, minutes=np.random.randint(0, 30))
        dinner_time = start_time + day_offset + timedelta(hours=19, minutes=np.random.randint(0, 30))
        
        # Add all three meals for this day
        meal_events.extend([
            (breakfast_time, breakfast_carbs),
            (lunch_time, lunch_carbs),
            (dinner_time, dinner_carbs)
        ])
        
    return CustomScenario(start_time=start_time, scenario=meal_events)

# =============================================================================
# ENVIRONMENT CREATION
# =============================================================================

def create_standard_env(scenario, patient_id):
    """
    Create a standard SimGlucose T1D simulation environment.
    
    Assembles a complete simulation environment with:
    - Virtual T1D patient from the SimGlucose database
    - Dexcom CGM sensor for glucose monitoring
    - Insulet insulin pump for delivery
    
    Args:
        scenario: CustomScenario object defining meal events
        patient_id: Patient identifier string (e.g. 'adolescent#003')
        
    Returns:
        T1DSimEnv: Configured simulation environment ready for use
    """
    patient = T1DPatient.withName(patient_id)
    sensor = CGMSensor.withName('Dexcom', seed=1)  # Fixed seed for reproducibility
    pump = InsulinPump.withName('Insulet')
    return T1DSimEnv(patient, sensor, pump, scenario)


# =============================================================================
# GLYCAEMIC METRICS CALCULATION
# =============================================================================

# =============================================================================
# GLYCAEMIC METRICS CALCULATION
# =============================================================================

def calculate_detailed_metrics(df):
    """
    Calculate comprehensive glycaemic control metrics from simulation data.
    
    Computes the standard clinical metrics for evaluating blood glucose control:
    - TIR: Time in Range (70-180 mg/dL) - optimal zone
    - TBR: Time Below Range (<70 mg/dL) - hypoglycaemia
    - TBR Severe: Time in severe hypoglycaemia (<54 mg/dL)
    - TAR: Time Above Range (>180 mg/dL) - hyperglycaemia
    - TAR Severe: Time in severe hyperglycaemia (>250 mg/dL)
    - Mean BG: Average blood glucose
    - CV%: Coefficient of variation (measure of glucose variability)
    - HBGI/LBGI: High/Low Blood Glucose Index (risk metrics)
    
    Args:
        df: DataFrame with 'BG' column containing blood glucose values
        
    Returns:
        dict: Dictionary containing all computed metrics, or None if empty data
    """
    BG = df['BG'].values
    if len(BG) == 0:
        return None
    
    # Calculate time-in-range percentages for each glycaemic zone
    tir = ((BG >= 70) & (BG <= 180)).sum() / len(BG) * 100
    tbr = (BG < 70).sum() / len(BG) * 100
    tbr_severe = (BG < 54).sum() / len(BG) * 100
    tar = (BG > 180).sum() / len(BG) * 100
    tar_severe = (BG > 250).sum() / len(BG) * 100
    
    # Calculate statistical measures
    mean_bg = BG.mean()
    std_bg = BG.std()
    cv_bg = (std_bg / mean_bg) * 100 if mean_bg > 0 else 0  # Coefficient of variation
    
    # Calculate risk indices (based on Kovatchev formula)
    hbgi = ((np.log(BG[BG > 180]) ** 1.084) - 5.381).sum() / len(BG) if (BG > 180).any() else 0
    lbgi = ((np.log(BG[BG < 70]) ** 1.084) - 5.381).sum() / len(BG) if (BG < 70).any() else 0
    
    return {
        'TIR (70-180)': tir,
        'TBR (<70)': tbr,
        'TBR Severe (<54)': tbr_severe,
        'TAR (>180)': tar,
        'TAR Severe (>250)': tar_severe,
        'Mean BG': mean_bg,
        'Std BG': std_bg,
        'CV%': cv_bg,
        'HBGI': hbgi,
        'LBGI': lbgi
    }


# =============================================================================
# METRICS DISPLAY
# =============================================================================

# =============================================================================
# METRICS DISPLAY
# =============================================================================

def print_metrics_table(results_dict):
    """
    Display a formatted comparison table of glycaemic metrics.
    
    Prints a console table comparing key metrics across all controllers,
    making it easy to evaluate relative performance at a glance.
    
    Args:
        results_dict: Dictionary mapping controller names to result DataFrames
    """
    # Print table header
    print("\n" + "="*90 + "\n" + " " * 30 + "COMPARATIVE METRICS" + "\n" + "="*90)
    
    # Build header row with controller names
    header = f"{'Metric':<20}"
    for name in results_dict.keys():
        header += f"{name:>20}"
    print(header + "\n" + "-"*90)
    
    # Calculate metrics for all controllers
    all_metrics = {}
    for name, df in results_dict.items():
        all_metrics[name] = calculate_detailed_metrics(df)
    
    # Define which metrics to display (subset of all available)
    metric_names = [
        'TIR (70-180)',
        'TBR (<70)',
        'TBR Severe (<54)',
        'TAR (>180)',
        'TAR Severe (>250)',
        'Mean BG',
        'CV%'
    ]
    
    # Print each metric row
    for metric in metric_names:
        row = f"{metric:<20}"
        for name in results_dict.keys():
            value = all_metrics[name][metric]
            # Format BG values without %, others with %
            row += f"{value:>20.2f}" if 'BG' in metric else f"{value:>19.2f}%"
        print(row)
    
    print("="*90)