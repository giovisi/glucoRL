import setup_paths  # Must be first import - configures sys.path for simglucose
import numpy as np
from datetime import datetime, timedelta
from simglucose.patient.t1dpatient import T1DPatient
from simglucose.sensor.cgm import CGMSensor
from simglucose.actuator.pump import InsulinPump
from simglucose.simulation.env import T1DSimEnv
from simglucose.simulation.scenario import CustomScenario


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
def create_standard_env(scenario, patient_id):
    patient = T1DPatient.withName(patient_id)
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