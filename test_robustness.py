import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import os
import sys
from datetime import datetime, timedelta
from stable_baselines3 import PPO
from custom_env import CustomT1DEnv
from simglucose.simulation.scenario import CustomScenario
import random

# --- AGGIUNGI QUESTO BLOCCO ---
# Imposta la cartella di lavoro corrente alla cartella dove si trova questo file
os.chdir(os.path.dirname(os.path.abspath(__file__)))
# ------------------------------

# ================= CONFIGURAZIONE =================
PATIENT_ID = 'adolescent#003'
MODEL_PATH = "train/checkpoints_paper/best_model.zip"  # Usa SEMPRE il best model
OUTPUT_FOLDER = f"test/robustness/Robustness_Test_{datetime.now().strftime('%Y%m%d_%H%M')}"
# ==================================================

def get_scenarios():
    """Definisce scenari di stress test per l'agente"""
    now = datetime.now()
    start_time = datetime.combine(now.date(), datetime.min.time())
    
    scenarios = {}

    # 1. High Carbs (Stress Iperglicemia)
    high_carb_meals = [
        (timedelta(hours=8), 80),   # Colazione abbondante
        (timedelta(hours=13), 110), # Pranzo enorme
        (timedelta(hours=20), 90)   # Cena pesante
    ]
    scenarios['High_Carbs'] = CustomScenario(start_time=start_time, scenario=[(start_time + t, c) for t, c in high_carb_meals])

    # 2. Missed Lunch (Stress Ipoglicemia - L'agente sa non dare insulina?)
    missed_lunch_meals = [
        (timedelta(hours=8), 40),
        # Niente pranzo!
        (timedelta(hours=20), 60)
    ]
    scenarios['Missed_Lunch'] = CustomScenario(start_time=start_time, scenario=[(start_time + t, c) for t, c in missed_lunch_meals])

    # 3. Late Dinner (Rischio Ipo Notturna)
    late_dinner_meals = [
        (timedelta(hours=8), 40),
        (timedelta(hours=13), 60),
        (timedelta(hours=22, minutes=30), 70) # Cena molto tardi
    ]
    scenarios['Late_Dinner'] = CustomScenario(start_time=start_time, scenario=[(start_time + t, c) for t, c in late_dinner_meals])

    # 4. Random Chaos (Robustezza generale)
    # Generiamo pasti a caso
    rand_meals = []
    for _ in range(4): # 4 pasti random
        t = timedelta(hours=random.randint(6, 22), minutes=random.randint(0, 59))
        c = random.randint(20, 100)
        rand_meals.append((start_time + t, c))
    rand_meals.sort(key=lambda x: x[0])
    scenarios['Random_Chaos'] = CustomScenario(start_time=start_time, scenario=rand_meals)

    return scenarios

def analyze_episode(df):
    """Calcola metriche cliniche dettagliate"""
    BG = df['BG'].values
    if len(BG) == 0: return {}
    
    total = len(BG)
    tir = ((BG >= 70) & (BG <= 180)).sum() / total * 100
    tbr = (BG < 70).sum() / total * 100
    tar = (BG > 180).sum() / total * 100
    severe_hypo = (BG < 54).sum() / total * 100
    mean_bg = BG.mean()
    std_bg = BG.std()
    cv = (std_bg / mean_bg) * 100 if mean_bg > 0 else 0 # Coeff. variazione (stabilità)
    
    return {
        'TIR': tir,
        'TBR': tbr,
        'TAR': tar,
        'Severe Hypo': severe_hypo,
        'Mean BG': mean_bg,
        'CV': cv
    }

def plot_scenarios_grid(results, output_folder):
    """Crea un grafico 2x2 con tutti gli scenari"""
    scenarios = list(results.keys())
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    axes = axes.flatten()

    for idx, name in enumerate(scenarios):
        df = results[name]
        ax = axes[idx]
        
        # Plot BG
        ax.plot(df.index, df['BG'], color='#1f77b4', linewidth=2, label='RL Agent')
        
        # Zone
        ax.axhline(180, color='orange', linestyle='--', alpha=0.5)
        ax.axhline(70, color='red', linestyle='--', alpha=0.5)
        ax.fill_between(df.index, 70, 180, color='green', alpha=0.1)
        
        # Titolo con metriche
        metrics = analyze_episode(df)
        title = f"{name}\nTIR: {metrics['TIR']:.1f}% | TBR: {metrics['TBR']:.1f}% | Mean: {metrics['Mean BG']:.0f}"
        ax.set_title(title, fontweight='bold', fontsize=10)
        ax.grid(True, alpha=0.3)
        
        # Formatta asse X
        import matplotlib.dates as mdates
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=45)

    plt.tight_layout()
    plt.savefig(os.path.join(output_folder, "robustness_grid.png"))
    plt.close()

def main():
    if not os.path.exists(MODEL_PATH):
        print(f"ERRORE: Modello non trovato in {MODEL_PATH}")
        print("Assicurati di aver finito il training o rinomina il file corretto.")
        return

    os.makedirs(OUTPUT_FOLDER, exist_ok=True)
    print(f"--- AVVIO TEST DI ROBUSTEZZA ---")
    print(f"Modello: {MODEL_PATH}")
    print(f"Output: {OUTPUT_FOLDER}")

    try:
        model = PPO.load(MODEL_PATH)
    except Exception as e:
        print(f"Errore caricamento modello: {e}")
        return

    scenarios = get_scenarios()
    results = {}
    report_lines = []
    
    report_lines.append(f"ROBUSTNESS TEST REPORT")
    report_lines.append(f"Date: {datetime.now()}")
    report_lines.append(f"Model: {MODEL_PATH}")
    report_lines.append("-" * 60)
    report_lines.append(f"{'SCENARIO':<15} | {'TIR %':<8} | {'TBR %':<8} | {'TAR %':<8} | {'MEAN':<6} | {'CV %':<6}")
    report_lines.append("-" * 60)

    for name, scenario in scenarios.items():
        print(f"Running scenario: {name}...")
        
        # Crea Env
        env = CustomT1DEnv(patient_name=PATIENT_ID, custom_scenario=scenario, episode_days=1)
        obs, _ = env.reset()
        
        bg_hist = []
        time_hist = []
        
        # Run Episode
        done = False
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, _, terminated, truncated, _ = env.step(action)
            
            bg_hist.append(env.env.patient.observation.Gsub)
            time_hist.append(env.env.time)
            
            if terminated or truncated:
                done = True
                
        # Save Data
        df = pd.DataFrame({'BG': bg_hist}, index=time_hist)
        results[name] = df
        df.to_csv(os.path.join(OUTPUT_FOLDER, f"{name}.csv"))
        
        # Analyze
        m = analyze_episode(df)
        line = f"{name:<15} | {m['TIR']:>7.1f}  | {m['TBR']:>7.1f}  | {m['TAR']:>7.1f}  | {m['Mean BG']:>6.0f} | {m['CV']:>6.1f}"
        report_lines.append(line)
        print(f"  -> TIR: {m['TIR']:.1f}%")

    # Save Report
    with open(os.path.join(OUTPUT_FOLDER, "report.txt"), "w") as f:
        f.write("\n".join(report_lines))

    # Plots
    print("Generazione grafici...")
    plot_scenarios_grid(results, OUTPUT_FOLDER)
    
    print(f"\n✅ TEST COMPLETATO. Vedi cartella: {OUTPUT_FOLDER}")
    print("\n--- REPORT ANTEPRIMA ---")
    for line in report_lines:
        print(line)

if __name__ == "__main__":
    main()