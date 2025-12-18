import matplotlib.pyplot as plt
import os
import numpy as np

def print_glycemic_metrics(df, controller_name):
    """Calcola e stampa TIR, TBR, TAR e media."""
    BG = df['BG'].values
    total_steps = len(BG)
    
    if total_steps == 0:
        print(f"No data for {controller_name}")
        return

    tir = ((BG >= 70) & (BG <= 180)).sum() / total_steps * 100
    tbr = (BG < 70).sum() / total_steps * 100
    tar = (BG > 180).sum() / total_steps * 100
    mean_bg = BG.mean()
    
    print(f"\n--- {controller_name} Glycemic Metrics ---")
    print(f"TIR (70-180 mg/dL): {tir:.2f}%")
    print(f"TBR (<70 mg/dL):     {tbr:.2f}%")
    print(f"TAR (>180 mg/dL):    {tar:.2f}%")
    print(f"Mean BG:             {mean_bg:.2f} mg/dL")

def plot_comparison(results_dict, patient_id="Unknown", save_path=None):
    """
    Genera grafici comparativi MIGLIORATI con subplot multipli
    """
    n_methods = len(results_dict)
    if n_methods == 0:
        return

    # FIGURA PRINCIPALE: Time series + statistiche
    fig = plt.figure(figsize=(16, 10))
    gs = fig.add_gridspec(n_methods + 1, 2, hspace=0.3, wspace=0.3)
    
    # Trova range temporale comune
    first_df = list(results_dict.values())[0]
    t_start = first_df.index[0]
    t_end = first_df.index[-1]
    
    # Colori per metodi
    colors = {'Basal-Bolus': '#1f77b4', 'PID': '#ff7f0e', 'RL Smart': '#2ca02c'}
    
    # =============================================================================
    # SUBPLOT 1-N: Time series glucosio
    # =============================================================================
    for idx, (name, df) in enumerate(results_dict.items()):
        ax = fig.add_subplot(gs[idx, 0])
        
        # Plot BG con zona target evidenziata
        ax.plot(df.index, df['BG'], linewidth=1.5, color=colors.get(name, 'black'), alpha=0.8)
        
        # Zone colorate
        ax.axhline(180, color='red', linestyle='--', linewidth=1, alpha=0.4)
        ax.axhline(70, color='red', linestyle='--', linewidth=1, alpha=0.4)
        ax.axhline(250, color='darkred', linestyle=':', linewidth=1, alpha=0.3)
        ax.axhline(54, color='darkred', linestyle=':', linewidth=1, alpha=0.3)
        ax.fill_between(df.index, 70, 180, color='green', alpha=0.15, label='Target Range')
        
        # Calcola metriche per titolo
        BG = df['BG'].values
        tir = ((BG >= 70) & (BG <= 180)).sum() / len(BG) * 100
        mean_bg = BG.mean()
        
        ax.set_ylabel('Glucose (mg/dL)', fontsize=10)
        ax.set_title(f'{name} | TIR: {tir:.1f}% | Mean: {mean_bg:.1f} mg/dL', 
                     fontsize=11, fontweight='bold')
        ax.grid(True, linestyle=':', alpha=0.4)
        ax.set_ylim([40, 400])
        
        if idx == n_methods - 1:
            ax.set_xlabel('Time', fontsize=10)
        else:
            ax.set_xticklabels([])

    # =============================================================================
    # SUBPLOT N+1: Distribuzione glucosio (Violini)
    # =============================================================================
    ax_violin = fig.add_subplot(gs[:, 1])
    
    bg_distributions = []
    labels = []
    
    for name, df in results_dict.items():
        bg_distributions.append(df['BG'].values)
        labels.append(name)
    
    parts = ax_violin.violinplot(bg_distributions, positions=range(len(labels)),
                                   showmeans=True, showmedians=True, widths=0.7)
    
    # Colora violini
    for idx, pc in enumerate(parts['bodies']):
        method_name = labels[idx]
        pc.set_facecolor(colors.get(method_name, 'gray'))
        pc.set_alpha(0.6)
    
    # Zone target
    ax_violin.axhline(180, color='red', linestyle='--', linewidth=1, alpha=0.5, label='Hyperglycemia')
    ax_violin.axhline(70, color='red', linestyle='--', linewidth=1, alpha=0.5, label='Hypoglycemia')
    ax_violin.fill_between([-0.5, len(labels)-0.5], 70, 180, color='green', alpha=0.1)
    
    ax_violin.set_xticks(range(len(labels)))
    ax_violin.set_xticklabels(labels, rotation=15, ha='right')
    ax_violin.set_ylabel('Glucose Distribution (mg/dL)', fontsize=11, fontweight='bold')
    ax_violin.set_title('Glucose Distribution Comparison', fontsize=12, fontweight='bold')
    ax_violin.grid(axis='y', linestyle=':', alpha=0.4)
    ax_violin.set_ylim([40, 400])
    ax_violin.legend(loc='upper right', fontsize=9)
    
    # Titolo generale
    fig.suptitle(f'Glucose Control Comparison - Patient: {patient_id}', 
                 fontsize=14, fontweight='bold', y=0.995)
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"[PLOT] Saved to: {save_path}")
    
    plt.show()

def save_all_results(results_dict, folder_name="simulation_results"):
    """Salva CSV e Immagini."""
    if not os.path.exists(folder_name):
        os.makedirs(folder_name)

    for name, df in results_dict.items():
        safe_name = name.replace(" ", "_").lower()
        file_path = os.path.join(folder_name, f"{safe_name}.csv")
        df.to_csv(file_path)

    plot_path = os.path.join(folder_name, "comparison_plot.png")
    plot_comparison(results_dict, save_path=plot_path)

def plot_glycemic_zones_pie(results_dict, save_path=None):
    """
    Crea un pie chart delle zone glicemiche per confronto
    """
    fig, axes = plt.subplots(1, len(results_dict), figsize=(5*len(results_dict), 5))
    
    if len(results_dict) == 1:
        axes = [axes]
    
    for ax, (name, df) in zip(axes, results_dict.items()):
        BG = df['BG'].values
        
        # Calcola percentuali
        severe_hypo = (BG < 54).sum() / len(BG) * 100
        hypo = ((BG >= 54) & (BG < 70)).sum() / len(BG) * 100
        tir = ((BG >= 70) & (BG <= 180)).sum() / len(BG) * 100
        hyper = ((BG > 180) & (BG <= 250)).sum() / len(BG) * 100
        severe_hyper = (BG > 250).sum() / len(BG) * 100
        
        sizes = [severe_hypo, hypo, tir, hyper, severe_hyper]
        labels = [f'Severe Hypo\n(<54)', f'Hypo\n(54-70)', f'TIR\n(70-180)', 
                  f'Hyper\n(180-250)', f'Severe Hyper\n(>250)']
        colors = ['#8b0000', '#ff6347', '#90ee90', '#ffa500', '#8b0000']
        explode = (0.1, 0.05, 0, 0.05, 0.1)
        
        ax.pie(sizes, explode=explode, labels=labels, colors=colors, autopct='%1.1f%%',
               shadow=True, startangle=90)
        ax.set_title(f'{name}\nTIR: {tir:.1f}%', fontweight='bold')
    
    plt.suptitle('Glycemic Zones Distribution', fontsize=14, fontweight='bold')
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    
    plt.show()


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