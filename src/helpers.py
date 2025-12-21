"""
Helper Functions for T1D Blood Glucose Control Analysis and Visualisation.

This module provides utility functions for:
- Computing glycaemic metrics (TIR, TBR, TAR)
- Generating comparative visualisation plots
- Saving simulation results to files
- Performing advanced statistical analysis

All visualisations follow clinical standards for blood glucose zones.
"""

import matplotlib.pyplot as plt
import os
import numpy as np


# =============================================================================
# GLYCAEMIC METRICS COMPUTATION
# =============================================================================

def print_glycemic_metrics(df, controller_name):
    """
    Calculate and print standard glycaemic control metrics.
    
    Computes the key clinical metrics used to evaluate blood glucose control:
    - TIR (Time in Range): percentage of time BG is between 70-180 mg/dL
    - TBR (Time Below Range): percentage of time BG is below 70 mg/dL (hypoglycaemia)
    - TAR (Time Above Range): percentage of time BG is above 180 mg/dL (hyperglycaemia)
    - Mean BG: average blood glucose level
    
    Args:
        df: DataFrame containing a 'BG' column with blood glucose values
        controller_name: Name of the controller for display purposes
    """
    BG = df['BG'].values
    total_steps = len(BG)
    
    if total_steps == 0:
        print(f"No data for {controller_name}")
        return

    # Calculate percentage of time in each glycaemic zone
    tir = ((BG >= 70) & (BG <= 180)).sum() / total_steps * 100   # Target range
    tbr = (BG < 70).sum() / total_steps * 100                     # Hypoglycaemia
    tar = (BG > 180).sum() / total_steps * 100                    # Hyperglycaemia
    mean_bg = BG.mean()
    
    # Display formatted metrics
    print(f"\n--- {controller_name} Glycemic Metrics ---")
    print(f"TIR (70-180 mg/dL): {tir:.2f}%")
    print(f"TBR (<70 mg/dL):     {tbr:.2f}%")
    print(f"TAR (>180 mg/dL):    {tar:.2f}%")
    print(f"Mean BG:             {mean_bg:.2f} mg/dL")


# =============================================================================
# VISUALISATION FUNCTIONS
# =============================================================================

def plot_comparison(results_dict, patient_id="Unknown", save_path=None):
    """
    Generate comparative visualisation plots with multiple subplots.
    
    Creates a comprehensive figure showing:
    - Time series of blood glucose for each controller method
    - Violin plot distribution comparison across all methods
    - Target range highlighting and key metrics in titles
    
    Args:
        results_dict: Dictionary mapping controller names to DataFrames with 'BG' column
        patient_id: Patient identifier for plot title
        save_path: Optional file path to save the figure
    """
    n_methods = len(results_dict)
    if n_methods == 0:
        return

    # Create main figure with grid layout for subplots
    fig = plt.figure(figsize=(16, 10))
    gs = fig.add_gridspec(n_methods + 1, 2, hspace=0.3, wspace=0.3)
    
    # Extract time range from first dataset for consistency
    first_df = list(results_dict.values())[0]
    t_start = first_df.index[0]
    t_end = first_df.index[-1]
    
    # Colour palette for different control methods
    colors = {'Basal-Bolus': '#1f77b4', 'PID': '#ff7f0e', 'RL Smart': '#2ca02c'}
    
    # =========================================================================
    # TIME SERIES SUBPLOTS: Blood glucose over time for each controller
    # =========================================================================
    for idx, (name, df) in enumerate(results_dict.items()):
        ax = fig.add_subplot(gs[idx, 0])
        
        # Plot blood glucose trajectory
        ax.plot(df.index, df['BG'], linewidth=1.5, color=colors.get(name, 'black'), alpha=0.8)
        
        # Add clinical threshold lines:
        # - 180 mg/dL: hyperglycaemia threshold
        # - 70 mg/dL: hypoglycaemia threshold  
        # - 250 mg/dL: severe hyperglycaemia
        # - 54 mg/dL: severe hypoglycaemia
        ax.axhline(180, color='red', linestyle='--', linewidth=1, alpha=0.4)
        ax.axhline(70, color='red', linestyle='--', linewidth=1, alpha=0.4)
        ax.axhline(250, color='darkred', linestyle=':', linewidth=1, alpha=0.3)
        ax.axhline(54, color='darkred', linestyle=':', linewidth=1, alpha=0.3)
        
        # Highlight target range (70-180 mg/dL) with green shading
        ax.fill_between(df.index, 70, 180, color='green', alpha=0.15, label='Target Range')
        
        # Calculate metrics for subplot title
        BG = df['BG'].values
        tir = ((BG >= 70) & (BG <= 180)).sum() / len(BG) * 100
        mean_bg = BG.mean()
        
        # Configure subplot appearance
        ax.set_ylabel('Glucose (mg/dL)', fontsize=10)
        ax.set_title(f'{name} | TIR: {tir:.1f}% | Mean: {mean_bg:.1f} mg/dL', 
                     fontsize=11, fontweight='bold')
        ax.grid(True, linestyle=':', alpha=0.4)
        ax.set_ylim([40, 400])  # Fixed y-axis for comparison
        
        # Only show x-axis label on bottom subplot
        if idx == n_methods - 1:
            ax.set_xlabel('Time', fontsize=10)
        else:
            ax.set_xticklabels([])

    # =========================================================================
    # VIOLIN PLOT SUBPLOT: Distribution comparison across all methods
    # Shows density of glucose values - narrower violin = more stable control
    # =========================================================================
    ax_violin = fig.add_subplot(gs[:, 1])
    
    # Collect glucose distributions for each controller
    bg_distributions = []
    labels = []
    
    for name, df in results_dict.items():
        bg_distributions.append(df['BG'].values)
        labels.append(name)
    
    # Create violin plot with mean and median markers
    parts = ax_violin.violinplot(bg_distributions, positions=range(len(labels)),
                                   showmeans=True, showmedians=True, widths=0.7)
    
    # Apply colour scheme to violin bodies
    for idx, pc in enumerate(parts['bodies']):
        method_name = labels[idx]
        pc.set_facecolor(colors.get(method_name, 'gray'))
        pc.set_alpha(0.6)
    
    # Add clinical threshold reference lines
    ax_violin.axhline(180, color='red', linestyle='--', linewidth=1, alpha=0.5, label='Hyperglycemia')
    ax_violin.axhline(70, color='red', linestyle='--', linewidth=1, alpha=0.5, label='Hypoglycemia')
    ax_violin.fill_between([-0.5, len(labels)-0.5], 70, 180, color='green', alpha=0.1)
    
    # Configure violin plot appearance
    ax_violin.set_xticks(range(len(labels)))
    ax_violin.set_xticklabels(labels, rotation=15, ha='right')
    ax_violin.set_ylabel('Glucose Distribution (mg/dL)', fontsize=11, fontweight='bold')
    ax_violin.set_title('Glucose Distribution Comparison', fontsize=12, fontweight='bold')
    ax_violin.grid(axis='y', linestyle=':', alpha=0.4)
    ax_violin.set_ylim([40, 400])
    ax_violin.legend(loc='upper right', fontsize=9)
    
    # Add overall figure title
    fig.suptitle(f'Glucose Control Comparison - Patient: {patient_id}', 
                 fontsize=14, fontweight='bold', y=0.995)
    
    # Save figure if path provided
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"[PLOT] Saved to: {save_path}")
    
    plt.show()


# =============================================================================
# FILE I/O FUNCTIONS
# =============================================================================

def save_all_results(results_dict, folder_name="simulation_results"):
    """
    Save all simulation results to CSV files and generate comparison plot.
    
    Creates the output folder if it doesn't exist, saves each controller's
    results to a separate CSV file and generates a comparison plot.
    
    Args:
        results_dict: Dictionary mapping controller names to result DataFrames
        folder_name: Output directory name (created if doesn't exist)
    """
    # Create output directory if needed
    if not os.path.exists(folder_name):
        os.makedirs(folder_name)

    # Save each controller's results to CSV
    for name, df in results_dict.items():
        safe_name = name.replace(" ", "_").lower()
        file_path = os.path.join(folder_name, f"{safe_name}.csv")
        df.to_csv(file_path)

    # Generate and save comparison plot
    plot_path = os.path.join(folder_name, "comparison_plot.png")
    plot_comparison(results_dict, save_path=plot_path)


# =============================================================================
# PIE CHART VISUALISATION
# =============================================================================

def plot_glycemic_zones_pie(results_dict, save_path=None):
    """
    Create pie charts showing glycaemic zone distribution for each controller.
    
    Displays the percentage of time spent in each of the five clinical zones:
    - Severe hypoglycaemia (<54 mg/dL)
    - Hypoglycaemia (54-70 mg/dL)
    - Target range (70-180 mg/dL)
    - Hyperglycaemia (180-250 mg/dL)
    - Severe hyperglycaemia (>250 mg/dL)
    
    Args:
        results_dict: Dictionary mapping controller names to result DataFrames
        save_path: Optional file path to save the figure
    """
    fig, axes = plt.subplots(1, len(results_dict), figsize=(5*len(results_dict), 5))
    
    # Handle single controller case (axes is not a list)
    if len(results_dict) == 1:
        axes = [axes]
    
    for ax, (name, df) in zip(axes, results_dict.items()):
        BG = df['BG'].values
        
        # Calculate percentage of time in each glycaemic zone
        severe_hypo = (BG < 54).sum() / len(BG) * 100
        hypo = ((BG >= 54) & (BG < 70)).sum() / len(BG) * 100
        tir = ((BG >= 70) & (BG <= 180)).sum() / len(BG) * 100
        hyper = ((BG > 180) & (BG <= 250)).sum() / len(BG) * 100
        severe_hyper = (BG > 250).sum() / len(BG) * 100
        
        # Configure pie chart data and appearance
        sizes = [severe_hypo, hypo, tir, hyper, severe_hyper]
        labels = [f'Severe Hypo\n(<54)', f'Hypo\n(54-70)', f'TIR\n(70-180)', 
                  f'Hyper\n(180-250)', f'Severe Hyper\n(>250)']
        colors = ['#8b0000', '#ff6347', '#90ee90', '#ffa500', '#8b0000']
        explode = (0.1, 0.05, 0, 0.05, 0.1)  # Emphasise extreme zones
        
        ax.pie(sizes, explode=explode, labels=labels, colors=colors, autopct='%1.1f%%',
               shadow=True, startangle=90)
        ax.set_title(f'{name}\nTIR: {tir:.1f}%', fontweight='bold')
    
    plt.suptitle('Glycemic Zones Distribution', fontsize=14, fontweight='bold')
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    
    plt.show()


# =============================================================================
# ADVANCED ANALYSIS FUNCTIONS
# =============================================================================


def plot_glycemic_distribution(results_dict, save_path):
    """
    Generate a violin plot to compare glucose density distributions.
    
    Violin plots show the probability density of glucose values. A narrower
    violin indicates more stable glucose control (less variability), whilst
    a wider violin suggests greater fluctuations.
    
    Args:
        results_dict: Dictionary mapping controller names to result DataFrames
        save_path: File path to save the generated figure
    """
    plt.figure(figsize=(10, 6))
    
    # Collect data and labels for plotting
    data_to_plot = []
    labels = []
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c']  # Blue, Orange, Green
    
    for name, df in results_dict.items():
        if not df.empty:
            data_to_plot.append(df['BG'].values)
            labels.append(name)
            
    if not data_to_plot:
        return

    # Create violin plot with statistical markers
    parts = plt.violinplot(data_to_plot, showmeans=True, showmedians=True)
    
    # Apply colour scheme to violin bodies
    for i, pc in enumerate(parts['bodies']):
        pc.set_facecolor(colors[i % len(colors)])
        pc.set_alpha(0.6)
    
    # Add clinical reference lines and target zone
    plt.axhline(180, color='red', linestyle='--', alpha=0.5, label='Hyper Limit')
    plt.axhline(70, color='red', linestyle='--', alpha=0.5, label='Hypo Limit')
    plt.fill_between([0.5, len(labels)+0.5], 70, 180, color='green', alpha=0.1, label='Target Range')
    # Configure plot appearance
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
    Generate a stacked bar chart showing time in each clinical glycaemic zone.
    
    Displays the five standard clinical zones as stacked bars, allowing
    easy visual comparison of glycaemic control quality across controllers.
    
    Args:
        results_dict: Dictionary mapping controller names to result DataFrames
        save_path: File path to save the generated figure
    """
    # Define the five standard clinical glycaemic zones (mg/dL)
    zones = {
        'Very Low (<54)': (-np.inf, 54),      # Severe hypoglycaemia
        'Low (54-70)': (54, 70),               # Hypoglycaemia
        'Target (70-180)': (70, 180),          # Optimal range
        'High (180-250)': (180, 250),          # Hyperglycaemia
        'Very High (>250)': (250, np.inf)      # Severe hyperglycaemia
    }
    
    # Colour scheme: red for dangerous zones, green for target, orange for high
    zone_colors = ['#8b0000', '#ff4444', '#32cd32', '#ffa500', '#8b4500']
    
    names = []
    zone_data = {k: [] for k in zones.keys()}
    
    # Calculate percentage in each zone for every controller
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

    # Create stacked bar chart
    fig, ax = plt.subplots(figsize=(10, 6))
    bottom = np.zeros(len(names))  # Track cumulative height for stacking
    
    for (zone_label, values), color in zip(zone_data.items(), zone_colors):
        ax.bar(names, values, bottom=bottom, label=zone_label, color=color, alpha=0.8, width=0.6)
        
        # Add percentage labels for zones with sufficient space (>5%)
        for i, v in enumerate(values):
            if v > 5:
                ax.text(i, bottom[i] + v/2, f"{v:.1f}%", ha='center', va='center', 
                        color='white' if color != '#32cd32' else 'black', fontweight='bold')
        bottom += values

    # Configure plot appearance
    ax.set_ylabel('Percentage of Time (%)')
    ax.set_title('Clinical Glycemic Zones Breakdown')
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

def perform_advanced_analysis(results_dict, folder):
    """
    Generate all advanced visualisation plots for comprehensive analysis.
    
    Creates violin distribution and clinical zones breakdown plots,
    saving them to the specified output folder.
    
    Args:
        results_dict: Dictionary mapping controller names to result DataFrames
        folder: Output directory path for saving generated plots
    """
    print("\n[ANALYSIS] Generating advanced plots...")
    
    # Generate violin plot showing glucose distribution density
    plot_glycemic_distribution(results_dict, os.path.join(folder, "analysis_violin.png"))
    
    # Generate stacked bar chart of clinical glycaemic zones
    plot_tir_breakdown(results_dict, os.path.join(folder, "analysis_zones.png"))
    
    print("  ✓ Violin plot saved.")
    print("  ✓ Clinical Zones plot saved.")