import matplotlib.pyplot as plt
import pandas as pd
import os

def print_glycemic_metrics(df, controller_name):
    """
    Calculates and prints TIR, TBR, TAR, and Mean BG.
    """
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
    Plots results. If save_path is provided, saves the image instead of just showing it.
    """
    n_plots = len(results_dict)
    if n_plots == 0:
        print("No data to plot.")
        return

    fig, axes = plt.subplots(n_plots, 1, figsize=(12, 5 * n_plots), sharex=True)
    if n_plots == 1: axes = [axes]
    
    for ax, (name, df) in zip(axes, results_dict.items()):
        ax.plot(df.index, df['BG'], label=f'{name} Trace', linewidth=2)
        ax.axhline(180, color='red', linestyle='--', alpha=0.5)
        ax.axhline(70, color='red', linestyle='--', alpha=0.5)
        ax.fill_between(df.index, 70, 180, color='green', alpha=0.1)
        ax.set_ylabel('BG (mg/dL)')
        ax.set_title(f'{name} Control ({patient_id})')
        ax.grid(True, linestyle=':', alpha=0.6)
        ax.legend(loc='upper right')

    axes[-1].set_xlabel('Time')
    plt.tight_layout()

    # SAVE LOGIC
    if save_path:
        plt.savefig(save_path, dpi=300) # dpi=300 makes it high quality
        print(f"Plot saved to: {save_path}")
    
    plt.show()


def save_all_results(results_dict, folder_name="simulation_results"):
    """
    Creates a folder, saves all DataFrames as CSVs, and saves the Plot.
    """
    # 1. Create the folder if it doesn't exist
    if not os.path.exists(folder_name):
        os.makedirs(folder_name)
        print(f"Created directory: {folder_name}")

    # 2. Save each DataFrame as a CSV file
    for name, df in results_dict.items():
        # Clean the name to be filename-safe (remove spaces)
        safe_name = name.replace(" ", "_").lower()
        file_path = os.path.join(folder_name, f"{safe_name}.csv")
        df.to_csv(file_path)
        print(f"Saved data: {file_path}")

    # 3. Save the combined plot in the same folder
    plot_path = os.path.join(folder_name, "comparison_plot.png")
    plot_comparison(results_dict, save_path=plot_path)