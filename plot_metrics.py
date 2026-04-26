import pandas as pd
import matplotlib.pyplot as plt
import os

def plot_metrics(csv_path, output_path):
    if not os.path.exists(csv_path):
        print(f"Error: Could not find {csv_path}")
        return
        
    df = pd.read_csv(csv_path)
    
    # Check if necessary columns exist
    required_cols = ['Epoch', 'Total Loss', 'Heatmap Loss', 'Regression Loss', 'Learning Rate']
    if not all(col in df.columns for col in required_cols):
        print(f"Error: Missing columns in CSV. Required: {required_cols}")
        return

    # Create a figure with 2 subplots (Loss and LR)
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
    
    # Function for exponential moving average smoothing
    def smooth(scalars, weight=0.6):
        last = scalars.iloc[0]
        smoothed = []
        for point in scalars:
            smoothed_val = last * weight + (1 - weight) * point
            smoothed.append(smoothed_val)
            last = smoothed_val
        return smoothed

    # Plot Losses
    # Raw data as light transparent lines
    ax1.plot(df['Epoch'], df['Total Loss'], color='black', alpha=0.2)
    ax1.plot(df['Epoch'], df['Heatmap Loss'], color='blue', alpha=0.2)
    ax1.plot(df['Epoch'], df['Regression Loss'], color='red', alpha=0.2)
    
    # Smoothed curves
    ax1.plot(df['Epoch'], smooth(df['Total Loss']), label='Total Loss (Smoothed)', color='black', linewidth=2.5)
    ax1.plot(df['Epoch'], smooth(df['Heatmap Loss']), label='Heatmap Loss (Smoothed)', color='blue', linewidth=2)
    ax1.plot(df['Epoch'], smooth(df['Regression Loss']), label='Regression Loss (Smoothed)', color='red', linewidth=2)
    
    ax1.set_yscale('log')  # Log scale makes the drop more visible and curve-like
    ax1.set_title('Training Losses over Epochs (Log Scale & Smoothed)')
    ax1.set_ylabel('Loss Value')
    ax1.legend()
    ax1.grid(True, linestyle=':', alpha=0.6)
    
    # Plot Learning Rate
    ax2.plot(df['Epoch'], df['Learning Rate'], label='Learning Rate', color='green', linewidth=2)
    ax2.set_title('Learning Rate over Epochs (OneCycleLR)')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('LR')
    ax2.ticklabel_format(axis='y', style='sci', scilimits=(0,0))
    ax2.legend()
    ax2.grid(True, linestyle=':', alpha=0.6)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    print(f"Successfully saved metrics visualization to {output_path}")
    
if __name__ == "__main__":
    csv_file = "output/training_metrics.csv"
    output_img = "output/training_metrics_plot.png"
    plot_metrics(csv_file, output_img)
