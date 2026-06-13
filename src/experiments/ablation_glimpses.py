import os
import torch
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from src.experiments.runner import train_rl_agent, evaluate_rl_agent

def run_glimpse_ablation(dataset_dir, results_dir, glimpse_counts=[2, 4, 6, 8, 12], epochs=15):
    """
    Sweeps over different glimpse counts for the RL Attention Agent,
    trains each model, evaluates accuracy, and records inference time
    to plot an accuracy vs compute tradeoff.
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\n=== Starting Glimpse Count Ablation on {device} ===")
    
    results = []
    
    for g in glimpse_counts:
        print(f"\n--- Training RL Agent with {g} glimpses ---")
        # We use seed=42 for all ablation runs for consistency
        model, val_acc = train_rl_agent(dataset_dir, results_dir, epochs=epochs, seed=42, num_glimpses=g)
        
        # To get the accurate inference time, we can use profile_inference_time
        from src.models.baselines import profile_inference_time
        mean_time = profile_inference_time(model, f"RL_Attention_{g}_glimpses", device)
        
        results.append({
            'glimpses': g,
            'val_accuracy': val_acc,
            'inference_time': mean_time
        })
        
    df = pd.DataFrame(results)
    ablation_csv_path = os.path.join(results_dir, "glimpse_ablation.csv")
    df.to_csv(ablation_csv_path, index=False)
    print(f"\nSaved ablation results to {ablation_csv_path}")
    
    # Generate Dual-Axis Plot
    fig, ax1 = plt.subplots(figsize=(8, 5))
    
    color1 = 'tab:blue'
    ax1.set_xlabel('Number of Glimpses')
    ax1.set_ylabel('Validation Accuracy', color=color1)
    ax1.plot(df['glimpses'], df['val_accuracy'], marker='o', color=color1, linewidth=2)
    ax1.tick_params(axis='y', labelcolor=color1)
    
    ax2 = ax1.twinx()
    color2 = 'tab:red'
    ax2.set_ylabel('Inference Time (s/image)', color=color2)
    ax2.plot(df['glimpses'], df['inference_time'], marker='s', color=color2, linewidth=2, linestyle='--')
    ax2.tick_params(axis='y', labelcolor=color2)
    
    plt.title('Tradeoff: Accuracy vs. Inference Time by Glimpse Count')
    fig.tight_layout()
    plt.savefig(os.path.join(results_dir, "glimpse_ablation.png"), dpi=300)
    plt.savefig(os.path.join(results_dir, "glimpse_ablation.pdf"))
    plt.close()
    print("Saved glimpse ablation plot.")

if __name__ == '__main__':
    base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    run_glimpse_ablation(os.path.join(base_dir, "dataset"), os.path.join(base_dir, "results"))
