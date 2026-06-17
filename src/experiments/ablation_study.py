"""
RL Attention Ablation Study for CompositeVision.

Runs controlled ablations to isolate the contribution of each component
in the Recurrent Attention Model:

1. Glimpse count scaling: 1, 2, 4, 8 (default), 16 glimpses
2. Random crop baseline: 8 uniformly random locations (no learned policy)
3. (Optional) Backbone swap: ViT as glimpse feature extractor

The random crop baseline is the most important control — it determines
whether the learned RL policy contributes beyond simple multi-crop ensembling.
"""

import os
import sys
import json
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms
import pandas as pd
import numpy as np
from tqdm import tqdm
import time
import copy

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from src.models.baselines import CompositeDataset
from src.models.rl_attention import (
    RecurrentAttentionModel, RecurrentAttentionEnsemble, compute_a2c_loss
)


class RandomCropAttentionModel(RecurrentAttentionModel):
    """
    Control baseline: replaces the learned location policy with 
    uniformly random fixation points.
    
    Everything else is identical to RecurrentAttentionModel:
    - Same ResNet50 backbone for glimpse extraction
    - Same multi-scale glimpse extraction
    - Same LSTM state aggregation
    - Same classifier
    
    If this model matches the learned-policy model, the RL contribution
    is negligible and the benefit comes from multi-crop ensembling alone.
    """
    
    def forward(self, images):
        B = images.size(0)
        device = images.device
        hx = torch.zeros(B, self.rnn.hidden_size, device=device)
        cx = torch.zeros(B, self.rnn.hidden_size, device=device)
        
        log_probs = []
        entropies = []
        values = []
        logits_list = []
        
        for i in range(self.num_glimpses):
            # CRITICAL DIFFERENCE: Random locations instead of learned policy
            locs = torch.rand(B, 2, device=device) * 2 - 1  # Uniform(-1, 1)
            
            g_t_concat, g_t_avg = self.glimpse_net(images, locs)
            hx, cx = self.rnn(g_t_concat.detach(), (hx, cx))
            
            # Dummy RL stats (no policy to optimize)
            log_probs.append(torch.zeros(B, device=device))
            entropies.append(torch.zeros(B, device=device))
            values.append(self.baseline(hx).squeeze(-1))
            logits_list.append(self.classifier(g_t_avg))
        
        # Average foveated logits over all glimpse time steps
        foveated_logits = torch.stack(logits_list, dim=1).mean(dim=1)
        
        # Global pathway (identical to parent)
        with torch.no_grad():
            global_features = self.global_extractor(images)
            global_logits = self.global_classifier(global_features)
        
        logits = 0.5 * global_logits + 0.5 * foveated_logits
        
        return logits, torch.stack(log_probs, dim=1), torch.stack(values, dim=1), torch.stack(entropies, dim=1)


def train_ablation_variant(dataset_dir, output_dir, variant_name, num_glimpses=8,
                           use_random_crop=False, epochs=20):
    """
    Train a single ablation variant.
    
    Args:
        dataset_dir: path to dataset with metadata.json and images/
        output_dir: where to save model checkpoint and results
        variant_name: identifier string (e.g., 'glimpse_4', 'random_crop_8')
        num_glimpses: number of glimpses for this variant
        use_random_crop: if True, uses RandomCropAttentionModel
        epochs: training epochs
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\n{'='*60}")
    print(f"Training ablation: {variant_name}")
    print(f"  Glimpses: {num_glimpses}, Random crop: {use_random_crop}")
    print(f"  Device: {device}")
    print(f"{'='*60}")
    
    transform_train = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    transform_val = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    dataset_train = CompositeDataset(
        os.path.join(dataset_dir, "metadata.json"),
        os.path.join(dataset_dir, "images"),
        transform=transform_train, split="train"
    )
    dataloader_train = DataLoader(dataset_train, batch_size=64, shuffle=True)
    
    dataset_val = CompositeDataset(
        os.path.join(dataset_dir, "metadata.json"),
        os.path.join(dataset_dir, "images"),
        transform=transform_val, split="test"
    )
    dataloader_val = DataLoader(dataset_val, batch_size=64, shuffle=False)
    
    # Create model based on variant type
    ModelClass = RandomCropAttentionModel if use_random_crop else RecurrentAttentionModel
    model = ModelClass(
        patch_size=48, num_classes=1000, hidden_dim=512, num_glimpses=num_glimpses
    ).to(device)
    
    optimizer = optim.Adam(model.parameters(), lr=2e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    
    best_val_acc = 0.0
    best_model_state = None
    patience = 5
    epochs_no_improve = 0
    
    training_log = []
    
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        total_reward = 0
        
        for images, items in tqdm(dataloader_train, desc=f"[{variant_name}] Epoch {epoch+1}/{epochs} Train"):
            images = images.to(device)
            target1 = items['class1'].clone().detach().to(device)
            
            optimizer.zero_grad()
            logits, log_probs, values, entropies = model(images)
            
            probs = torch.softmax(logits, dim=-1)
            preds = torch.argmax(probs, dim=-1)
            rewards = (preds == target1).float().to(device)
            
            if use_random_crop:
                # Random crop model: only classification loss (no RL policy to train)
                loss = torch.nn.functional.cross_entropy(logits, target1)
            else:
                rl_loss = compute_a2c_loss(log_probs, values, rewards, entropies=entropies)
                cls_loss = torch.nn.functional.cross_entropy(logits, target1)
                loss = rl_loss + cls_loss
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.5)
            optimizer.step()
            
            total_loss += loss.item()
            total_reward += (preds == target1).float().mean().item()
        
        train_loss = total_loss / len(dataloader_train)
        train_acc = total_reward / len(dataloader_train)
        
        # Validation
        model.eval()
        val_correct = 0
        val_total = 0
        with torch.no_grad():
            for images, items in tqdm(dataloader_val, desc=f"[{variant_name}] Epoch {epoch+1}/{epochs} Val"):
                images = images.to(device)
                target1 = items['class1'].clone().detach().to(device)
                logits, _, _, _ = model(images)
                preds = torch.argmax(torch.softmax(logits, dim=-1), dim=-1)
                val_correct += (preds == target1).sum().item()
                val_total += target1.size(0)
        
        val_acc = val_correct / val_total
        
        training_log.append({
            'epoch': epoch + 1,
            'train_loss': train_loss,
            'train_acc': train_acc,
            'val_acc': val_acc,
        })
        
        print(f"  Epoch {epoch+1}: Train Acc={train_acc:.4f}, Val Acc={val_acc:.4f}")
        
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_model_state = {k: v.clone() for k, v in model.state_dict().items()}
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                print(f"  Early stopping at epoch {epoch+1}")
                break
        
        scheduler.step()
    
    # Save best model
    variant_dir = os.path.join(output_dir, variant_name)
    os.makedirs(variant_dir, exist_ok=True)
    
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
    
    torch.save(model.state_dict(), os.path.join(variant_dir, "model.pth"))
    
    with open(os.path.join(variant_dir, "training_log.json"), 'w') as f:
        json.dump(training_log, f, indent=2)
    
    print(f"  Best val acc for {variant_name}: {best_val_acc:.4f}")
    return model, best_val_acc


def evaluate_ablation_variant(model, variant_name, dataset_dir, output_dir):
    """Evaluate a trained ablation variant and save per-image predictions."""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.eval()
    
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    dataset = CompositeDataset(
        os.path.join(dataset_dir, "metadata.json"),
        os.path.join(dataset_dir, "images"),
        transform=transform, split="test"
    )
    dataloader = DataLoader(dataset, batch_size=64, shuffle=False)
    
    results = []
    with torch.no_grad():
        for images, items in tqdm(dataloader, desc=f"Evaluating {variant_name}"):
            images = images.to(device)
            
            if device.type == 'cuda':
                torch.cuda.synchronize()
            batch_start = time.time()
            
            logits, _, _, _ = model(images)
            probs = torch.softmax(logits, dim=-1)
            
            if device.type == 'cuda':
                torch.cuda.synchronize()
            per_image_time = (time.time() - batch_start) / images.size(0)
            
            top5_prob, top5_idx = torch.topk(probs, 5, dim=-1)
            
            for i in range(images.size(0)):
                results.append({
                    'filename': items['filename'][i],
                    'class1': items['class1'][i].item() if isinstance(items['class1'][i], torch.Tensor) else items['class1'][i],
                    'class2': items['class2'][i].item() if isinstance(items['class2'][i], torch.Tensor) else items['class2'][i],
                    'composition_type': items['composition_type'][i],
                    'salience': items['salience'][i],
                    'model': variant_name,
                    'top1_pred': top5_idx[i][0].item(),
                    'top1_conf': top5_prob[i][0].item(),
                    'top5_preds': top5_idx[i].cpu().numpy().tolist(),
                    'inference_time': per_image_time,
                })
    
    variant_dir = os.path.join(output_dir, variant_name)
    os.makedirs(variant_dir, exist_ok=True)
    df = pd.DataFrame(results)
    df.to_csv(os.path.join(variant_dir, "predictions.csv"), index=False)
    
    accuracy = (df['top1_pred'] == df['class1']).mean()
    print(f"  {variant_name} test accuracy: {accuracy*100:.2f}%")
    
    return df, accuracy


def run_ablation_study(dataset_dir, output_dir, epochs=20):
    """
    Run the complete ablation study.
    
    Variants:
    1. glimpse_1:     1 glimpse  (lower bound)
    2. glimpse_2:     2 glimpses (scaling curve)
    3. glimpse_4:     4 glimpses (scaling curve)
    4. glimpse_8:     8 glimpses (default, reference)
    5. glimpse_16:    16 glimpses (diminishing returns)
    6. random_crop_8: 8 random locations (isolates policy contribution)
    """
    variants = [
        {'name': 'glimpse_1',     'num_glimpses': 1,  'random': False},
        {'name': 'glimpse_2',     'num_glimpses': 2,  'random': False},
        {'name': 'glimpse_4',     'num_glimpses': 4,  'random': False},
        {'name': 'glimpse_8',     'num_glimpses': 8,  'random': False},
        {'name': 'glimpse_16',    'num_glimpses': 16, 'random': False},
        {'name': 'random_crop_8', 'num_glimpses': 8,  'random': True},
    ]
    
    os.makedirs(output_dir, exist_ok=True)
    summary = []
    all_predictions = []
    
    for variant in variants:
        model, best_val_acc = train_ablation_variant(
            dataset_dir, output_dir,
            variant_name=variant['name'],
            num_glimpses=variant['num_glimpses'],
            use_random_crop=variant['random'],
            epochs=epochs,
        )
        
        df_preds, test_acc = evaluate_ablation_variant(
            model, variant['name'], dataset_dir, output_dir
        )
        
        summary.append({
            'variant': variant['name'],
            'num_glimpses': variant['num_glimpses'],
            'random_crop': variant['random'],
            'best_val_acc': round(best_val_acc * 100, 2),
            'test_acc': round(test_acc * 100, 2),
        })
        all_predictions.append(df_preds)
        
        # Free GPU memory
        del model
        torch.cuda.empty_cache() if torch.cuda.is_available() else None
    
    # Save summary
    summary_df = pd.DataFrame(summary)
    summary_df.to_csv(os.path.join(output_dir, "ablation_summary.csv"), index=False)
    
    print("\n" + "="*60)
    print("ABLATION STUDY SUMMARY")
    print("="*60)
    print(summary_df.to_string(index=False))
    
    # Run statistical tests between key pairs
    from src.metrics.statistical_tests import mcnemar_test, bootstrap_accuracy_ci, cohens_h
    
    # Combine all predictions for statistical testing
    combined_df = pd.concat(all_predictions, ignore_index=True)
    
    # Key comparison: learned policy vs random crop (both 8 glimpses)
    df_learned = combined_df[combined_df['model'] == 'glimpse_8'].sort_values('filename')
    df_random = combined_df[combined_df['model'] == 'random_crop_8'].sort_values('filename')
    
    if len(df_learned) > 0 and len(df_random) > 0:
        mcnemar = mcnemar_test(
            df_learned['top1_pred'].values,
            df_random['top1_pred'].values,
            df_learned['class1'].values
        )
        
        acc_learned = bootstrap_accuracy_ci(df_learned['top1_pred'].values, df_learned['class1'].values)
        acc_random = bootstrap_accuracy_ci(df_random['top1_pred'].values, df_random['class1'].values)
        
        print(f"\n--- Critical Comparison: Learned Policy vs Random Crop (8 glimpses) ---")
        print(f"  Learned policy: {acc_learned[0]*100:.2f}% [{acc_learned[1]*100:.2f}%, {acc_learned[2]*100:.2f}%]")
        print(f"  Random crop:    {acc_random[0]*100:.2f}% [{acc_random[1]*100:.2f}%, {acc_random[2]*100:.2f}%]")
        print(f"  McNemar's chi2: {mcnemar['chi2']}, p={mcnemar['p_value']:.6f}")
        print(f"  Cohen's h: {cohens_h(acc_learned[0], acc_random[0])}")
        print(f"  Significant: {mcnemar['significant']}")
    
    # Generate ablation figure
    _plot_ablation_results(summary, output_dir)
    
    return summary_df


def _plot_ablation_results(summary, output_dir):
    """Generate the glimpse scaling curve figure."""
    import matplotlib.pyplot as plt
    
    # Filter to learned-policy variants only for the scaling curve
    learned = [s for s in summary if not s['random_crop']]
    random = [s for s in summary if s['random_crop']]
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Scaling curve
    glimpses = [s['num_glimpses'] for s in learned]
    accs = [s['test_acc'] for s in learned]
    ax.plot(glimpses, accs, 'o-', color='#2196F3', linewidth=2, markersize=8, label='Learned RL Policy')
    
    # Random crop baseline (horizontal line)
    if random:
        random_acc = random[0]['test_acc']
        ax.axhline(y=random_acc, color='#F44336', linestyle='--', linewidth=2, 
                    label=f'Random Crop (8 glimpses): {random_acc:.1f}%')
    
    ax.set_xlabel('Number of Glimpses', fontsize=13)
    ax.set_ylabel('Test Accuracy (%)', fontsize=13)
    ax.set_title('RL Attention Ablation: Glimpse Count Scaling Curve', fontsize=14)
    ax.set_xticks(glimpses)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    
    # Annotate points
    for g, a in zip(glimpses, accs):
        ax.annotate(f'{a:.1f}%', (g, a), textcoords="offset points", 
                    xytext=(0, 12), ha='center', fontsize=10)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "ablation_glimpse_scaling.png"), dpi=300, bbox_inches='tight')
    plt.savefig(os.path.join(output_dir, "ablation_glimpse_scaling.pdf"), bbox_inches='tight')
    plt.close()
    print(f"Saved ablation figure to {output_dir}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Run RL Attention ablation study")
    parser.add_argument("--dataset-dir", type=str, default=None)
    parser.add_argument("--output-dir", type=str, default=None)
    parser.add_argument("--epochs", type=int, default=20)
    args = parser.parse_args()
    
    base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    
    if args.dataset_dir is None:
        args.dataset_dir = os.path.join(base_dir, "dataset")
    if args.output_dir is None:
        args.output_dir = os.path.join(base_dir, "results", "ablations")
    
    run_ablation_study(args.dataset_dir, args.output_dir, args.epochs)
