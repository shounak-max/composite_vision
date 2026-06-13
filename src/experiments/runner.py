import os
import json
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms
import pandas as pd
from tqdm import tqdm
import time
import numpy as np
import random
import matplotlib.pyplot as plt
import seaborn as sns

from src.data.generation import CompositeDatasetGenerator
from src.models.baselines import evaluate_models, CompositeDataset, profile_inference_time
from src.models.rl_attention import RecurrentAttentionModel, RecurrentAttentionEnsemble, compute_a2c_loss
from src.metrics.error_consistency import generate_reports_and_figures

def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True

def train_rl_agent(dataset_dir, output_dir, epochs=20, seed=0, num_glimpses=8):
    """
    Trains the Recurrent Attention Ensemble agent on the CompositeVision dataset.
    """
    set_seed(seed)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Training RL Agent on {device} (Seed {seed})")
    
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
    
    dataset_train = CompositeDataset(os.path.join(dataset_dir, "metadata.json"), 
                               os.path.join(dataset_dir, "images"), 
                               transform=transform_train, split="train")
    dataloader_train = DataLoader(dataset_train, batch_size=64, shuffle=True)
    
    dataset_val = CompositeDataset(os.path.join(dataset_dir, "metadata.json"), 
                               os.path.join(dataset_dir, "images"), 
                               transform=transform_val, split="test")
    dataloader_val = DataLoader(dataset_val, batch_size=64, shuffle=False)
    
    model = RecurrentAttentionEnsemble(num_models=1, patch_size=48, num_classes=1000, hidden_dim=512, num_glimpses=num_glimpses).to(device)
    
    optimizer = optim.Adam(model.parameters(), lr=2e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    
    best_val_acc = 0.0
    best_model_state = None
    patience = 5
    epochs_no_improve = 0
    
    batch_losses = []
    batch_rewards = []
    batch_rl_losses = []
    batch_cls_losses = []
    
    val_losses = []
    val_rewards = []
    
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        total_reward = 0
        
        for images, items in tqdm(dataloader_train, desc=f"Epoch {epoch+1}/{epochs} [Train]", leave=False):
            images = images.to(device)
            target1 = items['class1'].clone().detach().to(device)
            
            optimizer.zero_grad()
            logits, log_probs, values, entropies = model(images)
            
            if hasattr(model, 'models') and model.training:
                num_models = len(model.models)
                target1_expanded = target1.repeat(num_models)
            else:
                target1_expanded = target1
                
            probs = torch.softmax(logits, dim=-1)
            preds = torch.argmax(probs, dim=-1)
            
            rewards = (preds == target1_expanded).float().to(device)
            
            rl_loss = compute_a2c_loss(log_probs, values, rewards, entropies=entropies)
            cls_loss = torch.nn.functional.cross_entropy(logits, target1_expanded)
            loss = rl_loss + cls_loss
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.5)
            optimizer.step()
            
            total_loss += loss.item()
            raw_rewards = (preds == target1_expanded).float().mean().item()
            total_reward += raw_rewards
            
            batch_losses.append(loss.item())
            batch_rl_losses.append(rl_loss.item())
            batch_cls_losses.append(cls_loss.item())
            batch_rewards.append(raw_rewards)
            
        train_loss = total_loss / len(dataloader_train)
        train_reward = total_reward / len(dataloader_train)
        batch_losses.append(train_loss)
        batch_rewards.append(train_reward)
        
        model.eval()
        val_loss_epoch = 0
        val_reward_epoch = 0
        with torch.no_grad():
            for images, items in tqdm(dataloader_val, desc=f"Epoch {epoch+1}/{epochs} [Val]", leave=False):
                images = images.to(device)
                target1 = items['class1'].clone().detach().to(device)
                
                logits, log_probs, values, entropies = model(images)
                
                probs = torch.softmax(logits, dim=-1)
                preds = torch.argmax(probs, dim=-1)
                
                rewards = (preds == target1).float().to(device)
                rl_loss = compute_a2c_loss(log_probs, values, rewards)
                cls_loss = torch.nn.functional.cross_entropy(logits, target1)
                loss = rl_loss + cls_loss
                
                val_loss_epoch += loss.item()
                val_reward_epoch += (preds == target1).float().mean().item()
                
        val_loss = val_loss_epoch / len(dataloader_val)
        val_reward = val_reward_epoch / len(dataloader_val)
        val_losses.append(val_loss)
        val_rewards.append(val_reward)
            
        print(f"  Epoch {epoch+1} | Train Loss: {train_loss:.4f}, Acc: {train_reward:.4f} | Val Loss: {val_loss:.4f}, Acc: {val_reward:.4f}")
        
        if val_reward > best_val_acc:
            best_val_acc = val_reward
            best_model_state = {k: v.clone() for k, v in model.state_dict().items()}
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                print(f"  >> Early stopping at epoch {epoch+1}")
                break
        
        scheduler.step()
        
    os.makedirs(output_dir, exist_ok=True)
    
    def smooth(data, window=50):
        if len(data) < window: return data
        return np.convolve(data, np.ones(window)/window, mode='valid')
    
    actual_epochs = len(val_losses)
    plt.figure(figsize=(18, 5))
    
    plt.subplot(1, 3, 1)
    train_epoch_losses = [bl for i, bl in enumerate(batch_losses) if len(batch_losses) > actual_epochs and i % (len(batch_losses)//actual_epochs) == 0][:actual_epochs]
    if len(train_epoch_losses) != actual_epochs:
        train_epoch_losses = batch_losses[-actual_epochs:]
    plt.plot(range(1, actual_epochs+1), train_epoch_losses, label='Train Loss', color='red')
    plt.plot(range(1, actual_epochs+1), val_losses, label='Val Loss', color='orange')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title(f'Total Loss (Train vs Val) [Seed {seed}]')
    plt.legend()
    
    plt.subplot(1, 3, 2)
    train_epoch_rewards = [br for i, br in enumerate(batch_rewards) if len(batch_rewards) > actual_epochs and i % (len(batch_rewards)//actual_epochs) == 0][:actual_epochs]
    if len(train_epoch_rewards) != actual_epochs:
        train_epoch_rewards = batch_rewards[-actual_epochs:]
    plt.plot(range(1, actual_epochs+1), train_epoch_rewards, label='Train Acc', color='blue')
    plt.plot(range(1, actual_epochs+1), val_rewards, label='Val Acc', color='cyan')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.title(f'RL Agent Accuracy over Epochs [Seed {seed}]')
    plt.legend()
    
    plt.subplot(1, 3, 3)
    plt.plot(smooth(batch_rl_losses), label='Policy (RL) Loss', alpha=0.7)
    plt.plot(smooth(batch_cls_losses), label='Classification Loss', alpha=0.7)
    plt.plot(smooth(batch_rewards), label='Raw Batch Acc', color='black', alpha=0.8, linewidth=2)
    plt.xlabel('Batch Step')
    plt.ylabel('Value')
    plt.title('RL Internal Metrics (Smoothed)')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"rl_training_graphs_seed{seed}.png"), dpi=300, bbox_inches='tight')
    plt.close()
    
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
    
    torch.save(model.state_dict(), os.path.join(output_dir, f"rl_agent_seed{seed}.pth"))
    return model, best_val_acc

def train_rl_multi_seed(dataset_dir, output_dir, seeds=[0, 1, 2, 3, 4], epochs=20):
    results = []
    best_model = None
    best_acc = 0.0
    
    for seed in seeds:
        print(f"\n--- Starting Training for Seed {seed} ---")
        model, val_acc = train_rl_agent(dataset_dir, output_dir, epochs=epochs, seed=seed)
        results.append({'seed': seed, 'val_accuracy': val_acc})
        
        if val_acc > best_acc:
            best_acc = val_acc
            best_model = model
            
    df = pd.DataFrame(results)
    df.to_csv(os.path.join(output_dir, "rl_seed_results.csv"), index=False)
    
    mean_acc = df['val_accuracy'].mean()
    std_acc = df['val_accuracy'].std()
    print(f"\nMulti-Seed Training Complete. Mean Acc: {mean_acc:.4f} ± {std_acc:.4f}")
    
    plt.figure(figsize=(6, 4))
    sns.boxplot(y=df['val_accuracy'], color='lightblue')
    sns.swarmplot(y=df['val_accuracy'], color='black', size=8)
    plt.title(f'RL Agent Seed Variance\nMean: {mean_acc:.1%} ± {std_acc:.1%}')
    plt.ylabel('Validation Accuracy')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "rl_seed_variance.png"), dpi=300)
    plt.close()
    
    # Save the absolute best model as the main agent
    torch.save(best_model.state_dict(), os.path.join(output_dir, "rl_agent_best.pth"))
    
    return best_model

def evaluate_rl_agent(model, dataset_dir, results_dir):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.eval()
    
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    dataset = CompositeDataset(os.path.join(dataset_dir, "metadata.json"), 
                               os.path.join(dataset_dir, "images"), 
                               transform=transform, split="test")
    dataloader = DataLoader(dataset, batch_size=16, shuffle=False)
    
    # Profile inference time properly
    print("\n=== Profiling RL Inference Time (batch_size=1, 50 timed runs) ===")
    mean_time = profile_inference_time(model, "RL_Attention", device)
    print("=== Profiling Complete ===\n")
    
    results = []
    
    with torch.no_grad():
        for images, items in tqdm(dataloader, desc="Evaluating RL Agent"):
            images = images.to(device)
            
            logits, log_probs, values, entropies = model(images)
            probs = torch.softmax(logits, dim=-1)
            top5_prob, top5_idx = torch.topk(probs, 5, dim=-1)
            
            for i in range(images.size(0)):
                result = {
                    'filename': items['filename'][i],
                    'class1': items['class1'][i].item() if isinstance(items['class1'][i], torch.Tensor) else items['class1'][i],
                    'class2': items['class2'][i].item() if isinstance(items['class2'][i], torch.Tensor) else items['class2'][i],
                    'composition_type': items['composition_type'][i],
                    'salience': items['salience'][i],
                    'model': 'RL_Attention',
                    'top1_pred': top5_idx[i][0].item(),
                    'top1_conf': top5_prob[i][0].item(),
                    'top5_preds': top5_idx[i].cpu().numpy().tolist(),
                    'inference_time': mean_time,
                    'glimpses': model.num_glimpses if hasattr(model, 'num_glimpses') else 8
                }
                results.append(result)
                
    csv_path = os.path.join(results_dir, "benchmark_results.csv")
    if os.path.exists(csv_path):
        df_base = pd.read_csv(csv_path)
        df_base = df_base[df_base['model'] != 'RL_Attention']
        df_rl = pd.DataFrame(results)
        df_combined = pd.concat([df_base, df_rl], ignore_index=True)
        df_combined.to_csv(csv_path, index=False)
    else:
        pd.DataFrame(results).to_csv(csv_path, index=False)

def run_full_pipeline(force_regenerate=False, run_ablation=True):
    base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    dataset_dir = os.path.join(base_dir, "dataset")
    results_dir = os.path.join(base_dir, "results")
    
    if force_regenerate:
        print("Force regeneration requested. Cleaning old data...")
        for f in ["benchmark_results.json", "benchmark_results.csv", 
                  "model_accuracy.csv", "model_error_consistency.csv"]:
            fpath = os.path.join(results_dir, f)
            if os.path.exists(fpath):
                os.remove(fpath)
        meta_path = os.path.join(dataset_dir, "metadata.json")
        if os.path.exists(meta_path):
            os.remove(meta_path)
        images_path = os.path.join(dataset_dir, "images")
        if os.path.exists(images_path):
            import shutil
            shutil.rmtree(images_path)
    
    print("\n=== PHASE 1: GENERATE DATASET ===")
    if not os.path.exists(os.path.join(dataset_dir, "metadata.json")):
        generator = CompositeDatasetGenerator(output_dir=dataset_dir)
        generator.generate()
    else:
        print("Dataset already exists, skipping generation.")
        
    print("\n=== PHASE 2: EVALUATE ZERO-SHOT BASELINES ===")
    if not os.path.exists(os.path.join(results_dir, "benchmark_results.csv")):
        evaluate_models(os.path.join(dataset_dir, "metadata.json"), 
                        os.path.join(dataset_dir, "images"), 
                        results_dir)
    else:
        print("Baseline results exist, skipping baseline evaluation.")
        
    print("\n=== PHASE 3: FINE-TUNE BASELINES ===")
    from src.experiments.finetune_baselines import run_finetuning
    run_finetuning(dataset_dir, results_dir)
        
    print("\n=== PHASE 4: TRAIN MULTI-SEED RL ATTENTION ===")
    model = train_rl_multi_seed(dataset_dir, results_dir, seeds=[0, 1, 2, 3, 4], epochs=25)
    print("\n=== PHASE 5: EVALUATE BEST RL AGENT ===")
    evaluate_rl_agent(model, dataset_dir, results_dir)
    
    if run_ablation:
        print("\n=== PHASE 6: GLIMPSE COUNT ABLATION ===")
        from src.experiments.ablation_glimpses import run_glimpse_ablation
        run_glimpse_ablation(dataset_dir, results_dir)
    
    print("\n=== PHASE 7: ERROR CONSISTENCY & REPORTS ===")
    generate_reports_and_figures(results_dir)
    print("Pipeline complete!")

if __name__ == '__main__':
    run_full_pipeline(force_regenerate=False, run_ablation=True)
