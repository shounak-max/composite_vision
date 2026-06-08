import os
import json
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms
import pandas as pd
from tqdm import tqdm
from src.data.generation import CompositeDatasetGenerator
from src.models.baselines import evaluate_models, CompositeDataset
from src.models.rl_attention import RecurrentAttentionModel, RecurrentAttentionEnsemble, compute_reinforce_loss
from src.metrics.error_consistency import generate_reports_and_figures
import time

def train_rl_agent(dataset_dir, output_dir, epochs=20):
    """
    Trains the Recurrent Attention Ensemble agent on the CompositeVision dataset.
    
    CONCEPT: 
    The agent is rewarded (+1) only if its final classification matches the PRIMARY 
    shape class ('class1') of the composite image. This forces the agent's location 
    policy to learn to seek out shape-defining features rather than being distracted 
    by conflicting textures or colors.
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Training on {device}")
    
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    dataset_train = CompositeDataset(os.path.join(dataset_dir, "metadata.json"), 
                               os.path.join(dataset_dir, "images"), 
                               transform=transform, split="train")
    dataloader_train = DataLoader(dataset_train, batch_size=16, shuffle=True)
    
    dataset_val = CompositeDataset(os.path.join(dataset_dir, "metadata.json"), 
                               os.path.join(dataset_dir, "images"), 
                               transform=transform, split="test")
    dataloader_val = DataLoader(dataset_val, batch_size=16, shuffle=False)
    
    # Fine-tuned hyperparameters: larger patches (48), wider LSTM (512), more glimpses (8)
    model = RecurrentAttentionEnsemble(num_models=3, patch_size=48, num_classes=1000, hidden_dim=512, num_glimpses=8).to(device)
    optimizer = optim.Adam(model.parameters(), lr=2e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    
    model.train()
    import matplotlib.pyplot as plt
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
        
        for images, items in tqdm(dataloader_train, desc=f"Epoch {epoch+1}/{epochs} [Train]"):
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
            
            # Reward agent ONLY for identifying the primary shape class (class1)
            # This is the core mechanism that instills a shape-bias in the agent.
            rewards = (preds == target1_expanded).float().to(device)
            
            # RL Policy Loss + Value Loss + Entropy Bonus
            rl_loss = compute_reinforce_loss(log_probs, values, rewards, entropies=entropies)
            # Standard Classification Loss (Cross Entropy)
            cls_loss = torch.nn.functional.cross_entropy(logits, target1_expanded)
            
            # The total loss jointly optimizes the feature extractor/classifier (via cls_loss) 
            # and the location policy (via rl_loss)
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
        
        # Validation Phase
        model.eval()
        val_loss_epoch = 0
        val_reward_epoch = 0
        with torch.no_grad():
            for images, items in tqdm(dataloader_val, desc=f"Epoch {epoch+1}/{epochs} [Val]"):
                images = images.to(device)
                target1 = items['class1'].clone().detach().to(device)
                
                logits, log_probs, values, entropies = model(images)
                
                probs = torch.softmax(logits, dim=-1)
                preds = torch.argmax(probs, dim=-1)
                
                rewards = (preds == target1).float().to(device)
                
                # CONCEPT: During validation, the agent acts deterministically (log_probs=0).
                # The RL policy loss is therefore 0. The returned rl_loss here only reflects 
                # the Value Network's mean squared error (MSE) against the ensemble reward,
                # which acts as a helpful metric for tracking baseline accuracy over time.
                rl_loss = compute_reinforce_loss(log_probs, values, rewards)
                cls_loss = torch.nn.functional.cross_entropy(logits, target1)
                
                loss = rl_loss + cls_loss
                
                val_loss_epoch += loss.item()
                val_reward_epoch += (preds == target1).float().mean().item()
                
        val_loss = val_loss_epoch / len(dataloader_val)
        val_reward = val_reward_epoch / len(dataloader_val)
        val_losses.append(val_loss)
        val_rewards.append(val_reward)
            
        print(f"Epoch {epoch+1} | Train Loss: {train_loss:.4f}, Acc: {train_reward:.4f} | Val Loss: {val_loss:.4f}, Acc: {val_reward:.4f}")
        scheduler.step()
        
    os.makedirs(output_dir, exist_ok=True)
    
    import numpy as np
    def smooth(data, window=50):
        if len(data) < window: return data
        return np.convolve(data, np.ones(window)/window, mode='valid')
    
    # Plotting 3-panel RL Dashboard
    plt.figure(figsize=(18, 5))
    
    plt.subplot(1, 3, 1)
    plt.plot(range(1, epochs+1), [bl for i, bl in enumerate(batch_losses) if i % (len(batch_losses)//epochs) == 0][:epochs], label='Train Loss', color='red')
    plt.plot(range(1, epochs+1), val_losses, label='Val Loss', color='orange')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Total Loss (Train vs Val)')
    plt.legend()
    
    plt.subplot(1, 3, 2)
    plt.plot(range(1, epochs+1), [br for i, br in enumerate(batch_rewards) if i % (len(batch_rewards)//epochs) == 0][:epochs], label='Train Reward (Acc)', color='blue')
    plt.plot(range(1, epochs+1), val_rewards, label='Val Reward (Acc)', color='cyan')
    plt.xlabel('Epoch')
    plt.ylabel('Reward / Accuracy')
    plt.title('RL Agent Reward over Epochs')
    plt.legend()
    
    plt.subplot(1, 3, 3)
    plt.plot(smooth(batch_rl_losses), label='Policy (RL) Loss', alpha=0.7)
    plt.plot(smooth(batch_cls_losses), label='Classification Loss', alpha=0.7)
    plt.plot(smooth(batch_rewards), label='Raw Batch Reward', color='black', alpha=0.8, linewidth=2)
    plt.xlabel('Batch Step')
    plt.ylabel('Value')
    plt.title('RL Internal Metrics (Smoothed)')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "rl_training_graphs.png"))
    plt.close()
    torch.save(model.state_dict(), os.path.join(output_dir, "rl_agent.pth"))
    return model

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
    
    results = []
    
    start_time = time.time()
    
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
                    'inference_time': (time.time() - start_time) / len(dataset),
                    'glimpses': model.num_glimpses
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

def run_full_pipeline(force_regenerate=False):
    """
    Executes the entire CompositeVision research pipeline end-to-end.
    
    Phases:
    1. Dataset Generation: Downloads true ImageNet data, creates cognitive conflict composites.
    2. Baselines: Evaluates standard CNNs/ViTs/CLIP to establish baseline shape/texture biases.
    3. RL Training: Trains the Recurrent Attention Ensemble to develop a human-like shape bias.
    4. RL Evaluation: Tests the RL agent on the same composites.
    5. Analysis: Generates error consistency matrices and comprehensive scientific reports.
    """
    base_dir = "d:/gitfork/composite_vision_research"
    dataset_dir = os.path.join(base_dir, "dataset")
    results_dir = os.path.join(base_dir, "results")
    
    # Clean old results if forcing regeneration
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
    
    print("=== PHASE 1: GENERATE DATASET (Real ImageNet via Imagenette) ===")
    if not os.path.exists(os.path.join(dataset_dir, "metadata.json")):
        generator = CompositeDatasetGenerator(output_dir=dataset_dir)
        generator.generate()
    else:
        print("Dataset already exists, skipping generation.")
        
    print("=== PHASE 2: EVALUATE BASELINES ===")
    if not os.path.exists(os.path.join(results_dir, "benchmark_results.json")):
        evaluate_models(os.path.join(dataset_dir, "metadata.json"), 
                        os.path.join(dataset_dir, "images"), 
                        results_dir)
    else:
        print("Baseline results exist, skipping baseline evaluation.")
        
    print("=== PHASE 3 & 4: TRAIN AND EVALUATE RL ATTENTION ===")
    model = train_rl_agent(dataset_dir, results_dir, epochs=50)
    evaluate_rl_agent(model, dataset_dir, results_dir)
    
    print("=== PHASE 5: ERROR CONSISTENCY & REPORTS ===")
    generate_reports_and_figures(results_dir)
    print("Pipeline complete!")

if __name__ == '__main__':
    run_full_pipeline(force_regenerate=True)
