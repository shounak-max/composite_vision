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
from src.models.rl_attention import RecurrentAttentionModel, compute_reinforce_loss
from src.metrics.error_consistency import generate_reports_and_figures
import time

def train_rl_agent(dataset_dir, output_dir, epochs=20):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Training on {device}")
    
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    dataset = CompositeDataset(os.path.join(dataset_dir, "metadata.json"), 
                               os.path.join(dataset_dir, "images"), 
                               transform=transform, split="train")
    
    dataloader = DataLoader(dataset, batch_size=16, shuffle=True)
    
    model = RecurrentAttentionModel(num_classes=1000).to(device)
    optimizer = optim.Adam(model.parameters(), lr=3e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    
    model.train()
    import matplotlib.pyplot as plt
    batch_losses = []
    batch_rewards = []
    
    for epoch in range(epochs):
        total_loss = 0
        total_reward = 0
        
        for images, items in tqdm(dataloader, desc=f"Epoch {epoch+1}/{epochs}"):
            images = images.to(device)
            target1 = items['class1'].clone().detach().to(device)
            target2 = items['class2'].clone().detach().to(device)
            
            optimizer.zero_grad()
            logits, log_probs, values, entropies = model(images)
            
            probs = torch.softmax(logits, dim=-1)
            preds = torch.argmax(probs, dim=-1)
            
            # Reward agent if it identifies EITHER the shape (class1) or texture/occlusion (class2)
            rewards = ((preds == target1) | (preds == target2)).float().to(device)
            
            # Normalize rewards to stabilize REINFORCE variance
            rewards = (rewards - rewards.mean()) / (rewards.std() + 1e-8)
            
            rl_loss = compute_reinforce_loss(log_probs, values, rewards)
            cls_loss = torch.nn.functional.cross_entropy(logits, target1)
            
            # Maximize entropy to prevent premature mode collapse
            ENTROPY_COEF = 0.01
            entropy_loss = -ENTROPY_COEF * entropies.mean()
            
            loss = rl_loss + cls_loss + entropy_loss
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.5)
            optimizer.step()
            
            total_loss += loss.item()
            # Track raw accuracy as reward for graphing
            raw_rewards = ((preds == target1) | (preds == target2)).float().mean().item()
            total_reward += raw_rewards
            
            batch_losses.append(loss.item())
            batch_rewards.append(raw_rewards)
            
        print(f"Epoch {epoch+1} Loss: {total_loss/len(dataloader):.4f} Reward: {total_reward/len(dataloader):.4f}")
        scheduler.step()
        
    os.makedirs(output_dir, exist_ok=True)
    
    import numpy as np
    def smooth(data, window=10):
        if len(data) < window: return data
        return np.convolve(data, np.ones(window)/window, mode='valid')

    # Plotting
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(batch_losses, alpha=0.3, label='Raw Loss', color='red')
    plt.plot(smooth(batch_losses), label='Smoothed Loss', color='darkred')
    plt.xlabel('Batch')
    plt.ylabel('Loss')
    plt.title('RL Agent Training Loss')
    plt.legend()
    
    plt.subplot(1, 2, 2)
    plt.plot(batch_rewards, alpha=0.3, label='Raw Reward', color='blue')
    plt.plot(smooth(batch_rewards), label='Smoothed Reward (Accuracy)', color='darkblue')
    plt.xlabel('Batch')
    plt.ylabel('Reward')
    plt.title('RL Agent Training Reward')
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
    run_full_pipeline(force_regenerate=False)
