import os
import json
import torch
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import transforms
import pandas as pd
from tqdm import tqdm

from src.models.baselines import load_models, CompositeDataset, profile_inference_time

def unfreeze_last_block(model, model_name):
    for param in model.parameters():
        param.requires_grad = False
        
    if model_name == 'ResNet50':
        for param in model.layer4.parameters(): param.requires_grad = True
        for param in model.fc.parameters(): param.requires_grad = True
    elif model_name == 'ViT-B/16':
        for name, param in model.named_parameters():
            if 'encoder_layer_11' in name or 'heads' in name:
                param.requires_grad = True
    elif model_name == 'ConvNeXt':
        for name, param in model.named_parameters():
            if 'features.7' in name or 'classifier' in name:
                param.requires_grad = True
    return model

def finetune_model(model, model_name, dataset_dir, output_dir, epochs=10, lr=1e-4):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    model = unfreeze_last_block(model, model_name)
    
    transform_train = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    dataset_train = CompositeDataset(os.path.join(dataset_dir, "metadata.json"), 
                               os.path.join(dataset_dir, "images"), 
                               transform=transform_train, split="train")
    dataloader_train = DataLoader(dataset_train, batch_size=32, shuffle=True)
    
    optimizer = optim.Adam([p for p in model.parameters() if p.requires_grad], lr=lr)
    
    print(f"\nFine-tuning {model_name} on Composite Training Set...")
    model.train()
    
    for epoch in range(epochs):
        total_loss = 0
        total_correct = 0
        
        for images, items in tqdm(dataloader_train, desc=f"Epoch {epoch+1}/{epochs}", leave=False):
            images = images.to(device)
            target1 = items['class1'].clone().detach().to(device)
            
            optimizer.zero_grad()
            logits = model(images)
            loss = F.cross_entropy(logits, target1)
            
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            preds = torch.argmax(logits, dim=-1)
            total_correct += (preds == target1).sum().item()
            
        acc = total_correct / len(dataset_train)
        print(f"  Epoch {epoch+1} | Loss: {total_loss/len(dataloader_train):.4f} | Acc: {acc:.4f}")
        
    torch.save(model.state_dict(), os.path.join(output_dir, f"{model_name.replace('/', '_')}_FT.pth"))
    return model

def evaluate_finetuned(models_dict, dataset_dir, results_dir):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    dataset_test = CompositeDataset(os.path.join(dataset_dir, "metadata.json"), 
                               os.path.join(dataset_dir, "images"), 
                               transform=transform, split="test")
    dataloader_test = DataLoader(dataset_test, batch_size=16, shuffle=False)
    
    print("\n=== Profiling Inference Times for Fine-tuned Models ===")
    inference_times = {}
    for ft_name, model in models_dict.items():
        model.eval()
        inference_times[ft_name] = profile_inference_time(model, ft_name, device)
    print("=== Profiling Complete ===\n")
    
    results = []
    
    with torch.no_grad():
        for images, items in tqdm(dataloader_test, desc="Evaluating Fine-tuned Models"):
            images = images.to(device)
            
            for ft_name, model in models_dict.items():
                logits = model(images)
                probs = F.softmax(logits, dim=-1)
                top5_prob, top5_idx = torch.topk(probs, 5, dim=-1)
                
                for i in range(images.size(0)):
                    result = {
                        'filename': items['filename'][i],
                        'class1': items['class1'][i].item() if isinstance(items['class1'][i], torch.Tensor) else items['class1'][i],
                        'class2': items['class2'][i].item() if isinstance(items['class2'][i], torch.Tensor) else items['class2'][i],
                        'composition_type': items['composition_type'][i],
                        'salience': items['salience'][i],
                        'model': ft_name,
                        'top1_pred': top5_idx[i][0].item(),
                        'top1_conf': top5_prob[i][0].item(),
                        'top5_preds': top5_idx[i].cpu().numpy().tolist(),
                        'inference_time': inference_times[ft_name]
                    }
                    results.append(result)
                    
    csv_path = os.path.join(results_dir, "benchmark_results.csv")
    if os.path.exists(csv_path):
        df_base = pd.read_csv(csv_path)
        # Remove any existing finetuned results to avoid duplicates
        df_base = df_base[~df_base['model'].isin(models_dict.keys())]
        df_ft = pd.DataFrame(results)
        df_combined = pd.concat([df_base, df_ft], ignore_index=True)
        df_combined.to_csv(csv_path, index=False)
    else:
        pd.DataFrame(results).to_csv(csv_path, index=False)

def run_finetuning(dataset_dir, results_dir):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # Load all baselines
    all_models, _ = load_models(device)
    
    target_models = ['ResNet50', 'ViT-B/16', 'ConvNeXt']
    ft_models_dict = {}
    
    for m_name in target_models:
        if m_name in all_models:
            model = all_models[m_name]
            ft_name = f"{m_name}_FT"
            print(f"Preparing to fine-tune {m_name}...")
            ft_model = finetune_model(model, m_name, dataset_dir, results_dir, epochs=10)
            ft_models_dict[ft_name] = ft_model
            
    evaluate_finetuned(ft_models_dict, dataset_dir, results_dir)

if __name__ == '__main__':
    base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    run_finetuning(os.path.join(base_dir, "dataset"), os.path.join(base_dir, "results"))
