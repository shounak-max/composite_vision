"""
Zero-Shot RL Agent Training.

Trains the Recurrent Attention Ensemble (RL Agent) purely on the raw, 
un-composited Imagenette dataset. The agent is exposed ONLY to natural 
images and must learn robust shape representations that generalize 
zero-shot to the composite benchmark.

Aggressive data augmentation (ColorJitter, RandomErasing, RandomAffine) 
is used to simulate compositional noise (color inversion, occlusion, patch shuffle) 
so the learned policy becomes robust.
"""

import os
import sys
import torch
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms, datasets
from tqdm import tqdm
from PIL import Image
import json
import matplotlib.pyplot as plt

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from src.models.rl_attention import RecurrentAttentionEnsemble, compute_a2c_loss
from src.data.generation import IMAGENETTE_CLASS_MAP

class MappedImagenetteDataset(Dataset):
    """Wraps torchvision ImageFolder to map classes to real ImageNet indices."""
    def __init__(self, root, transform=None):
        self.image_folder = datasets.ImageFolder(root, transform=transform)
        # torchvision assigns classes 0..9 alphabetically. We need real ImageNet indices.
        # image_folder.classes contains the original folder names (e.g., 'n01440764')
        self.class_to_idx = {i: IMAGENETTE_CLASS_MAP[c] for i, c in enumerate(self.image_folder.classes)}
        
    def __len__(self):
        return len(self.image_folder)
        
    def __getitem__(self, idx):
        img, fake_label = self.image_folder[idx]
        real_label = self.class_to_idx[fake_label]
        return img, real_label

def train_rl_on_raw_imagenette(dataset_dir, output_dir, epochs=20):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\n{'='*60}")
    print(f"Training RL Agent on RAW Imagenette (Zero-Shot Eval Mode)")
    print(f"Device: {device}")
    print(f"{'='*60}")
    
    raw_train_dir = os.path.join(dataset_dir, "imagenette2-160", "train")
    raw_val_dir = os.path.join(dataset_dir, "imagenette2-160", "val")
    
    if not os.path.exists(raw_train_dir):
        raise FileNotFoundError(f"Raw Imagenette data not found at {raw_train_dir}. Please run generation.py first.")
    
    # Aggressive augmentation to build robustness without seeing actual composites
    transform_train = transforms.Compose([
        transforms.Resize(256),
        transforms.RandomResizedCrop(224, scale=(0.5, 1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.2), # Simulates color/texture shifts
        transforms.RandomAffine(degrees=15, translate=(0.1, 0.1), scale=(0.9, 1.1)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        transforms.RandomErasing(p=0.5, scale=(0.02, 0.2), value='random') # Simulates occlusion
    ])
    
    transform_val = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    dataset_train = MappedImagenetteDataset(raw_train_dir, transform=transform_train)
    dataloader_train = DataLoader(dataset_train, batch_size=64, shuffle=True, num_workers=4, pin_memory=True)
    
    dataset_val = MappedImagenetteDataset(raw_val_dir, transform=transform_val)
    dataloader_val = DataLoader(dataset_val, batch_size=64, shuffle=False, num_workers=4, pin_memory=True)
    
    model = RecurrentAttentionEnsemble(
        num_models=1, patch_size=48, num_classes=1000, hidden_dim=512, num_glimpses=8
    ).to(device)
    
    optimizer = optim.Adam(model.parameters(), lr=2e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    
    best_val_acc = 0.0
    best_model_state = None
    patience = 5
    epochs_no_improve = 0
    
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        total_reward = 0
        
        for images, targets in tqdm(dataloader_train, desc=f"Epoch {epoch+1}/{epochs} [Train RL Raw]"):
            images, targets = images.to(device), targets.to(device)
            
            optimizer.zero_grad()
            logits, log_probs, values, entropies = model(images)
            
            probs = torch.softmax(logits, dim=-1)
            preds = torch.argmax(probs, dim=-1)
            rewards = (preds == targets).float()
            
            rl_loss = compute_a2c_loss(log_probs, values, rewards, entropies=entropies)
            cls_loss = torch.nn.functional.cross_entropy(logits, targets)
            loss = rl_loss + cls_loss
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.5)
            optimizer.step()
            
            total_loss += loss.item()
            total_reward += rewards.mean().item()
            
        train_loss = total_loss / len(dataloader_train)
        train_acc = total_reward / len(dataloader_train)
        
        # Validation
        model.eval()
        val_correct = 0
        val_total = 0
        with torch.no_grad():
            for images, targets in tqdm(dataloader_val, desc=f"Epoch {epoch+1}/{epochs} [Val RL Raw]"):
                images, targets = images.to(device), targets.to(device)
                logits, _, _, _ = model(images)
                preds = torch.argmax(torch.softmax(logits, dim=-1), dim=-1)
                val_correct += (preds == targets).sum().item()
                val_total += targets.size(0)
                
        val_acc = val_correct / val_total
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
        
    os.makedirs(output_dir, exist_ok=True)
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
        
    save_path = os.path.join(output_dir, "rl_agent_imagenette_zeroshot.pth")
    torch.save(model.state_dict(), save_path)
    print(f"Saved zero-shot RL model to {save_path} (Best Val Acc: {best_val_acc:.4f})")
    
    return model

if __name__ == "__main__":
    base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    train_rl_on_raw_imagenette(
        dataset_dir=os.path.join(base_dir, "dataset"),
        output_dir=os.path.join(base_dir, "results")
    )
