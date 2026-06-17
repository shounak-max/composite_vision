"""
Fine-Tune Baselines on Composite Dataset.

To provide a perfectly fair comparison to the RL Agent trained on composites,
we fine-tune standard feedforward baselines (ResNet50, ConvNeXt, ViT-B/16) 
on the exact same composite training set.

We freeze the lower layers and fine-tune only the final block and classification 
head to prevent catastrophic forgetting of their ImageNet pre-training, while 
allowing them to adapt to compositional noise (AdaIN, occlusion, etc.).
"""

import os
import sys
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms, models
import timm
from tqdm import tqdm
import pandas as pd

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from src.models.baselines import CompositeDataset

def setup_finetune_model(model_name, num_classes=1000):
    """
    Load pretrained model, freeze lower layers, and prep for fine-tuning.
    """
    if model_name == "ResNet50":
        model = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
        # Freeze all except layer4 and fc
        for name, param in model.named_parameters():
            if not ("layer4" in name or "fc" in name):
                param.requires_grad = False
    elif model_name == "ViT-B/16":
        model = models.vit_b_16(weights=models.ViT_B_16_Weights.DEFAULT)
        # Freeze all except the last 2 encoder blocks and heads
        for name, param in model.named_parameters():
            if not ("encoder.layers.encoder_layer_10" in name or 
                    "encoder.layers.encoder_layer_11" in name or 
                    "heads" in name):
                param.requires_grad = False
    elif model_name == "ConvNeXt":
        model = models.convnext_tiny(weights=models.ConvNeXt_Tiny_Weights.DEFAULT)
        # Freeze all except features.7 (last stage) and classifier
        for name, param in model.named_parameters():
            if not ("features.7" in name or "classifier" in name):
                param.requires_grad = False
    else:
        raise ValueError(f"Unsupported model: {model_name}")
        
    return model

def finetune_model(model_name, dataset_dir, output_dir, epochs=10):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\n{'='*60}")
    print(f"Fine-tuning {model_name} on Composite Training Set")
    print(f"Device: {device}")
    print(f"{'='*60}")
    
    transform_train = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    dataset_train = CompositeDataset(
        os.path.join(dataset_dir, "metadata.json"), 
        os.path.join(dataset_dir, "images"), 
        transform=transform_train, split="train"
    )
    dataloader_train = DataLoader(dataset_train, batch_size=64, shuffle=True, num_workers=4, pin_memory=True)
    
    model = setup_finetune_model(model_name).to(device)
    
    # Only optimize parameters that require gradients
    optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    
    model.train()
    for epoch in range(epochs):
        total_loss = 0
        total_correct = 0
        
        for images, items in tqdm(dataloader_train, desc=f"Epoch {epoch+1}/{epochs} [{model_name} FT]"):
            images = images.to(device)
            targets = items['class1'].to(device)
            
            optimizer.zero_grad()
            outputs = model(images)
            loss = nn.functional.cross_entropy(outputs, targets)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            preds = outputs.argmax(dim=-1)
            total_correct += (preds == targets).sum().item()
            
        train_loss = total_loss / len(dataloader_train)
        train_acc = total_correct / len(dataset_train)
        print(f"  Epoch {epoch+1}: Loss={train_loss:.4f}, Acc={train_acc:.4f}")
        scheduler.step()
        
    os.makedirs(output_dir, exist_ok=True)
    save_path = os.path.join(output_dir, f"{model_name.replace('/', '_')}_finetuned.pth")
    torch.save(model.state_dict(), save_path)
    print(f"Saved fine-tuned model to {save_path}")
    
    return model

if __name__ == "__main__":
    base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    dataset_dir = os.path.join(base_dir, "dataset")
    output_dir = os.path.join(base_dir, "results")
    
    for model_name in ["ResNet50", "ViT-B/16", "ConvNeXt"]:
        finetune_model(model_name, dataset_dir, output_dir, epochs=10)
