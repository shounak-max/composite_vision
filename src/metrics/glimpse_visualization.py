"""
Glimpse Trajectory Visualization for CompositeVision.

Extracts and visualizes the sequential fixation points (glimpse locations)
from the RL Attention Agent on representative composite images.

Produces:
1. Overlay images showing numbered fixation circles on the composite
2. Side-by-side: GradCAM (ResNet) vs. Glimpse trajectory (RL agent)
3. Per-composition-type visualization grids

These qualitative results make the RL mechanism interpretable and visually
compelling — critical for poster/spotlight presentation at conferences.
"""

import os
import sys
import torch
import torch.nn.functional as F
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from torchvision import transforms, models
import matplotlib.pyplot as plt
import matplotlib.patches as patches

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from src.models.rl_attention import RecurrentAttentionModel, RecurrentAttentionEnsemble
from src.models.baselines import CompositeDataset


def extract_glimpse_trajectory(model, image_tensor, device='cuda'):
    """
    Extract the sequence of fixation locations from a trained RL agent.
    
    Args:
        model: trained RecurrentAttentionModel (or first model in ensemble)
        image_tensor: preprocessed image tensor [1, 3, H, W]
        device: compute device
        
    Returns:
        list of (x, y) tuples in normalized coords [-1, 1], plus the prediction
    """
    # Get the actual RecurrentAttentionModel (unwrap ensemble if needed)
    if isinstance(model, RecurrentAttentionEnsemble):
        ram = model.models[0]
    else:
        ram = model
    
    ram.eval()
    image_tensor = image_tensor.to(device)
    
    B = image_tensor.size(0)
    hx = torch.zeros(B, ram.rnn.hidden_size, device=device)
    cx = torch.zeros(B, ram.rnn.hidden_size, device=device)
    locs = torch.zeros(B, 2, device=device)
    
    trajectory = []
    logits_list = []
    
    with torch.no_grad():
        for i in range(ram.num_glimpses):
            trajectory.append(locs.cpu().numpy()[0].copy())  # Save location
            
            g_t_concat, g_t_avg = ram.glimpse_net(image_tensor, locs)
            hx, cx = ram.rnn(g_t_concat.detach(), (hx, cx))
            
            mean_loc, std = ram.loc_net(hx)
            locs = mean_loc  # Deterministic (eval mode)
            
            logits_list.append(ram.classifier(g_t_avg))
        
        # Final prediction
        foveated_logits = torch.stack(logits_list, dim=1).mean(dim=1)
        global_features = ram.global_extractor(image_tensor)
        global_logits = ram.global_classifier(global_features)
        logits = 0.5 * global_logits + 0.5 * foveated_logits
        pred = torch.argmax(logits, dim=-1).item()
    
    return trajectory, pred


def compute_gradcam(model, image_tensor, target_class=None, device='cuda'):
    """
    Compute GradCAM heatmap for a ResNet model.
    
    Uses the gradient of the target class w.r.t. the last convolutional layer
    to produce a spatial attention map showing where the model "looks".
    
    Args:
        model: pretrained ResNet model
        image_tensor: preprocessed image [1, 3, H, W]
        target_class: class index (default: predicted class)
        device: compute device
        
    Returns:
        numpy array [H, W] with GradCAM heatmap values in [0, 1]
    """
    model.eval()
    image_tensor = image_tensor.to(device).requires_grad_(True)
    
    # Hook into the last conv layer
    activations = {}
    gradients = {}
    
    def forward_hook(module, input, output):
        activations['value'] = output.detach()
    
    def backward_hook(module, grad_input, grad_output):
        gradients['value'] = grad_output[0].detach()
    
    # Register hooks on layer4 (last conv block)
    target_layer = model.layer4[-1]
    fh = target_layer.register_forward_hook(forward_hook)
    bh = target_layer.register_full_backward_hook(backward_hook)
    
    # Forward pass
    output = model(image_tensor)
    
    if target_class is None:
        target_class = output.argmax(dim=1).item()
    
    # Backward pass for the target class
    model.zero_grad()
    output[0, target_class].backward()
    
    # Compute GradCAM
    grads = gradients['value']  # [1, C, h, w]
    acts = activations['value']  # [1, C, h, w]
    
    weights = grads.mean(dim=(2, 3), keepdim=True)  # Global average pooling of gradients
    cam = (weights * acts).sum(dim=1, keepdim=True)  # Weighted combination
    cam = F.relu(cam)  # Only positive contributions
    
    # Resize to image dimensions
    cam = F.interpolate(cam, size=(224, 224), mode='bilinear', align_corners=False)
    cam = cam.squeeze().cpu().numpy()
    
    # Normalize to [0, 1]
    if cam.max() > 0:
        cam = (cam - cam.min()) / (cam.max() - cam.min())
    
    # Clean up hooks
    fh.remove()
    bh.remove()
    
    return cam


def draw_trajectory_on_image(pil_image, trajectory, image_size=224):
    """
    Draw numbered fixation circles on a PIL image.
    
    Args:
        pil_image: original PIL image (will be copied)
        trajectory: list of (x, y) in [-1, 1] normalized coordinates
        image_size: expected size of the image
        
    Returns:
        PIL Image with trajectory overlay
    """
    img = pil_image.copy().resize((image_size, image_size))
    draw = ImageDraw.Draw(img)
    
    # Color gradient: blue (early) → red (late)
    n = len(trajectory)
    colors = []
    for i in range(n):
        r = int(255 * i / max(n - 1, 1))
        b = int(255 * (1 - i / max(n - 1, 1)))
        colors.append((r, 80, b))
    
    # Convert normalized coords to pixel coords
    points = []
    for loc in trajectory:
        px = int((loc[0] + 1) / 2 * image_size)
        py = int((loc[1] + 1) / 2 * image_size)
        px = max(0, min(image_size - 1, px))
        py = max(0, min(image_size - 1, py))
        points.append((px, py))
    
    # Draw connecting lines
    for i in range(len(points) - 1):
        draw.line([points[i], points[i + 1]], fill=(255, 255, 255, 180), width=2)
    
    # Draw circles with numbers
    radius = 10
    for i, (px, py) in enumerate(points):
        # Circle outline
        draw.ellipse(
            [px - radius, py - radius, px + radius, py + radius],
            fill=colors[i], outline='white', width=2
        )
        # Number label
        text = str(i + 1)
        try:
            font = ImageFont.truetype("arial.ttf", 11)
        except (OSError, IOError):
            font = ImageFont.load_default()
        
        bbox = draw.textbbox((0, 0), text, font=font)
        tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]
        draw.text((px - tw // 2, py - th // 2), text, fill='white', font=font)
    
    return img


def generate_comparison_figure(composite_image_path, rl_model, resnet_model,
                                metadata_item, output_path, device='cuda'):
    """
    Generate a side-by-side comparison figure:
    Left: GradCAM heatmap (ResNet50) 
    Right: Glimpse trajectory (RL Agent)
    
    Args:
        composite_image_path: path to the composite image
        rl_model: trained RL attention model
        resnet_model: pretrained ResNet50
        metadata_item: dict with class1, class2, composition_type, salience
        output_path: where to save the figure
        device: compute device
    """
    # Load and preprocess
    pil_img = Image.open(composite_image_path).convert('RGB')
    
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    img_tensor = transform(pil_img).unsqueeze(0)
    
    # Get RL trajectory
    trajectory, rl_pred = extract_glimpse_trajectory(rl_model, img_tensor, device)
    
    # Get GradCAM
    gradcam = compute_gradcam(resnet_model, img_tensor, target_class=metadata_item['class1'], device=device)
    
    # ResNet prediction
    resnet_model.eval()
    with torch.no_grad():
        resnet_logits = resnet_model(img_tensor.to(device))
        resnet_pred = torch.argmax(resnet_logits, dim=-1).item()
    
    # Build figure
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # Original image
    display_img = pil_img.resize((224, 224))
    axes[0].imshow(display_img)
    axes[0].set_title(f"Composite ({metadata_item['composition_type']})\n"
                      f"Shape: class {metadata_item['class1']}, "
                      f"Texture: class {metadata_item.get('class2', 'N/A')}",
                      fontsize=10)
    axes[0].axis('off')
    
    # GradCAM overlay
    axes[1].imshow(display_img)
    axes[1].imshow(gradcam, alpha=0.5, cmap='jet')
    correct_resnet = "✓" if resnet_pred == metadata_item['class1'] else "✗"
    axes[1].set_title(f"ResNet50 GradCAM\nPred: {resnet_pred} {correct_resnet}", fontsize=10)
    axes[1].axis('off')
    
    # Glimpse trajectory
    trajectory_img = draw_trajectory_on_image(pil_img, trajectory)
    axes[2].imshow(trajectory_img)
    correct_rl = "✓" if rl_pred == metadata_item['class1'] else "✗"
    axes[2].set_title(f"RL Glimpse Trajectory (8 steps)\nPred: {rl_pred} {correct_rl}", fontsize=10)
    axes[2].axis('off')
    
    plt.suptitle(f"Salience: {metadata_item['salience']}", fontsize=11, y=0.02)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()


def generate_all_visualizations(dataset_dir, results_dir, rl_checkpoint_path,
                                  num_per_type=3, device='cuda'):
    """
    Generate visualization figures for representative examples from each composition type.
    
    Selects examples where RL agent succeeded and ResNet failed (most compelling cases).
    
    Args:
        dataset_dir: path to dataset
        results_dir: path to results (contains benchmark_results.csv)
        rl_checkpoint_path: path to rl_agent.pth
        num_per_type: number of examples per composition type
        device: compute device
    """
    import pandas as pd
    import json
    
    viz_dir = os.path.join(results_dir, "visualizations")
    os.makedirs(viz_dir, exist_ok=True)
    
    # Load models
    print("Loading models for visualization...")
    resnet = models.resnet50(weights=models.ResNet50_Weights.DEFAULT).to(device)
    resnet.eval()
    
    rl_model = RecurrentAttentionEnsemble(
        num_models=1, patch_size=48, num_classes=1000, hidden_dim=512, num_glimpses=8
    ).to(device)
    rl_model.load_state_dict(torch.load(rl_checkpoint_path, map_location=device))
    rl_model.eval()
    
    # Load results to find interesting examples
    csv_path = os.path.join(results_dir, "benchmark_results.csv")
    df = pd.read_csv(csv_path)
    
    # Find cases where RL is correct and ResNet is wrong
    df_rl = df[df['model'] == 'RL_Attention'].set_index('filename')
    df_rn = df[df['model'] == 'ResNet50'].set_index('filename')
    
    common = set(df_rl.index) & set(df_rn.index)
    
    interesting_cases = []
    for fname in common:
        rl_row = df_rl.loc[fname]
        rn_row = df_rn.loc[fname]
        
        rl_correct = (rl_row['top1_pred'] == rl_row['class1'])
        rn_correct = (rn_row['top1_pred'] == rn_row['class1'])
        
        if rl_correct and not rn_correct:
            interesting_cases.append({
                'filename': fname,
                'class1': int(rl_row['class1']),
                'class2': int(rl_row['class2']) if rl_row['class2'] != -1 else -1,
                'composition_type': rl_row['composition_type'],
                'salience': rl_row['salience'],
            })
    
    print(f"Found {len(interesting_cases)} cases where RL correct & ResNet wrong")
    
    # Select representative examples per composition type
    comp_types = set(c['composition_type'] for c in interesting_cases)
    selected = []
    for comp in sorted(comp_types):
        cases = [c for c in interesting_cases if c['composition_type'] == comp]
        selected.extend(cases[:num_per_type])
    
    # Generate figures
    for i, case in enumerate(selected):
        img_path = os.path.join(dataset_dir, "images", case['filename'])
        if not os.path.exists(img_path):
            continue
        
        output_path = os.path.join(viz_dir, f"comparison_{case['composition_type']}_{i}.png")
        
        try:
            generate_comparison_figure(
                img_path, rl_model, resnet, case, output_path, device
            )
            print(f"  Generated: {output_path}")
        except Exception as e:
            print(f"  Error on {case['filename']}: {e}")
    
    # Generate grid figure (one row per composition type)
    _generate_grid_figure(selected, dataset_dir, rl_model, resnet, viz_dir, device)
    
    print(f"\nVisualization complete. Figures saved to {viz_dir}")


def _generate_grid_figure(cases, dataset_dir, rl_model, resnet, viz_dir, device):
    """Generate a grid figure with one representative example per composition type."""
    comp_types = sorted(set(c['composition_type'] for c in cases))
    
    if len(comp_types) == 0:
        return
    
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    fig, axes = plt.subplots(len(comp_types), 3, figsize=(15, 4 * len(comp_types)))
    if len(comp_types) == 1:
        axes = axes[np.newaxis, :]
    
    for row, comp in enumerate(comp_types):
        case = next(c for c in cases if c['composition_type'] == comp)
        img_path = os.path.join(dataset_dir, "images", case['filename'])
        
        try:
            pil_img = Image.open(img_path).convert('RGB')
            img_tensor = transform(pil_img).unsqueeze(0)
            display_img = pil_img.resize((224, 224))
            
            # RL trajectory
            trajectory, rl_pred = extract_glimpse_trajectory(rl_model, img_tensor, device)
            
            # GradCAM
            gradcam = compute_gradcam(resnet, img_tensor, target_class=case['class1'], device=device)
            
            # Original
            axes[row, 0].imshow(display_img)
            axes[row, 0].set_title(f"{comp}\n(class1={case['class1']})", fontsize=9)
            axes[row, 0].axis('off')
            
            # GradCAM
            axes[row, 1].imshow(display_img)
            axes[row, 1].imshow(gradcam, alpha=0.5, cmap='jet')
            axes[row, 1].set_title("ResNet50 GradCAM", fontsize=9)
            axes[row, 1].axis('off')
            
            # Trajectory
            traj_img = draw_trajectory_on_image(pil_img, trajectory)
            axes[row, 2].imshow(traj_img)
            axes[row, 2].set_title("RL Glimpse Path", fontsize=9)
            axes[row, 2].axis('off')
        except Exception as e:
            print(f"  Grid error for {comp}: {e}")
    
    # Column headers
    if len(comp_types) > 0:
        axes[0, 0].set_title("Composite Image\n" + axes[0, 0].get_title(), fontsize=9)
        axes[0, 1].set_title("ResNet50 GradCAM\n(where CNN looks)", fontsize=9)
        axes[0, 2].set_title("RL Glimpse Path\n(where agent foveates)", fontsize=9)
    
    plt.suptitle("GradCAM vs. Learned Foveation: Where Models Attend", fontsize=14, y=1.01)
    plt.tight_layout()
    plt.savefig(os.path.join(viz_dir, "comparison_grid.png"), dpi=300, bbox_inches='tight')
    plt.savefig(os.path.join(viz_dir, "comparison_grid.pdf"), bbox_inches='tight')
    plt.close()


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Generate glimpse trajectory visualizations")
    parser.add_argument("--dataset-dir", type=str, default=None)
    parser.add_argument("--results-dir", type=str, default=None)
    parser.add_argument("--rl-checkpoint", type=str, default=None)
    parser.add_argument("--num-per-type", type=int, default=3)
    args = parser.parse_args()
    
    base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    
    if args.dataset_dir is None:
        args.dataset_dir = os.path.join(base_dir, "dataset")
    if args.results_dir is None:
        args.results_dir = os.path.join(base_dir, "results")
    if args.rl_checkpoint is None:
        args.rl_checkpoint = os.path.join(base_dir, "results", "rl_agent.pth")
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    generate_all_visualizations(
        args.dataset_dir, args.results_dir, args.rl_checkpoint,
        args.num_per_type, device
    )
