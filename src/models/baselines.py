import os
import json
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import torchvision.models as tv_models
import timm
import open_clip
import pandas as pd
from tqdm import tqdm
import time

class CompositeDataset(Dataset):
    def __init__(self, metadata_path, images_dir, transform=None, split=None):
        with open(metadata_path, 'r') as f:
            self.metadata = json.load(f)
        if split:
            self.metadata = [item for item in self.metadata if item.get('split') == split]
        self.images_dir = images_dir
        self.transform = transform

    def __len__(self):
        return len(self.metadata)

    def __getitem__(self, idx):
        item = self.metadata[idx]
        img_path = os.path.join(self.images_dir, item['filename'])
        with Image.open(img_path) as img:
            image = img.convert('RGB')
        if self.transform:
            image = self.transform(image)
        return image, {k: (v if v is not None else -1) for k, v in item.items()}

def load_models(device):
    models = {}
    print("Loading ResNet50...")
    models['ResNet50'] = tv_models.resnet50(weights=tv_models.ResNet50_Weights.IMAGENET1K_V1).to(device).eval()
    
    print("Loading ResNet101...")
    models['ResNet101'] = tv_models.resnet101(weights=tv_models.ResNet101_Weights.IMAGENET1K_V1).to(device).eval()
    
    print("Loading ConvNeXt...")
    models['ConvNeXt'] = tv_models.convnext_tiny(weights=tv_models.ConvNeXt_Tiny_Weights.IMAGENET1K_V1).to(device).eval()
    
    print("Loading ViT-B/16...")
    models['ViT-B/16'] = tv_models.vit_b_16(weights=tv_models.ViT_B_16_Weights.IMAGENET1K_V1).to(device).eval()
    
    print("Loading DeiT...")
    models['DeiT'] = timm.create_model('deit_small_patch16_224', pretrained=True).to(device).eval()
    
    print("Loading CLIP ViT-B/32...")
    clip_model, _, clip_preprocess = open_clip.create_model_and_transforms('ViT-B-32', pretrained='laion2b_s34b_b79k')
    models['CLIP'] = clip_model.to(device).eval()
    
    return models, clip_preprocess


def profile_inference_time(model, model_name, device, input_size=(1, 3, 224, 224),
                           warmup_runs=10, timed_runs=50, is_clip=False, text_features=None):
    """
    Profile single-image inference time using torch.cuda.Event for GPU timing
    or time.perf_counter for CPU timing. Uses batch_size=1 for fair comparison.
    
    Args:
        model: the model to profile
        model_name: string name
        device: torch device
        input_size: input tensor shape (batch=1)
        warmup_runs: number of warm-up forward passes (discarded)
        timed_runs: number of timed forward passes
        is_clip: if True, use encode_image path
        text_features: precomputed CLIP text features (required if is_clip=True)
    
    Returns:
        mean_time: average inference time per image in seconds
    """
    dummy_input = torch.randn(*input_size, device=device)
    use_cuda = device.type == 'cuda'
    
    # Warm-up: fill GPU pipeline caches
    with torch.no_grad():
        for _ in range(warmup_runs):
            if is_clip:
                img_feat = model.encode_image(dummy_input)
                img_feat /= img_feat.norm(dim=-1, keepdim=True)
                _ = (100.0 * img_feat @ text_features.T)
            else:
                _ = model(dummy_input)
    
    if use_cuda:
        torch.cuda.synchronize()
    
    # Timed runs
    times = []
    for _ in range(timed_runs):
        if use_cuda:
            start_event = torch.cuda.Event(enable_timing=True)
            end_event = torch.cuda.Event(enable_timing=True)
            start_event.record()
        else:
            t_start = time.perf_counter()
        
        with torch.no_grad():
            if is_clip:
                img_feat = model.encode_image(dummy_input)
                img_feat /= img_feat.norm(dim=-1, keepdim=True)
                _ = (100.0 * img_feat @ text_features.T)
            else:
                _ = model(dummy_input)
        
        if use_cuda:
            end_event.record()
            torch.cuda.synchronize()
            elapsed_ms = start_event.elapsed_time(end_event)
            times.append(elapsed_ms / 1000.0)  # convert to seconds
        else:
            t_end = time.perf_counter()
            times.append(t_end - t_start)
    
    mean_time = sum(times) / len(times)
    print(f"  {model_name}: {mean_time*1000:.2f} ms/image (std: {(sum((t-mean_time)**2 for t in times)/len(times))**0.5*1000:.2f} ms)")
    return mean_time


def evaluate_models(metadata_path, images_dir, output_dir):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    models, clip_preprocess = load_models(device)
    
    standard_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    dataset = CompositeDataset(metadata_path, images_dir, transform=standard_transform, split="test")
    dataloader = DataLoader(dataset, batch_size=16, shuffle=False)
    
    clip_tokenizer = open_clip.get_tokenizer('ViT-B-32')
    classes_path = os.path.join(os.path.dirname(__file__), 'imagenet_classes.txt')
    with open(classes_path, 'r') as f:
        imagenet_classes = [f"a photo of a {line.strip()}" for line in f.readlines()]
    text = clip_tokenizer(imagenet_classes).to(device)
    with torch.no_grad():
        text_features = models['CLIP'].encode_text(text)
        text_features /= text_features.norm(dim=-1, keepdim=True)
    
    # --- Profile inference times BEFORE evaluation (batch_size=1, fair comparison) ---
    print("\n=== Profiling Inference Times (batch_size=1, 50 timed runs) ===")
    inference_times = {}
    for model_name, model in models.items():
        is_clip = (model_name == 'CLIP')
        inference_times[model_name] = profile_inference_time(
            model, model_name, device,
            is_clip=is_clip, text_features=text_features if is_clip else None
        )
    print("=== Profiling Complete ===\n")
    
    # --- Standard evaluation loop ---
    results = []
    
    with torch.no_grad():
        for images, items in tqdm(dataloader, desc="Evaluating"):
            images = images.to(device)
            
            for model_name, model in models.items():
                if model_name == 'CLIP':
                    image_features = model.encode_image(images)
                    image_features /= image_features.norm(dim=-1, keepdim=True)
                    logits = (100.0 * image_features @ text_features.T)
                    probs = F.softmax(logits, dim=-1)
                else:
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
                        'model': model_name,
                        'top1_pred': top5_idx[i][0].item(),
                        'top1_conf': top5_prob[i][0].item(),
                        'top5_preds': top5_idx[i].cpu().numpy().tolist(),
                        'inference_time': inference_times[model_name]
                    }
                    results.append(result)
                    
    os.makedirs(output_dir, exist_ok=True)
    df = pd.DataFrame(results)
    df.to_csv(os.path.join(output_dir, "benchmark_results.csv"), index=False)
    with open(os.path.join(output_dir, "benchmark_results.json"), 'w') as f:
        json.dump(results, f, indent=4)
    print("Evaluation complete. Results saved.")

if __name__ == '__main__':
    import os
    base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    evaluate_models(os.path.join(base_dir, "dataset", "metadata.json"), 
                    os.path.join(base_dir, "dataset", "images"), 
                    os.path.join(base_dir, "results"))
