import os
import json
import time
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
        image = Image.open(img_path).convert('RGB')
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
    
    try:
        print("Loading MAE (ImageNet fine-tuned)...")
        models['MAE'] = timm.create_model('vit_base_patch16_224.mae', pretrained=True).to(device).eval()
    except Exception as e:
        print(f"Skipping MAE: {e}")
        
    try:
        print("Loading DINOv2 (ImageNet linear probe/fine-tuned)...")
        # Attempt to load a timm dinov2 variant with a classification head
        models['DINOv2'] = timm.create_model('vit_base_patch14_dinov2.lvd142m', pretrained=True).to(device).eval()
    except Exception as e:
        print(f"Skipping DINOv2: {e}")
    
    print("Loading CLIP ViT-B/32...")
    clip_model, _, clip_preprocess = open_clip.create_model_and_transforms('ViT-B-32', pretrained='laion2b_s34b_b79k')
    models['CLIP'] = clip_model.to(device).eval()
    
    return models, clip_preprocess

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
        imagenet_classes_raw = [line.strip() for line in f.readlines()]
        
    clip_prompts = {
        'Standard': [f"a photo of a {cls}" for cls in imagenet_classes_raw],
        'Sketch': [f"a sketch of a {cls}" for cls in imagenet_classes_raw],
        'Shape': [f"the shape of a {cls}" for cls in imagenet_classes_raw]
    }
    
    clip_text_features = {}
    with torch.no_grad():
        for prompt_name, prompts in clip_prompts.items():
            text = clip_tokenizer(prompts).to(device)
            text_features = models['CLIP'].encode_text(text)
            text_features /= text_features.norm(dim=-1, keepdim=True)
            clip_text_features[prompt_name] = text_features
    
    results = []
    
    with torch.no_grad():
        for images, items in tqdm(dataloader, desc="Evaluating"):
            images = images.to(device)
            
            for model_name, model in models.items():
                # Warm up GPU for timing accuracy on first batch
                if device.type == 'cuda':
                    torch.cuda.synchronize()
                batch_start = time.time()
                
                if model_name == 'CLIP':
                    image_features = model.encode_image(images)
                    image_features /= image_features.norm(dim=-1, keepdim=True)
                    
                    if device.type == 'cuda':
                        torch.cuda.synchronize()
                    batch_elapsed = time.time() - batch_start
                    per_image_time = batch_elapsed / images.size(0)
                    
                    # Evaluate multiple prompts for CLIP
                    for prompt_name, text_features in clip_text_features.items():
                        logits = (100.0 * image_features @ text_features.T)
                        probs = F.softmax(logits, dim=-1)
                        top5_prob, top5_idx = torch.topk(probs, 5, dim=-1)
                        
                        for i in range(images.size(0)):
                            result = {
                                'filename': items['filename'][i],
                                'class1': items['class1'][i].item() if isinstance(items['class1'][i], torch.Tensor) else items['class1'][i],
                                'class2': items['class2'][i].item() if isinstance(items['class2'][i], torch.Tensor) else items['class2'][i],
                                'composition_type': items['composition_type'][i],
                                'salience': items['salience'][i],
                                'model': f"CLIP_{prompt_name}",
                                'top1_pred': top5_idx[i][0].item(),
                                'top1_conf': top5_prob[i][0].item(),
                                'top5_preds': top5_idx[i].cpu().numpy().tolist(),
                                'inference_time': per_image_time
                            }
                            results.append(result)
                    
                    continue # Skip the rest of the loop since CLIP is handled
                elif model_name == 'DINOv2':
                    # DINOv2 vit_base_patch14_dinov2.lvd142m strictly requires 518x518 input
                    resized_images = F.interpolate(images, size=(518, 518), mode='bicubic', align_corners=False)
                    logits = model(resized_images)
                    probs = F.softmax(logits, dim=-1)
                else:
                    logits = model(images)
                    probs = F.softmax(logits, dim=-1)
                
                if device.type == 'cuda':
                    torch.cuda.synchronize()
                batch_elapsed = time.time() - batch_start
                per_image_time = batch_elapsed / images.size(0)
                    
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
                        'inference_time': per_image_time
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
