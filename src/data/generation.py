import os
import json
import random
import torch
import numpy as np
from PIL import Image, ImageDraw
import torchvision.transforms.functional as F
import urllib.request
from tqdm import tqdm

class CompositeDatasetGenerator:
    def __init__(self, output_dir="dataset", num_classes=50):
        self.output_dir = output_dir
        self.num_classes = num_classes
        self.images_dir = os.path.join(output_dir, "images")
        os.makedirs(self.images_dir, exist_ok=True)
        self.metadata = []

    def _adain_rgb(self, content, style, alpha):
        c_np = np.array(content).astype(np.float32)
        s_np = np.array(style).astype(np.float32)
        
        c_mean, c_std = c_np.mean(axis=(0,1)), c_np.std(axis=(0,1)) + 1e-5
        s_mean, s_std = s_np.mean(axis=(0,1)), s_np.std(axis=(0,1)) + 1e-5
        
        adain = (c_np - c_mean) / c_std * s_std + s_mean
        adain = np.clip(adain, 0, 255)
        
        blended = (1 - alpha) * c_np + alpha * adain
        return Image.fromarray(np.clip(blended, 0, 255).astype(np.uint8))

    def _fft_texture_shape(self, shape_img, texture_img, alpha):
        shape_np = np.array(shape_img).astype(np.float32) / 255.0
        tex_np = np.array(texture_img).astype(np.float32) / 255.0
        
        out = np.zeros_like(shape_np)
        for i in range(3):
            fft_shape = np.fft.fft2(shape_np[:,:,i])
            fft_tex = np.fft.fft2(tex_np[:,:,i])
            
            amp_shape = np.abs(fft_shape)
            phase_shape = np.angle(fft_shape)
            amp_tex = np.abs(fft_tex)
            
            mixed_amp = (1 - alpha) * amp_shape + alpha * amp_tex
            mixed_fft = mixed_amp * np.exp(1j * phase_shape)
            out[:,:,i] = np.real(np.fft.ifft2(mixed_fft))
            
        out = np.clip(out * 255.0, 0, 255).astype(np.uint8)
        return Image.fromarray(out)

    def generate_adain(self, img1, img2, salience):
        alpha = 0.8 if salience == "high" else 0.4
        return self._adain_rgb(img1, img2, alpha)

    def generate_occlusion(self, img1, salience):
        img = img1.copy()
        draw = ImageDraw.Draw(img)
        w, h = img.size
        num_patches = 15 if salience == "high" else 5
        patch_size = w // 8
        for _ in range(num_patches):
            x = random.randint(0, max(1, w - patch_size))
            y = random.randint(0, max(1, h - patch_size))
            draw.rectangle([x, y, x+patch_size, y+patch_size], fill="black")
        return img

    def generate_superimposition(self, img1, img2, salience):
        alpha = 0.7 if salience == "high" else 0.3
        return Image.blend(img2, img1, alpha)

    def generate_texture_shape(self, img1, img2, salience):
        alpha = 0.3 if salience == "high" else 0.8
        return self._fft_texture_shape(img1, img2, alpha)

    def generate(self):
        print("Downloading sample real images to act as base ImageNet classes...")
        
        images_by_class = {}
        for target in range(self.num_classes):
            img_path = os.path.join(self.output_dir, f"base_class_{target}.jpg")
            try:
                req = urllib.request.Request(f"https://picsum.photos/seed/{target*10}/224/224", headers={'User-Agent': 'Mozilla/5.0'})
                with urllib.request.urlopen(req, timeout=10) as response, open(img_path, 'wb') as out_file:
                    out_file.write(response.read())
                img = Image.open(img_path).convert("RGB")
            except Exception as e:
                print(f"Failed to download image for class {target}: {e}. Using fallback noise.")
                img = Image.fromarray(np.random.randint(0, 256, (224, 224, 3), dtype=np.uint8))
            
            images_by_class[target] = [img, img]
            
        categories = list(images_by_class.keys())[:self.num_classes]
        
        idx = 0
        print("Generating composite stimuli...")
        for c1 in tqdm(categories):
            img1 = images_by_class[c1][0]
            c2_choices = [c for c in categories if c != c1]
            c2 = random.choice(c2_choices) if len(c2_choices) > 0 else c1
            img2 = images_by_class[c2][0] if len(images_by_class[c2]) > 0 else images_by_class[c1][0]
            
            compositions = [
                ("adain", self.generate_adain),
                ("occlusion", lambda i1, i2, s: self.generate_occlusion(i1, s)),
                ("superimposition", self.generate_superimposition),
                ("texture_shape", self.generate_texture_shape)
            ]
            
            for comp_name, comp_func in compositions:
                for salience in ["high", "low"]:
                    comp_img = comp_func(img1, img2, salience)
                    filename = f"{idx:05d}_{comp_name}_{salience}.png"
                    filepath = os.path.join(self.images_dir, filename)
                    comp_img.save(filepath)
                    
                    self.metadata.append({
                        "id": idx,
                        "filename": filename,
                        "class1": c1,
                        "class2": c2 if comp_name != "occlusion" else None,
                        "composition_type": comp_name,
                        "salience": salience,
                        "split": random.choices(["train", "val", "test"], weights=[0.6, 0.2, 0.2])[0]
                    })
                    idx += 1
                    
        meta_path = os.path.join(self.output_dir, "metadata.json")
        with open(meta_path, "w") as f:
            json.dump(self.metadata, f, indent=4)
        print(f"Generated {idx} stimuli. Metadata saved to {meta_path}")

if __name__ == '__main__':
    generator = CompositeDatasetGenerator(output_dir="d:/gitfork/composite_vision_research/dataset")
    generator.generate()
