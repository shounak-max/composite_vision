import os
import json
import random
import tarfile
import urllib.request
import shutil
import torch
import numpy as np
from PIL import Image, ImageDraw
import torchvision.transforms.functional as F
from tqdm import tqdm

# Imagenette: 10-class subset of ImageNet with REAL ImageNet class indices
# This ensures pretrained ImageNet models can produce meaningful accuracy
IMAGENETTE_URL = "https://s3.amazonaws.com/fast-ai-imageclas/imagenette2-160.tgz"

# Mapping from Imagenette folder names to real ImageNet-1K class indices
IMAGENETTE_CLASS_MAP = {
    "n01440764": 0,    # tench
    "n02102040": 217,  # English springer
    "n02979186": 482,  # cassette player
    "n03000684": 491,  # chain saw
    "n03028079": 497,  # church
    "n03394916": 566,  # French horn
    "n03417042": 569,  # garbage truck
    "n03425413": 571,  # gas pump
    "n03445777": 574,  # golf ball
    "n03888257": 701,  # parachute
}

IMAGENETTE_CLASSES = list(IMAGENETTE_CLASS_MAP.keys())
NUM_CLASSES = len(IMAGENETTE_CLASSES)  # 10 real classes


class CompositeDatasetGenerator:
    def __init__(self, output_dir="dataset", num_samples_per_pair=2):
        self.output_dir = output_dir
        self.images_dir = os.path.join(output_dir, "images")
        self.raw_dir = os.path.join(output_dir, "imagenette_raw")
        self.num_samples_per_pair = num_samples_per_pair
        os.makedirs(self.images_dir, exist_ok=True)
        self.metadata = []

    def _download_imagenette(self):
        """Download and extract Imagenette2-160 (~100MB)."""
        tgz_path = os.path.join(self.output_dir, "imagenette2-160.tgz")
        extract_dir = os.path.join(self.output_dir, "imagenette2-160")

        if os.path.exists(extract_dir):
            print("Imagenette already downloaded.")
            return extract_dir

        print(f"Downloading Imagenette2-160 (~100MB) from {IMAGENETTE_URL}...")
        os.makedirs(self.output_dir, exist_ok=True)
        urllib.request.urlretrieve(IMAGENETTE_URL, tgz_path)
        print("Download complete. Extracting...")

        tf = tarfile.open(tgz_path, "r:gz")
        tf.extractall(path=self.output_dir)
        tf.close()

        # Clean up the tarball to save space
        try:
            os.remove(tgz_path)
        except PermissionError:
            print("Note: Could not delete tarball (file locked). Continuing...")
        print("Extraction complete.")
        return extract_dir

    def _load_real_images(self, imagenette_dir):
        """Load real images organized by their true ImageNet class index."""
        images_by_class = {}
        val_dir = os.path.join(imagenette_dir, "val")

        for folder_name in IMAGENETTE_CLASSES:
            imagenet_idx = IMAGENETTE_CLASS_MAP[folder_name]
            class_dir = os.path.join(val_dir, folder_name)

            if not os.path.isdir(class_dir):
                print(f"Warning: {class_dir} not found, skipping class {folder_name}")
                continue

            class_images = []
            for fname in os.listdir(class_dir):
                if fname.lower().endswith(('.jpeg', '.jpg', '.png')):
                    fpath = os.path.join(class_dir, fname)
                    try:
                        img = Image.open(fpath).convert("RGB").resize((224, 224))
                        class_images.append(img)
                    except Exception as e:
                        print(f"  Skipping {fname}: {e}")

            if class_images:
                images_by_class[imagenet_idx] = class_images
                print(f"  Class {folder_name} (ImageNet idx={imagenet_idx}): {len(class_images)} images")

        return images_by_class

    # --- Composition functions (unchanged) ---

    def _adain_rgb(self, content, style, alpha):
        c_np = np.array(content).astype(np.float32)
        s_np = np.array(style).astype(np.float32)

        c_mean, c_std = c_np.mean(axis=(0, 1)), c_np.std(axis=(0, 1)) + 1e-5
        s_mean, s_std = s_np.mean(axis=(0, 1)), s_np.std(axis=(0, 1)) + 1e-5

        adain = (c_np - c_mean) / c_std * s_std + s_mean
        adain = np.clip(adain, 0, 255)

        blended = (1 - alpha) * c_np + alpha * adain
        return Image.fromarray(np.clip(blended, 0, 255).astype(np.uint8))

    def _fft_texture_shape(self, shape_img, texture_img, alpha):
        shape_np = np.array(shape_img).astype(np.float32) / 255.0
        tex_np = np.array(texture_img).astype(np.float32) / 255.0

        out = np.zeros_like(shape_np)
        for i in range(3):
            fft_shape = np.fft.fft2(shape_np[:, :, i])
            fft_tex = np.fft.fft2(tex_np[:, :, i])

            amp_shape = np.abs(fft_shape)
            phase_shape = np.angle(fft_shape)
            amp_tex = np.abs(fft_tex)

            mixed_amp = (1 - alpha) * amp_shape + alpha * amp_tex
            mixed_fft = mixed_amp * np.exp(1j * phase_shape)
            out[:, :, i] = np.real(np.fft.ifft2(mixed_fft))

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
            draw.rectangle([x, y, x + patch_size, y + patch_size], fill="black")
        return img

    def generate_superimposition(self, img1, img2, salience):
        alpha = 0.7 if salience == "high" else 0.3
        return Image.blend(img2, img1, alpha)

    def generate_texture_shape(self, img1, img2, salience):
        alpha = 0.3 if salience == "high" else 0.8
        return self._fft_texture_shape(img1, img2, alpha)

    def generate(self):
        # Step 1: Download real ImageNet subset
        imagenette_dir = self._download_imagenette()

        # Step 2: Load real images with real ImageNet class indices
        print("Loading real ImageNet images from Imagenette...")
        images_by_class = self._load_real_images(imagenette_dir)

        if len(images_by_class) < 2:
            raise RuntimeError("Need at least 2 classes with images to generate composites")

        class_indices = sorted(images_by_class.keys())
        print(f"\nLoaded {len(class_indices)} classes: {class_indices}")

        # Step 3: Generate composite stimuli using real images
        idx = 0
        print("\nGenerating composite stimuli from real ImageNet images...")

        for c1 in tqdm(class_indices, desc="Classes"):
            c2_choices = [c for c in class_indices if c != c1]

            for _ in range(self.num_samples_per_pair):
                c2 = random.choice(c2_choices)

                img1 = random.choice(images_by_class[c1])
                img2 = random.choice(images_by_class[c2])

                compositions = [
                    ("adain", self.generate_adain),
                    ("occlusion", lambda i1, i2, s: self.generate_occlusion(i1, s)),
                    ("superimposition", self.generate_superimposition),
                    ("texture_shape", self.generate_texture_shape),
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
                            "class1": c1,       # Real ImageNet class index
                            "class2": c2 if comp_name != "occlusion" else -1,
                            "composition_type": comp_name,
                            "salience": salience,
                            "split": random.choices(
                                ["train", "val", "test"], weights=[0.6, 0.2, 0.2]
                            )[0],
                        })
                        idx += 1

        meta_path = os.path.join(self.output_dir, "metadata.json")
        with open(meta_path, "w") as f:
            json.dump(self.metadata, f, indent=4)
        print(f"\nGenerated {idx} composite stimuli with REAL ImageNet labels.")
        print(f"Metadata saved to {meta_path}")
        print(f"Classes used (ImageNet indices): {class_indices}")


if __name__ == "__main__":
    generator = CompositeDatasetGenerator(
        output_dir="d:/gitfork/composite_vision_research/dataset"
    )
    generator.generate()
