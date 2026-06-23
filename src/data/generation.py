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
    """
    Generates the CompositeVision dataset used to probe architectural inductive biases.
    
    CONCEPT: 
    Instead of using standard images, this dataset creates 'composites' that pit 
    different visual features (shape, texture, color, edges) against each other.
    By observing which feature a model relies on to make its classification, we can 
    determine its bias (e.g., CNNs often show a texture bias, while humans and 
    foveated agents show a shape bias).
    """
    def __init__(self, output_dir="dataset", num_samples_per_pair=2, scale_factor=1):
        self.output_dir = output_dir
        self.images_dir = os.path.join(output_dir, "images")
        self.raw_dir = os.path.join(output_dir, "imagenette_raw")
        self.num_samples_per_pair = num_samples_per_pair
        self.scale_factor = scale_factor
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
        """Load real images organized by their true ImageNet class index.
        
        CRITICAL FIX: Split source images BEFORE compositing to prevent
        any source image from appearing in both train and test composites.
        Returns separate image pools for train and test.
        """
        train_images = {}
        test_images = {}
        val_dir = os.path.join(imagenette_dir, "val")

        for folder_name in IMAGENETTE_CLASSES:
            imagenet_idx = IMAGENETTE_CLASS_MAP[folder_name]
            class_dir = os.path.join(val_dir, folder_name)

            if not os.path.isdir(class_dir):
                print(f"Warning: {class_dir} not found, skipping class {folder_name}")
                continue

            class_images = []
            for fname in sorted(os.listdir(class_dir)):
                if fname.lower().endswith(('.jpeg', '.jpg', '.png')):
                    fpath = os.path.join(class_dir, fname)
                    try:
                        img = Image.open(fpath).convert("RGB").resize((224, 224))
                        class_images.append(img)
                    except Exception as e:
                        print(f"  Skipping {fname}: {e}")

            if class_images:
                # CONCEPTUAL FIX: Use a local Random instance instead of resetting the global random.seed(42) 
                # inside a loop. Resetting the global seed in a loop destroys the pseudo-random 
                # sequence for the rest of the script (including composition generation).
                local_rng = random.Random(42)  # Reproducible split generator
                local_rng.shuffle(class_images)
                
                # Split source images: first 70% for train, last 30% for test
                # This ensures ZERO source image overlap between splits, preventing data leakage.
                split_idx = int(len(class_images) * 0.7)
                train_images[imagenet_idx] = class_images[:split_idx]
                test_images[imagenet_idx] = class_images[split_idx:]
                print(f"  Class {folder_name} (idx={imagenet_idx}): "
                      f"{len(train_images[imagenet_idx])} train / "
                      f"{len(test_images[imagenet_idx])} test source images")

        return train_images, test_images

    # --- Composition functions ---
    # These functions generate specific cognitive conflict stimuli.
    # Salience ("high" or "low") dictates the intensity of the conflicting feature.

    def _adain_rgb(self, content, style, alpha):
        """
        Adaptive Instance Normalization (AdaIN) in RGB space.
        Transfers the color/texture distribution (mean and variance) of the 'style' image
        onto the 'content' image, preserving the spatial structure of the content.
        """
        c_np = np.array(content).astype(np.float32)
        s_np = np.array(style).astype(np.float32)

        c_mean, c_std = c_np.mean(axis=(0, 1)), c_np.std(axis=(0, 1)) + 1e-5
        s_mean, s_std = s_np.mean(axis=(0, 1)), s_np.std(axis=(0, 1)) + 1e-5

        adain = (c_np - c_mean) / c_std * s_std + s_mean
        adain = np.clip(adain, 0, 255)

        blended = (1 - alpha) * c_np + alpha * adain
        return Image.fromarray(np.clip(blended, 0, 255).astype(np.uint8))

    def _fft_texture_shape(self, shape_img, texture_img, alpha):
        """
        Fourier Transform based texture/shape swapping.
        CONCEPT: Combines the phase (structural shape) of one image with the 
        amplitude (texture/frequencies) of another image in the frequency domain.
        """
        shape_np = np.array(shape_img).astype(np.float32) / 255.0
        tex_np = np.array(texture_img).astype(np.float32) / 255.0

        out = np.zeros_like(shape_np)
        for i in range(3): # Process each RGB channel independently
            fft_shape = np.fft.fft2(shape_np[:, :, i])
            fft_tex = np.fft.fft2(tex_np[:, :, i])

            amp_shape = np.abs(fft_shape)
            phase_shape = np.angle(fft_shape)
            amp_tex = np.abs(fft_tex)

            # Mix amplitudes while keeping the phase of the shape image
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
        
    def generate_edge_conflict(self, img1, img2, salience):
        # Extract edges from img1 (shape) and blend with img2 (texture)
        from PIL import ImageFilter
        edges = img1.convert("L").filter(ImageFilter.FIND_EDGES).convert("RGB")
        alpha = 0.6 if salience == "high" else 0.3
        return Image.blend(img2, edges, alpha)
        
    def generate_patch_shuffle(self, img1, salience):
        # Shuffle patches of the image to destroy global shape but keep local texture
        img = img1.copy()
        w, h = img.size
        grid_size = 4 if salience == "high" else 8
        pw, ph = w // grid_size, h // grid_size
        patches = []
        for i in range(grid_size):
            for j in range(grid_size):
                box = (i*pw, j*ph, (i+1)*pw, (j+1)*ph)
                patches.append((box, img.crop(box)))
        
        # Shuffle a percentage of patches
        num_shuffle = len(patches) // (2 if salience == "high" else 4)
        indices = list(range(len(patches)))
        shuffle_idx = random.sample(indices, num_shuffle)
        shuffled_targets = shuffle_idx.copy()
        random.shuffle(shuffled_targets)
        
        for orig_idx, targ_idx in zip(shuffle_idx, shuffled_targets):
            orig_box = patches[orig_idx][0]
            targ_patch = patches[targ_idx][1]
            img.paste(targ_patch, orig_box)
        return img
        
    def generate_color_inversion(self, img1, img2, salience):
        from PIL import ImageOps
        inv_img1 = ImageOps.invert(img1)
        alpha = 0.7 if salience == "high" else 0.4
        return Image.blend(img2, inv_img1, alpha)

    def _generate_composites_for_split(self, images_by_class, split_name, 
                                        samples_per_pair, start_idx):
        """Generate composite images for a specific split using its own source images."""
        idx = start_idx
        class_indices = sorted(images_by_class.keys())
        
        for c1 in class_indices:
            c2_choices = [c for c in class_indices if c != c1]

            for _ in range(samples_per_pair):
                c2 = random.choice(c2_choices)

                img1 = random.choice(images_by_class[c1])
                img2 = random.choice(images_by_class[c2])

                compositions = [
                    ("adain", self.generate_adain),
                    ("occlusion", lambda i1, i2, s: self.generate_occlusion(i1, s)),
                    ("superimposition", self.generate_superimposition),
                    ("texture_shape", self.generate_texture_shape),
                    ("edge_conflict", self.generate_edge_conflict),
                    ("patch_shuffle", lambda i1, i2, s: self.generate_patch_shuffle(i1, s)),
                    ("color_inversion", self.generate_color_inversion),
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
                            "split": split_name,
                        })
                        idx += 1
        return idx

    def generate(self):
        # Step 1: Download real ImageNet subset
        imagenette_dir = self._download_imagenette()

        # Step 2: Load real images with SEPARATE pools for train and test
        print("Loading real ImageNet images from Imagenette...")
        train_images, test_images = self._load_real_images(imagenette_dir)

        if len(train_images) < 2:
            raise RuntimeError("Need at least 2 classes with images to generate composites")

        class_indices = sorted(train_images.keys())
        print(f"\nLoaded {len(class_indices)} classes: {class_indices}")

        # Step 3: Generate composites from SEPARATE source pools
        print(f"\nGenerating TRAIN composites (from train source images, scale_factor={self.scale_factor})...")
        idx = 0
        # 10 samples per pair for training (massive data scale)
        train_samples = 10 * self.scale_factor
        idx = self._generate_composites_for_split(train_images, "train", train_samples, idx)
        train_count = idx
        
        print(f"Generated {train_count} training composites.")
        
        print("Generating TEST composites (from test source images - ZERO overlap)...")
        # 8 samples per pair for testing (produces 1120 test stimuli for scale=1)
        test_samples = 8 * self.scale_factor
        idx = self._generate_composites_for_split(test_images, "test", test_samples, idx)
        test_count = idx - train_count
        
        print(f"Generated {test_count} test composites.")

        meta_path = os.path.join(self.output_dir, "metadata.json")
        with open(meta_path, "w") as f:
            json.dump(self.metadata, f, indent=4)
        print(f"\nTotal: {idx} composite stimuli with REAL ImageNet labels.")
        print(f"  Train: {train_count} (from {sum(len(v) for v in train_images.values())} source images)")
        print(f"  Test:  {test_count} (from {sum(len(v) for v in test_images.values())} source images)")
        print(f"  Source image overlap between splits: ZERO")
        print(f"Metadata saved to {meta_path}")


if __name__ == "__main__":
    import os
    import argparse
    
    parser = argparse.ArgumentParser(description="Generate CompositeVision Dataset")
    parser.add_argument("--scale-factor", type=int, default=1, 
                        help="Multiplier for dataset size (default 1 = 1120 test stimuli, 10 = 11200)")
    args = parser.parse_args()
    
    base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    generator = CompositeDatasetGenerator(
        output_dir=os.path.join(base_dir, "dataset"),
        scale_factor=args.scale_factor
    )
    generator.generate()
