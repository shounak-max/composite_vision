# CompositeVision Research: Methodology Analysis & Experimental Design Critique

This document provides a detailed breakdown of the experimental methodology used in the CompositeVision framework, specifically detailing how the dataset is generated, how the various models are evaluated, and a critical analysis of a fundamental flaw in the experimental comparison.

## 1. Dataset Generation Methodology

The dataset evaluated in this project is not a standard dataset (nor is it the `noisy_imagenette.csv`), but a synthetically generated benchmark called the **CompositeVision** dataset.

*   **Source Data:** The pipeline starts with [Imagenette](https://github.com/fastai/imagenette), a 10-class subset of the real ImageNet dataset.
*   **Data Split:** The raw Imagenette images are split into a Train pool (70%) and a Test pool (30%). This split occurs *before* any composition takes place, ensuring **zero source-image overlap** between training and testing.
*   **Composite Generation:** The script (`src/data/generation.py`) pairs images from different classes and blends them using 7 different composition techniques at two salience levels (high and low). The techniques include:
    *   **AdaIN:** Style transfer of texture.
    *   **Texture-Shape FFT Swap:** Swapping phase (shape) and amplitude (texture) in the frequency domain.
    *   **Occlusion:** Black patches overlaid on the image.
    *   **Patch Shuffle, Color Inversion, Edge Conflict, Superimposition.**
*   **Labels:** The generated composites are saved to `dataset/images/`. Their ground-truth labels are stored in `dataset/metadata.json`, which records the primary shape class (`class1`) and the conflicting texture/style class (`class2`) for each image.

## 2. Model Evaluation Methodology

The framework evaluates three distinct classes of models on the test split of the CompositeVision dataset. The metric of success is how often a model correctly predicts the primary shape class (`class1`).

### A. Standard Feedforward Baselines
*   **Models:** ResNet50, ResNet101, ConvNeXt, ViT-B/16, DeiT, CLIP (ViT-B/32).
*   **Training Data:** **None from this dataset.** These models use off-the-shelf, pre-trained weights from ImageNet-1K (e.g., `IMAGENET1K_V1` or LAION for CLIP). 
*   **Evaluation:** They are evaluated completely **zero-shot** on the composite images.

### B. Vision-Language Models (VLMs)
*   **Models:** GPT-4o, Gemini.
*   **Training Data:** Pre-trained foundation models.
*   **Evaluation:** Evaluated zero-shot using a direct prompting method via API (`src/experiments/vlm_evaluation.py`).

### C. Recurrent RL Attention Agent
*   **Model:** A custom architecture defined in `src/models/rl_attention.py` that extracts multi-scale "glimpses" of the image and uses an LSTM core to decide where to look next.
*   **Training Data:** **The generated CompositeVision Train split.** It uses a frozen ResNet50 backbone but fine-tunes its final convolutional layer (`layer4`) and trains its recurrent attention policy from scratch using Proximal Policy Optimization (PPO) / Advantage Actor-Critic (A2C).
*   **Evaluation:** Evaluated on the test split.

---

> [!WARNING]
> ## 3. Critical Flaw in the Experimental Design
> 
> The core conclusion of `experiment_results.md` claims that the RL Attention Agent achieves significantly higher shape bias and overall accuracy (63.21%) compared to feedforward models (like ResNet50 at 33.93%) due to its recurrent foveated architecture. 
> 
> **However, this comparison is scientifically unfair due to a massive discrepancy in training supervision.**

### The Source of the Flaw
The unfairness stems from how the RL Attention Agent was supervised compared to the baselines:

1.  **Explicit Target-Domain Supervision vs. Zero-Shot:** 
    In `src/experiments/runner.py`, the RL agent receives a positive reward **only when it correctly predicts `class1` (the shape class)** on the training composites. The agent is explicitly and heavily optimized to ignore texture/color/style (`class2`) and focus purely on shape. 
    Conversely, the baseline models (ResNets, ViTs) have only been trained on standard, natural ImageNet images where shape and texture are naturally correlated. They were never taught to prioritize shape over texture when presented with an unnatural conflict.
2.  **Distribution Shift:** 
    The RL Agent is trained directly on the exact visual distribution of the composite images (e.g., AdaIN artifacts, FFT swaps, black occlusion patches). The baselines have never seen these specific image corruptions during their standard ImageNet training.

### Conclusion
The RL agent's superior performance is not definitively proof that "recurrent foveated attention architectures inherently process shape better than CNNs/Transformers." Instead, it proves that **a model explicitly trained to prioritize shape on a specific set of visual corruptions will perform better than a zero-shot model that has never seen those corruptions.**

### Recommendations for a Fair Benchmark
To make this a mathematically and scientifically fair architectural comparison, the methodology must be updated to either:
1.  **Level Up the Baselines:** Fine-tune all baseline models (ResNet, ViT, ConvNeXt) on the exact same `train` split of composite images using standard Cross-Entropy loss against `class1`, and then compare their test performance against the RL agent.
2.  **Level Down the RL Agent:** Train the RL agent *only* on the standard, uncorrupted ImageNet dataset (or the raw Imagenette dataset) without explicitly teaching it to ignore texture conflicts, and then evaluate it zero-shot on the composites.
