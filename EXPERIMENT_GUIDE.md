# CompositeVision Experiment Guide

## Experiment Overview
This repository implements a controlled experiment to evaluate how different neural network architectures process ambiguous, composite visual stimuli. 

The pipeline performs four main tasks:
1. **Dataset Generation:** We construct a custom benchmark of 400 "composite stimuli" using 50 base images. We blend conflicting image pairs using four methods: AdaIN (style matching), Partial Occlusion, Superimposition, and Texture-Shape conflict (via FFT).
2. **Feedforward Baselines:** We evaluate State-of-the-Art pretrained feedforward models—including CNNs (ResNet, ConvNeXt) and Vision Transformers (ViT, DeiT, CLIP)—on these composite images to establish how standard architectures resolve visual conflicts.
3. **Recurrent RL Attention:** We train a custom Reinforcement Learning agent (mimicking human cortical feedback). Instead of processing the whole image at once, it takes sequential spatial "glimpses" (saccades) of the image, updating a recurrent memory state before making a final prediction.
4. **Error Consistency Analysis:** We calculate the *Error Consistency* between all models. This metric reveals if two distinct models make the *exact same mistakes*. By comparing the RL agent against CNNs and ViTs, we can quantify if recurrent attention resolves visual ambiguity differently than standard feedforward processing.

---

## Execution Instructions
To run the complete experimental pipeline:

1. **Install Dependencies**
   `pip install -r requirements.txt`
   or
   `conda env create -f environment.yml`

2. **Run the Full Pipeline**
   `python -m src.experiments.runner`
   
   This script will sequentially:
   - Generate the dataset of composite stimuli in `dataset/`
   - Evaluate feedforward baselines and save results in `results/`
   - Train the RL Attention agent and evaluate it.
   - Generate error consistency reports and figures in `results/`.

3. **Analyzing Results**
   All numerical data and plots (heatmaps, bar charts) are saved in the `results/` directory.
