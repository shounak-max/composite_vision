# CompositeVision Prototype

This repository implements the first working prototype of the CompositeVision framework. It is designed to facilitate research on how feedforward and recurrent vision models (including Visual Language Models like CLIP) process composite image stimuli.

## Features
- **Benchmark Creation**: Generates 400 composite stimuli based on 50 ImageNet categories, applying AdaIN blending, partial occlusion, superimposition, and texture-shape cue conflicts.
- **Model Benchmarking**: Automated evaluation for a suite of models including ResNet50, ResNet101, ConvNeXt, ViT-B/16, DeiT, and CLIP ViT-B/32.
- **Error Consistency**: Calculates model-model, composition-wise, and salience-wise error consistency metrics.
- **RL Iterative Attention**: Implements a recurrent attention agent simulating cortical feedback, trained via REINFORCE/PPO.

## Structure
- `src/data/`: Dataset generation logic.
- `src/models/`: Baseline feedforward models and the RL attention agent.
- `src/metrics/`: Error consistency evaluation and report generation.
- `src/experiments/`: The experiment runner orchestrating the end-to-end pipeline.
