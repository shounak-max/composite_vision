# CompositeVision — project notes

Implementation notes for the CompositeVision benchmark. The entry point for new readers is
[`README.md`](README.md).

## What the pipeline does

- **Stimulus generation** (`src/data/generation.py`): 11,200 evaluation stimuli from 10 Imagenette
  classes, crossing 7 composition types (AdaIN texture transfer, partial occlusion,
  superimposition, FFT texture–shape swap, edge conflict, patch shuffle, colour inversion) with 2
  salience levels and 80 exemplars per condition, plus a disjoint 14,000-stimulus training pool.
  Source classes are mapped to their native ImageNet-1K indices so any ImageNet-pretrained model
  can be scored in the full 1,000-class space.
- **Model benchmarking** (`src/experiments/runner.py`): ResNet-50, ResNet-101, ConvNeXt, ViT-B/16,
  DeiT, CLIP under three prompt templates, and the recurrent RL foveation agent, each evaluated
  zero-shot and, where applicable, fine-tuned.
- **Metrics** (`src/metrics/`): the Neither Rate decomposition (shape / texture / neither), pairwise
  error consistency, bootstrap confidence intervals, and two-proportion significance tests.
- **RL foveation agent** (`src/models/rl_attention.py`): sequential 48x48 glimpses over T=8 steps,
  a ResNet-50 encoder frozen except its final residual block, an LSTM core, and actor/value/
  classifier heads trained with A2C.

## Structure

- `src/data/` — dataset generation.
- `src/models/` — baselines, CLIP wrappers, RL agent.
- `src/metrics/` — Neither Rate, error consistency, CIs, significance tests.
- `src/experiments/` — runner, fine-tuning, VLM evaluation, RL training, ablation suite
  (`src/experiments/ablation_study.py`).

## Status

The released `results/` directory contains the completed 13-configuration comparison. The
pathway-decoupling ablation is reported as a preliminary estimate; the random-glimpse control and
the sequence-length / glimpse-size sweeps are implemented but not yet run.
