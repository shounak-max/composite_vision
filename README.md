# CompositeVision

A multi-artifact-type cue-conflict benchmark for measuring **shape-bias inflation**: the gap
between what a single ratio (the Geirhos shape-bias ratio) reports and what a model actually
does, decomposed into shape accuracy, texture accuracy and a **Neither Rate**.

Companion code and data for the CVIP 2026 paper *CompositeVision: Exposing Shape Bias Inflation
Through Multi-Artifact Cue-Conflict Benchmarking* (paper ID 1261).

## Download

The stimulus archive is served from [Releases](../../releases/tag/v1.0) so you do not have to
clone the whole repository:

| Archive | Contents | Size |
|---|---|---|
| [compositevision-test-v1.0.zip](../../releases/download/v1.0/compositevision-test-v1.0.zip) | 11,200 test stimuli + `metadata.json` | ~820 MB |
| [compositevision-train-v1.0.zip](../../releases/download/v1.0/compositevision-train-v1.0.zip) | 14,000 training stimuli + `metadata.json` | ~1.03 GB |
| [SHA256SUMS.txt](../../releases/download/v1.0/SHA256SUMS.txt) | checksums for both | 1 KB |

To reproduce the paper's numbers you need the test archive only. Each archive contains its own
copy of the full `metadata.json`, so `split` tells you which half you are looking at.

## What is in the benchmark

| | |
|---|---|
| Composition types | 7 — AdaIN texture transfer, partial occlusion, superimposition, FFT texture–shape swap, edge conflict, patch shuffle, colour inversion |
| Salience levels | 2 per type (`high`, `low`), 1,600 test stimuli per type |
| Source classes | 10 Imagenette classes, mapped to their native ImageNet-1K indices |
| Test split | 11,200 composites (7 x 2 x 10 x 80) |
| Training split | 14,000 composites from disjoint source images (70/30 source-image split) |
| Model configurations | 15 run end to end, 13 reported quantitatively |
| Score space | full 1,000-class ImageNet-1K logits; the 10 benchmark classes are read from their native indices |

The Neither Rate is computed only over stimuli that carry a second class. Occlusion uses a single
source image and patch shuffle permutes one image's own patches, so both are excluded from that
metric but retained for overall accuracy. Colour inversion does blend a second source image, so it
stays in the decomposition.

## Repository layout

```
dataset/
  images/            25,200 composite stimuli (11,200 test + 14,000 train)
  metadata.json      per-stimulus record: id, filename, class1, class2,
                     composition_type, salience, split
  imagenette2-160/   source images (see DATASHEET.md for licensing)
results/             benchmark CSVs, bootstrap CIs, statistical tests, figures
src/data/            stimulus generation (generation.py)
src/models/          feedforward baselines, CLIP wrappers, RL foveation agent
src/metrics/         Neither Rate, error consistency, bootstrap CIs, significance tests
src/experiments/     end-to-end runner, fine-tuning, VLM evaluation, RL training,
                     and the ablation suite (src/experiments/ablation_study.py)
```

## Quickstart

```bash
conda env create -f environment.yml     # or: pip install -r requirements.txt
conda activate compositevision

python src/experiments/runner.py                 # full benchmark pipeline
python src/experiments/finetune_baselines.py     # fine-tuned configurations
python src/experiments/train_rl_imagenette.py    # RL foveation agent (A2C)
python src/experiments/vlm_evaluation.py         # CLIP prompt templates
python src/experiments/ablation_study.py         # pathway / glimpse / sequence ablations
```

`runner.py` regenerates the stimulus archive deterministically from the seeded source split and
writes everything under `results/`. The released archive is already generated, so no model needs
to rerun generation to reproduce the paper's numbers.

## Data and reproducibility

- Stimulus archive: [`dataset/`](dataset) — 11,200 test composites plus the 14,000-composite training pool.
- Per-stimulus labels: [`dataset/metadata.json`](dataset/metadata.json) — the `split` field separates train from test; `class2` is `-1` where a stimulus has no second class.
- Results: [`results/benchmark_results.csv`](results) — per-stimulus predictions for every configuration, the source of every number in the paper.
- Provenance, construction parameters, licensing and intended use: [`DATASHEET.md`](DATASHEET.md).
- Seeds, determinism and known variance: [`REPRODUCIBILITY_GUIDE.md`](REPRODUCIBILITY_GUIDE.md).

## License

Code is released under the MIT License (see [`LICENSE`](LICENSE)). The stimuli are derived from
ImageNet through Imagenette and are subject to the source images' terms; see
[`DATASHEET.md`](DATASHEET.md) before redistributing them.

## Citation

```bibtex
@inproceedings{compositevision2026,
  author    = {Biswas, Soham and Banerjee, Shounak and Joshi, Piyush},
  title     = {CompositeVision: Exposing Shape Bias Inflation Through Multi-Artifact Cue-Conflict Benchmarking},
  booktitle = {Proceedings of the Conference on Computer Vision and Image Processing (CVIP)},
  year      = {2026}
}
```
