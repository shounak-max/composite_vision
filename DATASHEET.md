# Datasheet — CompositeVision

Following the *Datasheets for Datasets* template. This is the document promised in the paper's
Reproducibility section.

## Motivation

CompositeVision exists to audit a measurement rather than a model. The Geirhos shape-bias ratio
is computed only over stimuli a model assigns to either the shape class or the texture class, so
it is silent about how often a model commits to neither. The dataset provides a controlled,
multi-artifact-type stimulus set on which that quantity — the **Neither Rate** — is measurable
per artifact type, under a matched zero-shot / fine-tuned evaluation protocol.

## Composition

| Property | Value |
|---|---|
| Instances | 25,200 composite images: 11,200 test, 14,000 train |
| Per test instance | one composite, two source classes (or one, where no second class exists) |
| Composition types | 7, 1,600 test instances each |
| Salience levels | `high`, `low`; 5,600 test instances each |
| Source classes | 10 Imagenette classes, mapped to native ImageNet-1K indices |
| Labels | `metadata.json`: `id`, `filename`, `class1`, `class2`, `composition_type`, `salience`, `split` |
| Missing data | `class2 = -1` for occlusion, which has no second source object |
| Image format | PNG, generated at 160 px source resolution |
| Predictions | `results/benchmark_results.csv`: per-stimulus top-1/top-5 and confidence for 15 configurations |

Composition types and their construction parameters:

- **AdaIN texture transfer** — style blend alpha 0.8 (`high`) / 0.4 (`low`)
- **Partial occlusion** — 15 (`high`) / 5 (`low`) patches
- **Superimposition** — blend alpha 0.7 / 0.3
- **FFT texture–shape swap** — shape-retention alpha 0.3 / 0.8 (lower alpha retains less source shape, so for this type alone lower alpha means *higher* salience)
- **Edge conflict** — blend alpha 0.6 / 0.3
- **Patch shuffle** — 4x4 grid, half shuffled (`high`); 8x8, quarter shuffled (`low`); a spatial-layout disruption control that preserves local patch content
- **Colour inversion** — blend alpha 0.7 / 0.4, blended with a second source image

## Collection process

Stimuli are synthesised, not collected from people. Source photographs come from Imagenette
(`imagenette2-160`, a 10-class ImageNet subset). The 70/30 source-image split is applied at the
source level, so no source photograph contributes to both the training and test pools; all 25,200
composites are then generated deterministically from a fixed seed. `src/data/generation.py`
contains the generator. No human subjects were involved at any stage.

## Preprocessing, cleaning and labels

`class1` is the shape-consistent class and `class2` the texture-consistent class. For occlusion no
second object is present, so `class2 = -1`. For patch shuffle the recorded `class2` is a random
other class that is *not present in the image*; downstream metrics must therefore exclude patch
shuffle from the Neither Rate exactly as they exclude occlusion, which is what the paper does.
Colour inversion does blend a second source, so its `class2` is that source's class. All classes
are stored as native ImageNet-1K indices so any ImageNet-trained model can be scored without
remapping.

## Uses

**Intended.** Auditing single-ratio shape-bias metrics; measuring per-artifact-type shape and
texture accuracy; comparing zero-shot and fine-tuned behaviour on identical stimuli; benchmarking
recurrent or glimpse-limited architectures against feedforward baselines.

**Inappropriate.** The stimuli are synthetic composites, not natural images: they should not be
used as a measure of general visual capability, nor as a substitute for human psychophysical
study. This release contains **no human baseline**, so no claim about human versus machine shape
bias can be supported by these files alone. The 10-class source taxonomy is not representative of
full ImageNet-1K or fine-grained recognition, and models trained on these composites acquire a
restricted 10-class output prior that affects the Neither Rate.

## Packaging

The archive is published as two release assets so that a download does not require cloning the
repository: `compositevision-test-v1.0.zip` (11,200 test stimuli) and
`compositevision-train-v1.0.zip` (14,000 training stimuli). Both carry the full `metadata.json`,
the README, this datasheet and the license. SHA-256 checksums are published alongside them. The
source photographs in `dataset/imagenette2-160/` are not part of the release; obtain them from
Imagenette.

## Distribution and licensing

- **Code** (`src/`, scripts, docs): MIT License — see `LICENSE`.
- **Source photographs** (`dataset/imagenette2-160/`): redistributed as the Imagenette subset of ImageNet. Imagenette is published by fastai; the underlying photographs remain subject to ImageNet's terms of access, which permit non-commercial research use.
- **Constructed composites** (`dataset/images/`): derived from those photographs, so they inherit the same terms. They are released here for research use. Redistribution outside research contexts, or any commercial use, requires checking the terms of the original ImageNet images.
- **Metadata and results** (`dataset/metadata.json`, `results/`): released under the same terms as the code.

Users are responsible for complying with the source-image terms. Where a downstream use requires
a cleanly licensed copy, regenerate the stimuli from source images obtained under the appropriate
terms using `src/data/generation.py`.

## Maintenance

Maintained by the paper's authors (Soham Biswas, Shounak Banerjee, Piyush Joshi, IIIT Sri City).
The released archive is fixed to the version used in the paper; corrections will be noted here and
in the repository commit history rather than silently replacing stimuli.
