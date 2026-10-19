# `dataset/` - CompositeVision stimuli

This folder holds 25,200 composite images (11,200 test + 14,000 train) and the label file
`metadata.json`. Because it is far too large to browse here, the same archive is published as two
release assets:

- **[compositevision-test-v1.0.zip](https://github.com/shounak-max/composite_vision/releases/download/v1.0/compositevision-test-v1.0.zip)** - 11,200 test stimuli (~820 MB). This is the half needed to reproduce the paper's reported numbers.
- **[compositevision-train-v1.0.zip](https://github.com/shounak-max/composite_vision/releases/download/v1.0/compositevision-train-v1.0.zip)** - 14,000 training stimuli (~1.03 GB).
- **[SHA256SUMS.txt](https://github.com/shounak-max/composite_vision/releases/download/v1.0/SHA256SUMS.txt)** - checksums for both.

See the [release page](https://github.com/shounak-max/composite_vision/releases/tag/v1.0) for the
file list.

## Layout

```
images/            <id>_<type>_<salience>.png, e.g. 00000_adain_high.png
metadata.json      one record per stimulus:
                   id, filename, class1, class2, composition_type, salience, split
imagenette2-160/   source photographs (not part of the release)
```

## Using the labels

`split` is `test` or `train`; the two pools share no source photographs. `class1` is the
shape-consistent ImageNet-1K index and `class2` the texture-consistent one.

Two artifact types have no usable texture class and must be excluded from any Neither Rate or
Geirhos-ratio computation:

- **occlusion** - `class2` is `-1`; there is no second object.
- **patch shuffle** - `class2` records a randomly drawn class that is *not present in the image*.
  The stimulus is one source image with its own patches permuted.

Colour inversion *does* blend a second source image, so its `class2` is genuine.

See [`../DATASHEET.md`](../DATASHEET.md) for construction parameters, provenance and licensing.
