# multiseed

`multiseed` runs unsupervised group-level resting-state fMRI network extraction using a many-seed-based approach combined with dimensionality reduction. It's available on [PyPI](https://pypi.org/project/multiseed/).

In practice, it:

1. loads multiple rs-fMRI datasets grouped into cohorts,
2. builds one combined brain mask from all non-zero voxels,
3. samples random seed locations inside that mask,
4. turns each voxel into a vector of seed-correlation values,
5. reduces those vectors with Factor Analysis, PCA or ICA, and
6. writes one 3D map per extracted network/component for every dataset.

Because the mask, seed set, and reduction model are built jointly across all cohorts, the resulting feature maps are directly comparable across subjects/datasets.

## Quick start

Install using pip:

```bash
pip install multiseed
```

Generate a template config (which contains several explanatory comments):

```bash
python -m multiseed --write-default-config analysis.cfg
```

Edit `analysis.cfg`, then run the analysis:

```bash
python -m multiseed analysis.cfg
```

If you installed a console entry point (default on most systems), the same command is:

```bash
multiseed analysis.cfg
```

The CLI uses a fixed random seed, so repeated runs with the same inputs/config are reproducible.

## Programmatic use

The `multiseed` library exposes an extensive API encompassing almost all parts of its algorithm. See the [documentation](https://kenanhanke.github.io/multiseed) for more details.

## Simple config example

```ini
N_SEEDS = 1000
SEED_RADIUS = 10
REDUCTION_ALGORITHM = FactorAnalysis
RESULTS_FOLDER = results

[Controls]
/data/controls/sub01
/data/controls/sub02

[Patients]
/data/patients/*
```

Each dataset is one folder containing the 3D volumes from a single scan. Supported image formats are:

- `.img`, `.img.gz`, `.img.z`
- `.nii`, `.nii.gz`

A cohort (e.g. `Controls` or `Patients` in this example) is a named list of dataset folders.

See the template config for more parameters and information.

## What gets written

Resulting networks are written to the `RESULTS_FOLDER`, organized by cohort and dataset. The exact structure is:

```text
results/
  mask.hdr.gz
  mask.img.gz
  seed_visualization.hdr.gz
  seed_visualization.img.gz
  network_001/
    average_visualization.hdr.gz
    average_visualization.img.gz
    <Cohort Name>/
      result_for_dataset_000001.hdr.gz
      result_for_dataset_000001.img.gz
```

## Notes

- All datasets are assumed to have matching spatial dimensions.
- The combined mask is built from voxels that are nonzero in at least one dataset.
- The exported average visualizations are display-oriented summaries; for downstream numeric work, use the saved float results.
