#!/usr/bin/env python3

import logging
import numpy as np
from glob import glob
from dataset import Dataset
from image import Image, Mask
from reference import ReferenceBuilder, Reference
import tift
from feature_mapper import UntranslatedPCA
import os

PATH = "/home/khanke/data/HypoPark/hypoPark/00CONVERTED/CON/REDACTED/fmri301/MNINorm/new_PreProcAll"
NUM_OF_PRINCIPAL_COMPONENTS = 3
N = 200
SIZE_OF_COMPARISON_POINT_SAMPLE = 2000


def main():
    logging.basicConfig(format="[%(levelname)s %(asctime)s] %(message)s",
                        level=logging.DEBUG)

    # suppress numba's excessive debug messages
    numba_logger = logging.getLogger("numba")
    numba_logger.setLevel(logging.WARNING)

    # ensure reproducibility of random numbers
    rng = np.random.default_rng(seed=42)

    # dataset = tift.load_dataset(PATH)
    # dataset.save("dataset.npz")
    dataset = Dataset.load("dataset.npz")
    # mask = dataset.extract_mask()
    # mask.save("mask.npz")
    mask = Mask.load("mask_combined.npz")

    reference_builder = ReferenceBuilder(radius=20, mask=mask)
    reference_builder.sample(N, rng)
    visualization = reference_builder.visualized().normalized()
    visualization = visualization.scaled(256).converted(np.uint16)
    tift.save_image(visualization, "visualization.img.z")
    reference = reference_builder.build(dataset)

    pca = UntranslatedPCA(NUM_OF_PRINCIPAL_COMPONENTS)
    pca.fit([dataset],
            reference,
            samples_per_dataset=SIZE_OF_COMPARISON_POINT_SAMPLE,
            rng=rng)
    result = pca.transform(dataset, reference)
    tift.save_dataset(result, os.path.join(PATH, "untranslated_pca"))


if __name__ == "__main__":
    main()
