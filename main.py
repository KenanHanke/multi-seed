#!/usr/bin/env python3

import logging
import numpy as np
from glob import glob

from dataset import Dataset
from image import Image, Mask
from reference import ReferenceBuilder, Reference
import tift

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

    reference_builder = ReferenceBuilder(radius=10, mask=mask)
    reference_builder.sample(N, rng)
    reference = reference_builder.build(dataset)

    # --------- POSSIBLE INTERFACE --------- #
    # projection = PCA()
    # projection.calibrate(dataset, reference)
    #

    ##########################################
    ### NOT YET IMPLEMENTED IN NEW VERSION ###
    ##########################################
    # print("Creating projection...")
    # projection = create_projection(NUM_OF_PRINCIPAL_COMPONENTS,
    #                                reference_points, SIZE_OF_COMPARISON_POINT_SAMPLE, dataset)
    # print("Calculating results...")
    # results = calculate_results(projection, reference_points, dataset)
    # print("Saving results...")
    # save_results(results, PATH, np.uint8)


if __name__ == "__main__":
    main()
