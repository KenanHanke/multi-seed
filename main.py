from dataset import Dataset
from image import Image, Mask
import logging
import numpy as np


PATH = "/home/khanke/data/HypoPark/hypoPark/00CONVERTED/CON/REDACTED/fmri301/MNINorm/new_PreProcAll"
NUM_OF_PRINCIPAL_COMPONENTS = 3
N = 200
SIZE_OF_COMPARISON_POINT_SAMPLE = 2000


def main():
    logging.basicConfig(level=logging.INFO)

    # ensure reproducibility of random numbers
    rng = np.random.default_rng(seed=42)

    dataset = Dataset.load_tift(PATH)
    mask = dataset.extract_mask()
    # print("Sampling reference points...")
    # reference_points = sample_coords(N, dataset)
    # print("Creating projection...")
    # projection = create_projection(NUM_OF_PRINCIPAL_COMPONENTS,
    #                                reference_points, SIZE_OF_COMPARISON_POINT_SAMPLE, dataset)
    # print("Calculating results...")
    # results = calculate_results(projection, reference_points, dataset)
    # print("Saving results...")
    # save_results(results, PATH, np.uint8)


if __name__ == '__main__':
    main()
