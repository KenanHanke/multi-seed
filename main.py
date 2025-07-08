#!/usr/bin/env python3

import functools
import logging
import operator
import numpy as np
from config import Config
from dataset import Dataset
from image import Image, Mask
from reference import ReferenceBuilder
import tift
from feature_mapper import name_to_mapper_class
import os


def exec_config(config: Config, rng=None):
    parallel_io: bool = config.params["PARALLEL_IO"]
    results_folder = config.params["RESULTS_FOLDER"]
    if results_folder:
        os.makedirs(results_folder, exist_ok=True)

    # prepare the dataset loaders for each cohort
    cohort_loaders = {
        cohort: tift.load_datasets(paths, asynchronous=parallel_io)
        for cohort, paths in config.cohorts.items()
    }

    # get mask
    all_dataset_loader = functools.reduce(operator.add,
                                          cohort_loaders.values())
    mask = all_dataset_loader.extract_mask()

    # prepare reference builder
    reference_builder = ReferenceBuilder(radius=config.params["SEED_RADIUS"],
                                         mask=mask)
    reference_builder.sample(config.params["N_SEEDS"], rng)

    if results_folder:
        # create visualization of reference builder
        visualization = reference_builder.visualized().normalized()
        visualization = visualization.scaled(256).converted(np.uint16)
        visualization_folder = os.path.join(results_folder,
                                            "seed_visualization")
        os.makedirs(visualization_folder, exist_ok=True)
        visualization_path = os.path.join(visualization_folder,
                                          "visualization.img.z")
        tift.save_image(visualization, visualization_path)

    # initialize and fit mapper
    N_FEATURES = config.params["N_FEATURES"]
    REDUCTION_ALGORITHM = config.params["REDUCTION_ALGORITHM"]
    Mapper = name_to_mapper_class(REDUCTION_ALGORITHM)
    mapper = Mapper(N_FEATURES, reference_builder)
    mapper.fit(all_dataset_loader,
               samples_per_dataset=config.params["N_SAMPLES_PER_DATASET"],
               rng=rng)

    # transform the datasets
    cohort_result_dirs = {}
    for cohort, loader in cohort_loaders.items():
        cohort_result_dirs[cohort] = []
        for dataset, path in zip(loader, loader.paths):
            result = mapper.transform(dataset)

            res_dir = os.path.join(path, REDUCTION_ALGORITHM)
            raw_res_dir = os.path.join(res_dir, "raw_float_result")
            os.makedirs(raw_res_dir, exist_ok=True)

            for i, image in enumerate(result):
                image.save(os.path.join(raw_res_dir, f"{i}.npz"))

            cohort_result_dirs[cohort].append(res_dir)

    # flatten the list of result directories
    res_dirs = sum(cohort_result_dirs.values(), [])

    # normalize the results to floats in [0, 1] and save them
    for i in range(N_FEATURES):

        def image_iter_func():
            for res_dir in res_dirs:
                yield Image.load(
                    os.path.join(
                        res_dir,
                        "raw_float_result",
                        f"{i}.npz",
                    ))

        normalized_images = Image.normalize_all_lazy(image_iter_func)
        for res_dir, image in zip(res_dirs, normalized_images):
            norm_res_dir = os.path.join(res_dir, "normalized_float_result")
            os.makedirs(norm_res_dir, exist_ok=True)
            image.save(os.path.join(norm_res_dir, f"{i}.npz"))

    # also save the normalized results as ints scaled to [0, 4095]
    # using the tift compatible format; 4095 is the maximum value
    # that can be stored in a 12 bit image, which is the standard
    for res_dir in res_dirs:
        tift_res_dir = os.path.join(res_dir, "tift_normalized_scaled")
        os.makedirs(tift_res_dir, exist_ok=True)

        first = True
        for i in range(N_FEATURES):
            # load the normalized float result
            image = Image.load(
                os.path.join(
                    res_dir,
                    "normalized_float_result",
                    f"{i}.npz",
                ))

            # scale to [0, 4095]
            image = image.scaled(4095).converted(np.uint16)

            if first:
                # initialize the dataset
                first = False
                dimensions = image.dimensions
                dtype = image.dtype
                dataset = Dataset(dimensions, N_FEATURES, dtype=dtype)

            # put the image into the dataset
            dataset[i] = image

        tift.save_dataset(dataset, tift_res_dir)

    ####################################################################
    ####################################################################
    #        THIS IS WHERE GROUP COMPARISON WILL HAPPEN!!!!!!!!        #
    ####################################################################
    ####################################################################


def main():
    logging.basicConfig(format="[%(levelname)s %(asctime)s] %(message)s",
                        level=logging.DEBUG)

    # suppress numba's excessive debug messages
    numba_logger = logging.getLogger("numba")
    numba_logger.setLevel(logging.WARNING)

    # ensure reproducibility of random numbers
    rng = np.random.default_rng(seed=42)

    # # dataset = tift.load_dataset(PATH)
    # # dataset.save("dataset.npz")
    # dataset = Dataset.load("dataset.npz")
    # # mask = dataset.extract_mask()
    # # mask.save("mask.npz")
    # mask = Mask.load("mask_combined.npz")

    # reference_builder = ReferenceBuilder(radius=20, mask=mask)
    # reference_builder.sample(N, rng)
    # visualization = reference_builder.visualized().normalized()
    # visualization = visualization.scaled(256).converted(np.uint16)
    # tift.save_image(visualization, "visualization.img.z")
    # reference = reference_builder.build(dataset)

    # pca = PCAMapper(NUM_OF_PRINCIPAL_COMPONENTS, reference)
    # pca.fit([dataset],
    #         samples_per_dataset=SIZE_OF_COMPARISON_POINT_SAMPLE,
    #         rng=rng)
    # result = pca.transform(dataset)
    # tift.save_dataset(result, os.path.join(PATH, "pca"))

    with open("pdstudy.config", "r") as f:
        config_str = f.read()

    config = Config.from_str(config_str)

    exec_config(config, rng)


if __name__ == "__main__":
    main()
