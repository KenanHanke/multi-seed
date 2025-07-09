#!/usr/bin/env python3

import functools
import logging
import operator
import numpy as np
from config import Config
from dataset import Dataset
from image import Image, Mask
from reference import ReferenceBuilder
import io_utils
from feature_mapper import name_to_mapper_class
import os



# suppress numba's excessive debug messages
numba_logger = logging.getLogger("numba")
numba_logger.setLevel(logging.WARNING)


def exec_config(config: Config, rng=None):
    parallel_io: bool = config.params["PARALLEL_IO"]
    results_folder = config.params["RESULTS_FOLDER"]
    if results_folder:
        os.makedirs(results_folder, exist_ok=True)

    # prepare the dataset loaders for each cohort
    cohort_loaders = {
        cohort: io_utils.load_datasets_lazy(paths, asynchronous=parallel_io)
        for cohort, paths in config.cohorts.items()
    }

    # get mask
    all_dataset_loader = functools.reduce(operator.add,
                                          cohort_loaders.values())
    mask = all_dataset_loader.extract_mask()
    if results_folder:
        # save the mask to the results folder
        mask_path = os.path.join(results_folder, "mask.img")
        reformatted_mask = mask.converted(np.int16).scaled(255)
        io_utils.save_image(reformatted_mask, mask_path)

    # prepare reference builder
    reference_builder = ReferenceBuilder(radius=config.params["SEED_RADIUS"],
                                         mask=mask)
    reference_builder.sample(config.params["N_SEEDS"], rng)

    if results_folder:
        # create visualization of reference builder
        visualization = reference_builder.visualized().normalized().scaled(1023).converted(np.int16)
        visualization_path = os.path.join(results_folder,
                                          "seed_visualization.img")
        io_utils.save_image(visualization, visualization_path)

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


def main():
    logging.basicConfig(format="[%(levelname)s %(asctime)s] %(message)s",
                        level=logging.DEBUG)

    # ensure reproducibility of random numbers
    rng = np.random.default_rng(seed=42)

    with open("pdstudy.config", "r") as f:
        config_str = f.read()

    config = Config.from_str(config_str)

    exec_config(config, rng)


if __name__ == "__main__":
    main()
