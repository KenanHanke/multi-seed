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
    tift_compatibility_mode = config.params["TIFT_COMPATIBILITY_MODE"]
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
        mask_path = os.path.join(results_folder, "mask.img.gz")
        reformatted_mask = mask.converted(np.float32).scaled(255).converted(np.int16)
        io_utils.save_image(reformatted_mask, mask_path)

    # prepare reference builder
    reference_builder = ReferenceBuilder(radius=config.params["SEED_RADIUS"],
                                         mask=mask)
    reference_builder.sample(config.params["N_SEEDS"], rng)

    if results_folder:
        # create visualization of reference builder
        visualization = reference_builder.visualized().normalized().scaled(4095).converted(np.int16)
        visualization_path = os.path.join(results_folder,
                                          "seed_visualization.img.gz")
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
    cohort_internal_result_dirs = {}
    for cohort, loader in cohort_loaders.items():
        cohort_internal_result_dirs[cohort] = []
        for dataset, path in zip(loader, loader.paths):
            result = mapper.transform(dataset)

            internal_result_dir = os.path.join(path, REDUCTION_ALGORITHM)
            raw_res_dir = os.path.join(internal_result_dir, "raw_float_result")
            os.makedirs(raw_res_dir, exist_ok=True)

            for i, image in enumerate(result):
                image.save(os.path.join(raw_res_dir, f"{i}.npz"))

            cohort_internal_result_dirs[cohort].append(internal_result_dir)

    # flatten the list of result directories
    internal_result_dirs = sum(cohort_internal_result_dirs.values(), [])

    # normalize the results to floats in [0, 1] and save them
    for i in range(N_FEATURES):

        def image_iter_func():
            for res_dir in internal_result_dirs:
                yield Image.load(
                    os.path.join(
                        res_dir,
                        "raw_float_result",
                        f"{i}.npz",
                    ))

        normalized_images = Image.normalize_all_lazy(image_iter_func)
        for internal_result_dir, image in zip(internal_result_dirs, normalized_images):
            norm_res_dir = os.path.join(internal_result_dir, "normalized_float_result")
            os.makedirs(norm_res_dir, exist_ok=True)
            image.save(os.path.join(norm_res_dir, f"{i}.npz"))
            
    # generate the results folder contents with networks and with
    # an average background image visualization per network
    if results_folder:
        for network in range(N_FEATURES):
            network_folder_name = f"network_{network+1:03d}"
            image_sum = None
            for cohort, internal_result_dirs in cohort_internal_result_dirs.items():
                for i, internal_result_dir in enumerate(internal_result_dirs):
                    norm_res_dir = os.path.join(internal_result_dir, "normalized_float_result")
                    image_path = os.path.join(norm_res_dir, f"{network}.npz")
                    image = Image.load(image_path)
                    if image_sum is None:
                        image_sum = image.copy()
                    else:
                        image_sum += image
                    external_result_dir = os.path.join(results_folder, network_folder_name, cohort)
                    os.makedirs(external_result_dir, exist_ok=True)
                    io_utils.save_image(image, os.path.join(external_result_dir, f"result_for_dataset_{i+1:06d}.img.gz"))
            average_image = image_sum.normalized().scaled(4095).converted(np.int16)
            average_image_path = os.path.join(results_folder, network_folder_name, "average_visualization.img.gz")
            io_utils.save_image(average_image, average_image_path)
        if tift_compatibility_mode:
            # Write the TIFT compatibility script to the results folder
            origin = os.path.join(os.path.dirname(__file__), "tift_compatibility_script.py")
            destination = os.path.join(results_folder, "tift_compatibility_script.py")
            with open(origin, "r") as f:
                contents = f.read()
            with open(destination, "w") as f:
                f.write(contents)


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
