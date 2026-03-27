# multiseed/config.py

import glob
import os
from typing import Any
import functools
import operator
import numpy as np
from .image import Image
from .reference import ReferenceBuilder
from .ioutils import save_image, load_datasets_lazy
from .mapper import name_to_mapper_class
from .tift import TIFT_COMPATIBILITY_SCRIPT_CONTENTS


# When changing variables and their defaults, only the DEFAULT_CONFIG_STR
# variable needs to be changed. The rest of the code will adapt automatically,
# as the Config class has no concept of specific variables and their defaults.

DEFAULT_CONFIG_STR = r"""
# This is a config file for fMRI data analysis. Characters following hashtags
# on the same line are not interpreted. Empty lines and whitespace that is not
# between interpreted characters are ignored. The word dataset refers to a set
# of images from one scan. The word cohort refers to a group of datasets.

# -----------------------------------------------------------------------------

# The following parameters can be set at the top of the file. Values listed
# here are the defaults that will be used when not specified explicitly.
# Underscores, order of variables and whitespace around equals signs are
# ignored. Previously set variables can be referenced in later variables,
# e.g. RESULTS_FOLDER = path/to/folder/reduced_{REDUCTION_ALGORITHM}.


# Enqueueing subsequent datasets asynchronously uses 2x the RAM but saves time
PARALLEL_IO = FALSE

# Parameters for sampling of seeds
N_SEEDS = 1000                 # Number of seed regions to use for analysis
SEED_RADIUS = 10               # Seed radius in voxels

# Parameters for the reduction algorithm
REDUCTION_ALGORITHM = PCA      # See documentation for available algorithms
N_FEATURES = 20                # Number of dimensions kept during reduction
N_SAMPLES_PER_DATASET = 5000   # Number of sample voxels per dataset to use
                               #    when calibrating reduction algorithm

# The folder at the following path will be used to store the resulting
# networks for each dataset. It will be created if it does not exist.
# It will also additionally contain the seed visualization, a mask image
# and one average image per network. Setting this to NULL will mean that
# results are only stored in the dataset folders given in the cohorts
# using the internal file format.
RESULTS_FOLDER = NULL

# Setting this to TRUE will cause a Python script to be written to the results
# folder which can modify the file format to be compatible with the TIFT
# software package and generate TIFT project files
TIFT_COMPATIBILITY_MODE = TRUE

# -----------------------------------------------------------------------------

# The syntax for cohorts is as follows: Case and whitespace between characters
# are preserved, while empty lines and whitespace at the beginning and end of
# lines continue to be ignored. Paths should be to folders containing images
# of one scan in alphabetical order in their top level (one dataset). There
# must be at least one cohort and each must have at least one folder path.

[Cohort 1]
/path/to/one/folder/          # the path to a folder containing images
/path/to/multiple/*/fold?rs   # use glob patterns to match multiple folders
                              #   (glob patterns are expanded alphabetically)

[Cohort 2]
/final/path/example
""".strip()


class Config:

    def __init__(self, params: dict[str, Any], cohorts: dict[str, list[str]]):
        self.params = params
        self.cohorts = cohorts

    @staticmethod
    def _interpret_config_str(config_str: str):
        params = {}
        cohorts = {}

        current_cohort = None
        for line in config_str.split('\n'):
            line = line.partition('#')[0]  # remove comments
            line = line.strip()
            if not line: continue  # ignore empty lines

            # detect and handle cohort names
            if line.startswith('[') and line.endswith(']'):
                current_cohort = line[1:-1].strip()
                cohorts[current_cohort] = []

            # handle parameters
            elif current_cohort is None:
                key, _, value = line.partition('=')
                key, value = key.strip(), value.strip()

                # reinterpret value as if possible
                if value.isdigit():
                    value = int(value)
                elif value.lower() in ('true', 'false'):
                    value = value.lower() == 'true'
                elif value.lower() == 'null':
                    value = None
                else:
                    # format string values to allow referencing previously set
                    # variables; replace True/False/None with TRUE/FALSE/NULL
                    value = value.format_map({
                        k: "TRUE" if v == True else
                        "FALSE" if v == False else "NULL" if v is None else v
                        for k, v in params.items()
                    })

                # store interpreted value
                params[key] = value

            # handle cohort paths
            else:
                paths = [os.path.abspath(path) for path in glob.glob(line)]
                paths.sort()
                cohorts[current_cohort].extend(paths)

        return params, cohorts

    @staticmethod
    def write_default_config(path):
        with open(path, 'w') as f:
            f.write(DEFAULT_CONFIG_STR)

    @classmethod
    def from_str(cls, config_str):
        # Interpret default and custom config strings
        params, _ = cls._interpret_config_str(DEFAULT_CONFIG_STR)
        custom_params, cohorts = cls._interpret_config_str(config_str)

        # Override defaults with custom values
        params.update(custom_params)

        return cls(params, cohorts)

    @classmethod
    def from_file(cls, path):
        with open(path, 'r') as f:
            config_str = f.read()
        return cls.from_str(config_str)

    def __str__(self):
        a = []
        for k, v in self.params.items():
            if v is None:
                v = 'NULL'
            if not isinstance(v, str):
                v = str(v).upper()
            a.append(f"{k} = {v}")
        for k, v in self.cohorts.items():
            a.append(f"\n[{k}]")
            for path in v:
                a.append(path)
        return "\n".join(a)

    def run(self, rng=None):
        parallel_io: bool = self.params["PARALLEL_IO"]
        results_folder = self.params["RESULTS_FOLDER"]
        tift_compatibility_mode = self.params["TIFT_COMPATIBILITY_MODE"]
        if results_folder:
            os.makedirs(results_folder, exist_ok=True)

        # prepare the dataset loaders for each cohort
        cohort_loaders = {
            cohort: load_datasets_lazy(paths, asynchronous=parallel_io)
            for cohort, paths in self.cohorts.items()
        }

        # get mask
        all_dataset_loader = functools.reduce(operator.add,
                                            cohort_loaders.values())
        mask = all_dataset_loader.extract_mask()
        if results_folder:
            # save the mask to the results folder
            mask_path = os.path.join(results_folder, "mask.img.gz")
            reformatted_mask = mask.converted(np.float32).scaled(255).converted(np.int16)
            save_image(reformatted_mask, mask_path)

        # prepare reference builder
        reference_builder = ReferenceBuilder(radius=self.params["SEED_RADIUS"],
                                            mask=mask)
        reference_builder.sample(self.params["N_SEEDS"], rng)

        if results_folder:
            # create visualization of reference builder
            visualization = reference_builder.visualized().normalized().scaled(4095).converted(np.int16)
            visualization_path = os.path.join(results_folder,
                                            "seed_visualization.img.gz")
            save_image(visualization, visualization_path)

        # initialize and fit mapper
        N_FEATURES = self.params["N_FEATURES"]
        REDUCTION_ALGORITHM = self.params["REDUCTION_ALGORITHM"]
        Mapper = name_to_mapper_class(REDUCTION_ALGORITHM)
        mapper = Mapper(N_FEATURES, reference_builder)
        mapper.fit(all_dataset_loader,
                samples_per_dataset=self.params["N_SAMPLES_PER_DATASET"],
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
                        save_image(image, os.path.join(external_result_dir, f"result_for_dataset_{i+1:06d}.img.gz"))
                average_image = image_sum.normalized().scaled(4095).converted(np.int16)
                average_image_path = os.path.join(results_folder, network_folder_name, "average_visualization.img.gz")
                save_image(average_image, average_image_path)
            if tift_compatibility_mode:
                # Write the TIFT compatibility script to the results folder
                destination = os.path.join(results_folder, "tift_compatibility_script.py")
                with open(destination, "w") as f:
                    f.write(TIFT_COMPATIBILITY_SCRIPT_CONTENTS)