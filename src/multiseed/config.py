# src/multiseed/config.py

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
    """Parsed analysis configuration.

    Instances store two pieces of state:

    - `params`: scalar configuration values such as seed count, mapper choice,
      and output settings.
    - `cohorts`: mapping from cohort name to the dataset folder paths that
      belong to that cohort.

    A config string consists of a parameter section followed by one or more
    cohort sections. Parameter values support integers, booleans (`TRUE` /
    `FALSE`), `NULL`, and string interpolation against previously defined
    parameters.
    """

    def __init__(self, params: dict[str, Any], cohorts: dict[str, list[str]]):
        """Initialize a parsed configuration object.

        Args:
            params: Parsed top-level parameter values.
            cohorts: Mapping from cohort name to dataset directory paths.
        """
        self.params = params
        self.cohorts = cohorts

    @staticmethod
    def _interpret_config_str(
        config_str: str,
    ) -> tuple[dict[str, Any], dict[str, list[str]]]:
        """Parse a config string into parameter values and cohort path lists.

        Parsing rules:
        - Inline comments beginning with `#` are removed.
        - Empty lines are ignored.
        - Lines before the first `[Cohort Name]` header are treated as
          `KEY = VALUE` parameter assignments.
        - Integer, boolean, and `NULL` literals are converted to Python types.
        - Other parameter values are treated as strings and formatted with
          previously parsed parameters, with `True`/`False`/`None` exposed as
          `TRUE`/`FALSE`/`NULL`.
        - Lines inside a cohort section are treated as glob patterns; matches
          are expanded to absolute paths in sorted order.

        Args:
            config_str: Raw configuration text.

        Returns:
            A `(params, cohorts)` tuple where `params` contains parsed scalar
            values and `cohorts` maps cohort names to dataset folder paths.
        """
        params: dict[str, Any] = {}
        cohorts: dict[str, list[str]] = {}

        current_cohort = None

        for line in config_str.split("\n"):
            line = line.partition("#")[0].strip()
            if not line:
                continue

            if line.startswith("[") and line.endswith("]"):
                current_cohort = line[1:-1].strip()
                cohorts[current_cohort] = []
            elif current_cohort is None:
                key, _, value = line.partition("=")
                key, value = key.strip(), value.strip()

                if value.isdigit():
                    value = int(value)
                elif value.lower() in ("true", "false"):
                    value = value.lower() == "true"
                elif value.lower() == "null":
                    value = None
                else:
                    value = value.format_map(
                        {
                            k: (
                                "TRUE"
                                if v is True
                                else "FALSE"
                                if v is False
                                else "NULL"
                                if v is None
                                else v
                            )
                            for k, v in params.items()
                        }
                    )

                params[key] = value
            else:
                paths = [os.path.abspath(path) for path in glob.glob(line)]
                paths.sort()
                cohorts[current_cohort].extend(paths)

        return params, cohorts

    @staticmethod
    def write_default_config(path: str) -> None:
        """Write the built-in default configuration template to disk.

        Args:
            path: Destination file path.
        """
        with open(path, "w") as f:
            f.write(DEFAULT_CONFIG_STR)

    @classmethod
    def from_str(cls, config_str: str) -> "Config":
        """Create a config from raw text, overlaying it on the defaults.

        The default config is parsed first so omitted parameters inherit their
        default values. Cohorts come only from `config_str`.

        Args:
            config_str: User-provided configuration text.

        Returns:
            A fully populated `Config` instance.
        """
        params, _ = cls._interpret_config_str(DEFAULT_CONFIG_STR)
        custom_params, cohorts = cls._interpret_config_str(config_str)

        params.update(custom_params)
        return cls(params, cohorts)

    @classmethod
    def from_file(cls, path: str) -> "Config":
        """Load and parse a configuration file.

        Args:
            path: Path to the configuration file.

        Returns:
            A fully populated `Config` instance.
        """
        with open(path, "r") as f:
            config_str = f.read()
        return cls.from_str(config_str)

    def __str__(self) -> str:
        """Serialize the configuration back to its text representation.

        Returns:
            A config-like string containing all current parameters and cohort
            paths. `None`, booleans, and non-string values are rendered in the
            config file style (`NULL`, `TRUE`, `FALSE`, etc.).
        """
        lines = []

        for key, value in self.params.items():
            if value is None:
                value = "NULL"
            if not isinstance(value, str):
                value = str(value).upper()
            lines.append(f"{key} = {value}")

        for cohort_name, paths in self.cohorts.items():
            lines.append(f"\n[{cohort_name}]")
            for path in paths:
                lines.append(path)

        return "\n".join(lines)

    def run(self, rng=None) -> None:
        """Execute the full multi-seed group analysis described by this config.

        The pipeline performs the following steps:

        1. Build lazy dataset loaders for each cohort.
        2. Compute a combined mask across all datasets.
        3. Sample seed locations from that mask and optionally write mask/seed
           visualizations to the external results folder.
        4. Fit the configured dimensionality-reduction mapper on sampled voxel
           correlations from all datasets.
        5. Transform every dataset into feature images and save raw float
           outputs inside each dataset directory.
        6. Normalize each feature jointly across all datasets and save the
           normalized float outputs alongside the raw ones.
        7. Optionally export per-network Analyze images and average network
           visualizations to the external results folder, plus a helper script
           for TIFT compatibility.

        Args:
            rng: Optional NumPy random generator used for reproducible seed and
                voxel sampling. If `None`, downstream code uses its own default
                generator behavior.
        """
        parallel_io: bool = self.params["PARALLEL_IO"]
        results_folder = self.params["RESULTS_FOLDER"]
        tift_compatibility_mode = self.params["TIFT_COMPATIBILITY_MODE"]

        if results_folder:
            os.makedirs(results_folder, exist_ok=True)

        cohort_loaders = {
            cohort: load_datasets_lazy(paths, asynchronous=parallel_io)
            for cohort, paths in self.cohorts.items()
        }

        all_dataset_loader = functools.reduce(operator.add, cohort_loaders.values())
        mask = all_dataset_loader.extract_mask()

        if results_folder:
            mask_path = os.path.join(results_folder, "mask.img.gz")
            reformatted_mask = (
                mask.converted(np.float32).scaled(255).converted(np.int16)
            )
            save_image(reformatted_mask, mask_path)

        reference_builder = ReferenceBuilder(
            radius=self.params["SEED_RADIUS"],
            mask=mask,
        )
        reference_builder.sample(self.params["N_SEEDS"], rng)

        if results_folder:
            visualization = (
                reference_builder.visualized()
                .normalized()
                .scaled(4095)
                .converted(np.int16)
            )
            visualization_path = os.path.join(
                results_folder,
                "seed_visualization.img.gz",
            )
            save_image(visualization, visualization_path)

        n_features = self.params["N_FEATURES"]
        reduction_algorithm = self.params["REDUCTION_ALGORITHM"]
        mapper_cls = name_to_mapper_class(reduction_algorithm)
        mapper = mapper_cls(n_features, reference_builder)
        mapper.fit(
            all_dataset_loader,
            samples_per_dataset=self.params["N_SAMPLES_PER_DATASET"],
            rng=rng,
        )

        cohort_internal_result_dirs: dict[str, list[str]] = {}

        for cohort, loader in cohort_loaders.items():
            cohort_internal_result_dirs[cohort] = []

            for dataset, path in zip(loader, loader.paths):
                result = mapper.transform(dataset)

                internal_result_dir = os.path.join(path, reduction_algorithm)
                raw_result_dir = os.path.join(internal_result_dir, "raw_float_result")
                os.makedirs(raw_result_dir, exist_ok=True)

                for i, image in enumerate(result):
                    image.save(os.path.join(raw_result_dir, f"{i}.npz"))

                cohort_internal_result_dirs[cohort].append(internal_result_dir)

        internal_result_dirs = sum(cohort_internal_result_dirs.values(), [])

        for i in range(n_features):

            def image_iter_func():
                for result_dir in internal_result_dirs:
                    yield Image.load(
                        os.path.join(
                            result_dir,
                            "raw_float_result",
                            f"{i}.npz",
                        )
                    )

            normalized_images = Image.normalize_all_lazy(image_iter_func)

            for internal_result_dir, image in zip(
                internal_result_dirs, normalized_images
            ):
                normalized_result_dir = os.path.join(
                    internal_result_dir,
                    "normalized_float_result",
                )
                os.makedirs(normalized_result_dir, exist_ok=True)
                image.save(os.path.join(normalized_result_dir, f"{i}.npz"))

        if results_folder:
            for network in range(n_features):
                network_folder_name = f"network_{network + 1:03d}"
                image_sum = None

                for cohort, internal_result_dirs in cohort_internal_result_dirs.items():
                    for i, internal_result_dir in enumerate(internal_result_dirs):
                        normalized_result_dir = os.path.join(
                            internal_result_dir,
                            "normalized_float_result",
                        )
                        image_path = os.path.join(
                            normalized_result_dir,
                            f"{network}.npz",
                        )
                        image = Image.load(image_path)

                        if image_sum is None:
                            image_sum = image.copy()
                        else:
                            image_sum += image

                        external_result_dir = os.path.join(
                            results_folder,
                            network_folder_name,
                            cohort,
                        )
                        os.makedirs(external_result_dir, exist_ok=True)
                        save_image(
                            image,
                            os.path.join(
                                external_result_dir,
                                f"result_for_dataset_{i + 1:06d}.img.gz",
                            ),
                        )

                average_image = (
                    image_sum.normalized().scaled(4095).converted(np.int16)
                )
                average_image_path = os.path.join(
                    results_folder,
                    network_folder_name,
                    "average_visualization.img.gz",
                )
                save_image(average_image, average_image_path)

            if tift_compatibility_mode:
                destination = os.path.join(
                    results_folder,
                    "tift_compatibility_script.py",
                )
                with open(destination, "w") as f:
                    f.write(TIFT_COMPATIBILITY_SCRIPT_CONTENTS)