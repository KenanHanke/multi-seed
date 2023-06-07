import glob
import os

# When changing variables and their defaults, only the default_config_str
# variable needs to be changed. The rest of the code will adapt automatically,
# as the Config class has no concept of specific variables and their defaults.

default_config_str = """
# This is a config file for fMRI data analysis. Characters following hashtags
# on the same line are not interpreted. Empty lines and whitespace that is not
# between interpreted characters are ignored. The word dataset refers to a set
# of images from one scan. The word cohort refers to a group of datasets.

# -----------------------------------------------------------------------------

# The following parameters can be set at the top of the file. Values listed
# here are the defaults that will be used when not specified explicitly.
# Underscores, order of variables and whitespace around equals signs are
# ignored. Previously set variables can be referenced in later variables,
# e.g. COMPARISON_FOLDER = path/to/folder/reduced_{REDUCTION_ALGORITHM}.


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

# The folder at the following path will be used for results of interindividual
# comparison. Setting this to NULL will forgo this calculation entirely.
COMPARISON_FOLDER = NULL

# -----------------------------------------------------------------------------

# The syntax for cohorts is as follows. Case and whitespace between characters
# are preserved, while empty lines and whitespace at the beginning and end of
# lines continue to be ignored. Paths should be to folders containing images
# of one scan in alphabetical order in their top level (one dataset). There
# must be at least one cohort and each must have at least one folder path.

[Cohort 1]
/path/to/one/folder/          # the path to a folder containing images
/path/to/multiple/*/fold?rs   # use glob patterns to match multiple folders

[Cohort 2]
/final/path/example
""".strip()


class Config:

    def __init__(self, params: dict, cohorts: dict):
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

    @classmethod
    def from_default(cls):
        return cls(*cls._interpret_config_str(default_config_str))

    @classmethod
    def from_str(cls, config_str):
        # Interpret default and custom config strings
        params, _ = cls._interpret_config_str(default_config_str)
        custom_params, cohorts = cls._interpret_config_str(config_str)

        # Override defaults with custom values
        params.update(custom_params)

        return cls(params, cohorts)

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
