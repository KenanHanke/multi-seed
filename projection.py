class Projection:
    """
    Function that fulfills feature projection; not idempotent in the
    formal mathematical sense of a projection.
    """

    # I should perform PCA on the dataset without normalizing it
    # at all, because correlation coefficients are already normalized
    # to the range [-1, 1] and it makes sense that a correlation
    # coefficient of 0 should stay 0 after PCA.

    # After PCA, I should perform absolute value on the result, because
    # a strong negative correlation is just as informative for connection
    # detection as a strong positive correlation (inhibitory vs. excitatory
    # connections). A positive side effect of this is that cerebrospinal
    # fluid and similarly disconnected regions will be projected to the
    # origin and be black in the result.

    # I believe that performing absolute value after PCA is permissible
    # because PCA is a linear transformation and no transformation
    # took place beforehand. I NEED TO CHECK THIS!
