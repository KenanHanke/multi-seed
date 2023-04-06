class Projection:
    """
    Function that fulfills feature projection; not idempotent in the
    formal mathematical sense of a projection.
    """

    # I should perform PCA on the dataset without normalizing it
    # at all, because correlation coefficients are already inside
    # the range [-1, 1] and it makes sense that a correlation
    # coefficient of 0 should stay 0 after PCA (linear mappings
    # preserve 0).

    # After PCA, I should perform absolute value on the result, because
    # a strong negative correlation is just as informative for connection
    # detection as a strong positive correlation (inhibitory vs. excitatory
    # connections). A positive side effect of this is that cerebrospinal
    # fluid and similarly disconnected regions will be projected to the
    # origin and be black in the result.

    # I believe that performing absolute value after PCA is permissible
    # because PCA is a linear transformation and no transformation
    # took place beforehand through normalization. I NEED TO CHECK THIS!
    # Making each individual value of a vector its absolute value is
    # equivalent to several reflections which are different for each
    # vector, or whatever this means in high-dimensional space.

    # Variable importance of correlations with specific reference
    # points might also be interesting to track. Maybe I could create
    # a heat map of the brain that shows what reference points are
    # most important for separating the high-dimensional correlation
    # data.
