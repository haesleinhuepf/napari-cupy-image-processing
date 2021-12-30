
import numpy as np
from napari_plugin_engine import napari_hook_implementation


# This is the actual plugin function, where we export our function
# (The functions themselves are defined below)
@napari_hook_implementation
def napari_experimental_provide_function():
    from ._cupy_image_processing import gaussian_filter, \
        gaussian_laplace, \
        median_filter, \
        percentile_filter, \
        white_tophat, \
        morphological_gradient, \
        morphological_laplace, \
        wiener, \
        threshold_otsu,\
        binary_fill_holes, \
        label, \
        black_tophat, \
        minimum_filter, \
        maximum_filter, \
        binary_closing, \
        binary_erosion, \
        binary_opening, \
        binary_dilation, \
        measurements
    return [gaussian_filter,
            gaussian_laplace,
            median_filter,
            percentile_filter,
            white_tophat,
            morphological_gradient,
            morphological_laplace,
            wiener,
            threshold_otsu,
            binary_fill_holes,
            label,
            black_tophat,
            minimum_filter,
            maximum_filter,
            binary_closing,
            binary_erosion,
            binary_opening,
            binary_dilation,
            measurements]

