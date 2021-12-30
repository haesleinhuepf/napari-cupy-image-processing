import numpy as np


def test_with_numpy_2d():
    import numpy
    image = (np.random.random((100, 100)) > 0.5) * 1

    dummy_workflow(image)

def dummy_workflow(image):
    from napari_cupy_image_processing import \
        gaussian_filter, \
        gaussian_laplace, \
        median_filter, \
        percentile_filter, \
        white_tophat, \
        morphological_gradient, \
        morphological_laplace, \
        wiener, \
        threshold_otsu, \
        binary_fill_holes, \
        label, \
        black_tophat, \
        minimum_filter, \
        maximum_filter, \
        binary_closing, \
        binary_erosion, \
        binary_opening, \
        binary_dilation

    operations = [
        gaussian_filter,
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
        binary_dilation]

    for operation in operations:
        operation(image)
