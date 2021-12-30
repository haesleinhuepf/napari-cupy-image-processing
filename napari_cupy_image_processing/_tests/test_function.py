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


def test_measurements(make_napari_viewer):
    from napari_cupy_image_processing import measurements

    viewer = make_napari_viewer()

    image = np.asarray([
        [0, 1, 2, 3],
        [0, 1, 2, 3],
        [4, 4, 4, 4],
        [5, 5, 0, 0]
    ])
    labels = image

    image_layer = viewer.add_image(image)
    labels_layer = viewer.add_labels(labels)

    measurements(image_layer, labels_layer, viewer, size=True, intensity=True, position=True)
    result = labels_layer.properties


    reference = {'center_of_mass_0': [0.5, 0.5, 0.5, 2.0, 3.0],
                 'center_of_mass_1': [1.0, 2.0, 3.0, 1.5, 0.5],
                 'minimum_position_0': [0, 0, 0, 2, 3],
                 'minimum_position_1': [1, 2, 3, 0, 0],
                 'maximum_position_0': [0, 0, 0, 2, 3],
                 'maximum_position_1': [1, 2, 3, 0, 0],
                 'mean': [1.0, 2.0, 3.0, 4.0, 5.0],
                 'minimum': [1, 2, 3, 4, 5],
                 'maximum': [1, 2, 3, 4, 5],
                 'median': [1.0, 2.0, 3.0, 4.0, 5.0],
                 'standard_deviation': [0.0, 0.0, 0.0, 0.0, 0.0],
                 'pixel_count': [2.0, 2.0, 2.0, 4.0, 2.0]}

    for k, v in result.items():
        assert np.allclose(result[k], reference[k], 0.001)

    for k, v in reference.items():
        assert np.allclose(result[k], reference[k], 0.001)