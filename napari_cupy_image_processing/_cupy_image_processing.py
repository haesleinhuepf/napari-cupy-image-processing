import warnings
from typing import Callable
from functools import wraps
from toolz import curry
import inspect
import numpy as np
import napari
from napari_tools_menu import register_function
from napari_time_slicer import time_slicer

@curry
def plugin_function(
        function: Callable
) -> Callable:
    # copied from https://github.com/clEsperanto/pyclesperanto_prototype/blob/master/pyclesperanto_prototype/_tier0/_plugin_function.py
    @wraps(function)
    def worker_function(*args, **kwargs):
        import cupy

        sig = inspect.signature(function)
        # create mapping from position and keyword arguments to parameters
        # will raise a TypeError if the provided arguments do not match the signature
        # https://docs.python.org/3/library/inspect.html#inspect.Signature.bind
        bound = sig.bind(*args, **kwargs)
        # set default values for missing arguments
        # https://docs.python.org/3/library/inspect.html#inspect.BoundArguments.apply_defaults
        bound.apply_defaults()

        # copy images to GPU, and create output array if necessary
        for key, value in bound.arguments.items():
            if isinstance(value, np.ndarray):
                bound.arguments[key] = cupy.asarray(value)
            elif 'pyclesperanto_prototype._tier0._pycl.OCLArray' in str(type(value)) or \
                'dask.array.core.Array' in str(type(value)):
                # compatibility with pyclesperanto and dask
                bound.arguments[key] = cupy.asarray(np.asarray(value))

        # call the decorated function
        result = function(*bound.args, **bound.kwargs)

        if isinstance(result, cupy.ndarray):
            return np.asarray(result.get())
        else:
            return result
    worker_function.__module__ = "napari_cupy_image_processing"

    return worker_function


@register_function(menu="Filtering / noise removal > Gaussian (n-cupy)")
@time_slicer
@plugin_function
def gaussian_filter(image: napari.types.ImageData, sigma: float = 2, viewer: napari.Viewer = None) -> napari.types.ImageData:
    import cupyx
    from cupyx.scipy import ndimage
    return ndimage.gaussian_filter(image.astype(float), sigma)


@register_function(menu="Filtering / edge enhancement > Gaussian Laplace (n-cupy)")
@time_slicer
@plugin_function
def gaussian_laplace(image: napari.types.ImageData, sigma: float = 2, viewer: napari.Viewer = None) -> napari.types.ImageData:
    import cupyx
    from cupyx.scipy import ndimage
    return ndimage.gaussian_laplace(image.astype(float), sigma)


@register_function(menu="Filtering / noise removal > Median (n-cupy)")
@time_slicer
@plugin_function
def median_filter(image: napari.types.ImageData, radius: float = 2, viewer: napari.Viewer = None) -> napari.types.ImageData:
    import cupyx
    from cupyx.scipy import ndimage
    return ndimage.median_filter(image.astype(float), size=radius * 2 + 1)


@register_function(menu="Filtering / noise removal > Percentile (n-cupy)")
@time_slicer
@plugin_function
def percentile_filter(image: napari.types.ImageData, percentile : float = 50, radius: float = 2, viewer: napari.Viewer = None) -> napari.types.ImageData:
    import cupyx
    from cupyx.scipy import ndimage
    return ndimage.percentile_filter(image.astype(float), percentile=percentile, size=radius * 2 + 1)


@register_function(menu="Filtering / background removal > White Top-hat (n-cupy)")
@time_slicer
@plugin_function
def white_tophat(image: napari.types.ImageData, radius: float = 2, viewer: napari.Viewer = None) -> napari.types.ImageData:
    import cupyx
    from cupyx.scipy import ndimage
    return ndimage.white_tophat(image.astype(float), size=radius * 2 + 1)


@register_function(menu="Filtering / background removal > Black top-hat (n-cupy)")
@time_slicer
@plugin_function
def black_tophat(image: napari.types.ImageData, radius: float = 2, viewer: napari.Viewer = None) -> napari.types.ImageData:
    import cupyx
    from cupyx.scipy import ndimage
    return ndimage.black_tophat(image.astype(float), size=radius * 2 + 1)


@register_function(menu="Filtering / background removal > Minimum (n-cupy)")
@time_slicer
@plugin_function
def minimum_filter(image: napari.types.ImageData, radius: float = 2, viewer: napari.Viewer = None) -> napari.types.ImageData:
    import cupyx
    from cupyx.scipy import ndimage
    return ndimage.minimum_filter(image.astype(float), size=radius * 2 + 1)


@register_function(menu="Filtering / background removal > Maximum (n-cupy)")
@time_slicer
@plugin_function
def maximum_filter(image: napari.types.ImageData, radius: float = 2, viewer: napari.Viewer = None) -> napari.types.ImageData:
    import cupyx
    from cupyx.scipy import ndimage
    return ndimage.maximum_filter(image.astype(float), size=radius * 2 + 1)


@register_function(menu="Filtering / background removal > Morphological Gradient (n-cupy)")
@time_slicer
@plugin_function
def morphological_gradient(image: napari.types.ImageData, radius: float = 2, viewer: napari.Viewer = None) -> napari.types.ImageData:
    import cupyx
    from cupyx.scipy import ndimage
    return ndimage.morphological_gradient(image.astype(float), size=radius * 2 + 1)


@register_function(menu="Filtering > Morphological Laplace (n-cupy)")
@time_slicer
@plugin_function
def morphological_laplace(image: napari.types.ImageData, radius: float = 2, viewer: napari.Viewer = None) -> napari.types.ImageData:
    import cupyx
    from cupyx.scipy import ndimage
    return ndimage.morphological_laplace(image.astype(float), size=radius * 2 + 1)


@register_function(menu="Filtering / noise removal > Wiener (n-cupy)")
@time_slicer
@plugin_function
def wiener(image: napari.types.ImageData, radius: float = 2, viewer: napari.Viewer = None) -> napari.types.ImageData:
    from cupyx.scipy import signal
    return signal.wiener(image.astype(float), radius * 2 + 1)


@register_function(menu="Segmentation / binarization > Threshold (Otsu et al 1979, scikit-image, cupy)")
@time_slicer
@plugin_function
def threshold_otsu(image: napari.types.ImageData, viewer: napari.Viewer = None) -> napari.types.LabelsData:
    # adapted from https://github.com/clEsperanto/pyclesperanto_prototype/blob/master/pyclesperanto_prototype/_tier9/_threshold_otsu.py#L41
    import cupy

    minimum_intensity = image.min()
    maximum_intensity = image.max()

    range = maximum_intensity - minimum_intensity
    bin_centers = cupy.arange(256) * range / (255)

    histogram, _ = cupy.histogram(image, bins=256, range=(minimum_intensity, maximum_intensity))
    from skimage.filters import threshold_otsu

    threshold = threshold_otsu(hist=(histogram, bin_centers))

    return image > threshold


@register_function(menu="Segmentation post-processing > Binary fill holes (n-cupy)")
@time_slicer
@plugin_function
def binary_fill_holes(binary_image: napari.types.LabelsData, viewer: napari.Viewer = None) -> napari.types.LabelsData:
    import cupyx
    from cupyx.scipy import ndimage
    return ndimage.binary_fill_holes(binary_image)


@register_function(menu="Segmentation post-processing > Binary erosion (n-cupy)")
@time_slicer
@plugin_function
def binary_erosion(binary_image: napari.types.LabelsData, iterations: int = 1, viewer: napari.Viewer = None) -> napari.types.LabelsData:
    import cupyx
    from cupyx.scipy import ndimage
    return ndimage.binary_erosion(binary_image, iterations=iterations, brute_force=True)


@register_function(menu="Segmentation post-processing > Binary dilation (n-cupy)")
@time_slicer
@plugin_function
def binary_dilation(binary_image: napari.types.LabelsData, iterations: int = 1, viewer: napari.Viewer = None) -> napari.types.LabelsData:
    import cupyx
    from cupyx.scipy import ndimage
    return ndimage.binary_dilation(binary_image, iterations=iterations, brute_force=True)


@register_function(menu="Segmentation post-processing > Binary closing (n-cupy)")
@time_slicer
@plugin_function
def binary_closing(binary_image: napari.types.LabelsData, iterations: int = 1, viewer: napari.Viewer = None) -> napari.types.LabelsData:
    import cupyx
    from cupyx.scipy import ndimage
    return ndimage.binary_closing(binary_image, iterations=iterations, brute_force=True)


@register_function(menu="Segmentation post-processing > Binary opening (n-cupy)")
@time_slicer
@plugin_function
def binary_opening(binary_image: napari.types.LabelsData, iterations: int = 1, viewer: napari.Viewer = None) -> napari.types.LabelsData:
    import cupyx
    from cupyx.scipy import ndimage
    return ndimage.binary_opening(binary_image, iterations=iterations, brute_force=True)


@register_function(menu="Segmentation / labeling > Connected component labeling (n-cupy)")
@time_slicer
@plugin_function
def label(binary_image: napari.types.LabelsData, viewer: napari.Viewer = None) -> napari.types.LabelsData:
    import cupyx
    from cupyx.scipy import ndimage
    result, _ = ndimage.label(binary_image)
    return result


@register_function(menu="Measurement > Measurements (n-cupy)")
def measurements(intensity_image: napari.types.ImageData,
                 label_image: napari.types.LabelsData,
                 napari_viewer : napari.Viewer = None,
                 size: bool = True,
                 intensity: bool = True,
                 position: bool = False):
    import cupy
    from cupyx.scipy import ndimage

    from warnings import warn
    warn(
        'napari-cupy-image-processing measurements() is deprecated. Consider switching to napari-simpleitk-image-processing label_statistics()',
        DeprecationWarning)
    print(
        'napari-cupy-image-processing measurements() is deprecated. Consider switching to napari-simpleitk-image-processing label_statistics()')

    if intensity_image is not None and label_image is not None:

        labels = cupy.asarray(label_image)
        image = cupy.asarray(intensity_image).astype(np.float32)

        df = {}

        for l in range(1, labels.max().get() + 1):
            _append_to_column(df, "label", l)
            if position:
                for i, x in enumerate(ndimage.center_of_mass(image, labels, l)):
                    _append_to_column(df, "center_of_mass_" + str(i), x.get())
                for i, x in enumerate(ndimage.minimum_position(image, labels, l)):
                    _append_to_column(df, "minimum_position_" + str(i), x)
                for i, x in enumerate(ndimage.maximum_position(image, labels, l)):
                    _append_to_column(df, "maximum_position_" + str(i), x)

            mean = None
            if intensity:
                x = ndimage.mean(image, labels, l)
                mean = x.get()
                _append_to_column(df, "mean", x.get())
                x = ndimage.minimum(image, labels, l)
                _append_to_column(df, "minimum", x.get())
                x = ndimage.maximum(image, labels, l)
                _append_to_column(df, "maximum", x.get())
                x = ndimage.median(image, labels, l)
                _append_to_column(df, "median", x.get())
                x = ndimage.standard_deviation(image, labels, l)
                _append_to_column(df, "standard_deviation", x.get())

            if size:
                sum_labels = ndimage.sum_labels(image, labels, l).get()
                if mean is None:
                    mean = ndimage.mean(image, labels, l).get()
                pixel_count = sum_labels / mean
                _append_to_column(df, "pixel_count", pixel_count)

        result = {}
        for k, v in df.items():
            result[k] = np.asarray(v).tolist()

        if napari_viewer is not None:
            from napari_workflows._workflow import _get_layer_from_data
            labels_layer = _get_layer_from_data(napari_viewer, label_image)
            # Store results in the properties dictionary:
            labels_layer.properties = result

            # turn table into a widget
            from napari_skimage_regionprops import add_table
            add_table(labels_layer, napari_viewer)
        else:
            return result
    else:
        warnings.warn("Image and labels must be set.")

def _append_to_column(dictionary, column_name, value):
    if column_name not in dictionary.keys():
        dictionary[column_name] = []
    dictionary[column_name].append(value)
