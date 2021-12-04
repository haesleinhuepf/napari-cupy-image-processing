from typing import Callable
from functools import wraps
from toolz import curry
import inspect
import numpy as np
import cupy
import cupyx
from cupyx.scipy import ndimage
from cupyx.scipy import signal
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

    return worker_function


@register_function(menu="Filtering / noise removal > Gaussian (n-cupy)")
@time_slicer
@plugin_function
def gaussian_filter(image: napari.types.ImageData, sigma: float = 2, viewer: napari.Viewer = None) -> napari.types.ImageData:
    return cupyx.scipy.ndimage.gaussian_filter(image.astype(float), sigma)


@register_function(menu="Filtering / edge enhancement > Gaussian Laplace (n-cupy)")
@time_slicer
@plugin_function
def gaussian_laplace(image: napari.types.ImageData, sigma: float = 2, viewer: napari.Viewer = None) -> napari.types.ImageData:
    return cupyx.scipy.ndimage.gaussian_laplace(image.astype(float), sigma)


@register_function(menu="Filtering / noise removal > Median (n-cupy)")
@time_slicer
@plugin_function
def median_filter(image: napari.types.ImageData, radius: float = 2, viewer: napari.Viewer = None) -> napari.types.ImageData:
    return cupyx.scipy.ndimage.median_filter(image.astype(float), size=radius * 2 + 1)


@register_function(menu="Filtering / noise removal > Percentile (n-cupy)")
@time_slicer
@plugin_function
def percentile_filter(image: napari.types.ImageData, percentile : float = 50, radius: float = 2, viewer: napari.Viewer = None) -> napari.types.ImageData:
    return cupyx.scipy.ndimage.percentile_filter(image.astype(float), percentile=percentile, size=radius * 2 + 1)


@register_function(menu="Filtering / background removal > White Top-hat (n-cupy)")
@time_slicer
@plugin_function
def white_tophat(image: napari.types.ImageData, radius: float = 2, viewer: napari.Viewer = None) -> napari.types.ImageData:
    return cupyx.scipy.ndimage.white_tophat(image.astype(float), size=radius * 2 + 1)


@register_function(menu="Filtering / background removal > Black top-hat (n-cupy)")
@time_slicer
@plugin_function
def black_tophat(image: napari.types.ImageData, radius: float = 2, viewer: napari.Viewer = None) -> napari.types.ImageData:
    return cupyx.scipy.ndimage.black_tophat(image.astype(float), size=radius * 2 + 1)


@register_function(menu="Filtering / background removal > Minimum (n-cupy)")
@time_slicer
@plugin_function
def minimum_filter(image: napari.types.ImageData, radius: float = 2, viewer: napari.Viewer = None) -> napari.types.ImageData:
    return cupyx.scipy.ndimage.minimum_filter(image.astype(float), size=radius * 2 + 1)


@register_function(menu="Filtering / background removal > Maximum (n-cupy)")
@time_slicer
@plugin_function
def maximum_filter(image: napari.types.ImageData, radius: float = 2, viewer: napari.Viewer = None) -> napari.types.ImageData:
    return cupyx.scipy.ndimage.maximum_filter(image.astype(float), size=radius * 2 + 1)


@register_function(menu="Filtering / background removal > Morphological Gradient (n-cupy)")
@time_slicer
@plugin_function
def morphological_gradient(image: napari.types.ImageData, radius: float = 2, viewer: napari.Viewer = None) -> napari.types.ImageData:
    return cupyx.scipy.ndimage.morphological_gradient(image.astype(float), size=radius * 2 + 1)


@register_function(menu="Filtering > Morphological Laplace (n-cupy)")
@time_slicer
@plugin_function
def morphological_laplace(image: napari.types.ImageData, radius: float = 2, viewer: napari.Viewer = None) -> napari.types.ImageData:
    return cupyx.scipy.ndimage.morphological_laplace(image.astype(float), size=radius * 2 + 1)


@register_function(menu="Filtering / noise removal > Wiener (n-cupy)")
@time_slicer
@plugin_function
def wiener(image: napari.types.ImageData, radius: float = 2, viewer: napari.Viewer = None) -> napari.types.ImageData:
    return signal.wiener(image.astype(float), radius * 2 + 1)


@register_function(menu="Segmentation / binarization > Threshold (Otsu et al 1979, scikit-image, cupy)")
@time_slicer
@plugin_function
def threshold_otsu(image: napari.types.ImageData, viewer: napari.Viewer = None) -> napari.types.LabelsData:
    # adapted from https://github.com/clEsperanto/pyclesperanto_prototype/blob/master/pyclesperanto_prototype/_tier9/_threshold_otsu.py#L41

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
    return cupyx.scipy.ndimage.binary_fill_holes(binary_image)


@register_function(menu="Segmentation post-processing > Binary erosion (n-cupy)")
@time_slicer
@plugin_function
def binary_erosion(binary_image: napari.types.LabelsData, iterations: int = 1, viewer: napari.Viewer = None) -> napari.types.LabelsData:
    return cupyx.scipy.ndimage.binary_erosion(binary_image, iterations=iterations, brute_force=True)


@register_function(menu="Segmentation post-processing > Binary dilation (n-cupy)")
@time_slicer
@plugin_function
def binary_dilation(binary_image: napari.types.LabelsData, iterations: int = 1, viewer: napari.Viewer = None) -> napari.types.LabelsData:
    return cupyx.scipy.ndimage.binary_dilation(binary_image, iterations=iterations, brute_force=True)


@register_function(menu="Segmentation post-processing > Binary closing (n-cupy)")
@time_slicer
@plugin_function
def binary_closing(binary_image: napari.types.LabelsData, iterations: int = 1, viewer: napari.Viewer = None) -> napari.types.LabelsData:
    return cupyx.scipy.ndimage.binary_closing(binary_image, iterations=iterations, brute_force=True)


@register_function(menu="Segmentation post-processing > Binary opening (n-cupy)")
@time_slicer
@plugin_function
def binary_opening(binary_image: napari.types.LabelsData, iterations: int = 1, viewer: napari.Viewer = None) -> napari.types.LabelsData:
    return cupyx.scipy.ndimage.binary_opening(binary_image, iterations=iterations, brute_force=True)


@register_function(menu="Segmentation / labeling > Connected component labeling (n-cupy)")
@time_slicer
@plugin_function
def label(binary_image: napari.types.LabelsData, viewer: napari.Viewer = None) -> napari.types.LabelsData:
    result, _ = cupyx.scipy.ndimage.label(binary_image)
    return result

