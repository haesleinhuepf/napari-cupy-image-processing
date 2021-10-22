from typing import Callable
from functools import wraps
from toolz import curry
import inspect
import numpy as np
import cupy
import cupyx
from cupyx.scipy import ndimage
import napari
from napari_tools_menu import register_function

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
            elif 'pyclesperanto_prototype._tier0._pycl.OCLArray' in str(type(value)):
                # compatibility with pyclesperanto
                bound.arguments[key] = cupy.asarray(np.asarray(value))

        # call the decorated function
        result = function(*bound.args, **bound.kwargs)
        print("Result is ", str(type(result)))
        if isinstance(result, cupy.ndarray):
            return np.asarray(result.get())
        else:
            return result

    return worker_function

@register_function(menu="Filtering > Gaussian filter (cupy)")
@plugin_function
def gaussian_filter(image: napari.types.ImageData, sigma: float = 2) -> napari.types.ImageData:
    return cupyx.scipy.ndimage.gaussian_filter(image, sigma)

@register_function(menu="Segmentation > Connected component labeling (cupy)")
@plugin_function
def label(binary_image: napari.types.LabelsData) -> napari.types.LabelsData:
    result, _ = cupyx.scipy.ndimage.label(binary_image)
    return result

