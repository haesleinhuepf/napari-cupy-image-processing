
import numpy as np
from napari_plugin_engine import napari_hook_implementation


# This is the actual plugin function, where we export our function
# (The functions themselves are defined below)
@napari_hook_implementation
def napari_experimental_provide_function():
    from ._cupy_image_processing import gaussian_filter
    return [gaussian_filter]

