# napari-cupy-image-processing

[![License](https://img.shields.io/pypi/l/napari-cupy-image-processing.svg?color=green)](https://github.com/haesleinhuepf/napari-cupy-image-processing/raw/master/LICENSE)
[![PyPI](https://img.shields.io/pypi/v/napari-cupy-image-processing.svg?color=green)](https://pypi.org/project/napari-cupy-image-processing)
[![Python Version](https://img.shields.io/pypi/pyversions/napari-cupy-image-processing.svg?color=green)](https://python.org)
[![tests](https://github.com/haesleinhuepf/napari-cupy-image-processing/workflows/tests/badge.svg)](https://github.com/haesleinhuepf/napari-cupy-image-processing/actions)
[![codecov](https://codecov.io/gh/haesleinhuepf/napari-cupy-image-processing/branch/master/graph/badge.svg)](https://codecov.io/gh/haesleinhuepf/napari-cupy-image-processing)
[![Development Status](https://img.shields.io/pypi/status/napari-cupy-image-processing.svg)](https://en.wikipedia.org/wiki/Software_release_life_cycle#Alpha)
[![napari hub](https://img.shields.io/endpoint?url=https://api.napari-hub.org/shields/napari-cupy-image-processing)](https://napari-hub.org/plugins/napari-cupy-image-processing)


GPU-accelerated image processing using [cupy](https://cupy.dev) and [CUDA](https://en.wikipedia.org/wiki/CUDA)

## Usage

This napari plugin adds some menu entries to the Tools menu. You can recognize them with their suffix `(n-cupy)` in brackets.
Furthermore, it can be used from the [napari-assistant](https://www.napari-hub.org/plugins/napari-assistant) graphical user interface. 
Therefore, just click the menu `Tools > Utilities > Assistant (na)` or run `naparia` from the command line.

![img.png](https://github.com/haesleinhuepf/napari-cupy-image-processing/raw/main/docs/screenshot-with-tools-menu.png)

You can also call operations from python, e.g. as shown in this [demo notebook](https://github.com/haesleinhuepf/napari-cupy-image-processing/raw/main/docs/demo.ipynb).

## Installation

Follow the [instructions for installing cupy](https://docs.cupy.dev/en/stable/install.html#installing-cupy-from-conda-forge) on your computer first.

    conda install -c conda-forge cupy

Afterwards, you can install `napari-cupy-image-processing` via [pip]:

    pip install napari-cupy-image-processing

A more detailed example for installation (change 11.2 to your desired CUDA version):
```
conda create --name cupy_p39 python=3.9
conda activate cupy_p39
conda install -c conda-forge cupy cudatoolkit=11.2 napari
pip install napari-cupy-image-processing
```

## Contributing

Contributions are very welcome. Adding [cupy ndimage](https://docs.cupy.dev/en/stable/reference/ndimage.html) functions is quite easy as you can see in the 
[implementation of the current operations](https://github.com/haesleinhuepf/napari-cupy-image-processing/blob/main/napari_cupy_image_processing/_cupy_image_processing.py#L48). 
If you need another function in napari, just send a PR. Please make sure the tests pass locally before submitting a PR.

```
pip install pytest-cov pytest-qt
pytest --cov=napari_cupy_image_processing
```

This [napari] plugin was generated with [Cookiecutter] using with [@napari]'s [cookiecutter-napari-plugin] template.

## License

Distributed under the terms of the [MIT] license,
"napari-cupy-image-processing" is free and open source software

## Issues

If you encounter any problems, please [file an issue] along with a detailed description.

[napari]: https://github.com/napari/napari
[Cookiecutter]: https://github.com/audreyr/cookiecutter
[@napari]: https://github.com/napari
[MIT]: http://opensource.org/licenses/MIT
[BSD-3]: http://opensource.org/licenses/BSD-3-Clause
[GNU GPL v3.0]: http://www.gnu.org/licenses/gpl-3.0.txt
[GNU LGPL v3.0]: http://www.gnu.org/licenses/lgpl-3.0.txt
[Apache Software License 2.0]: http://www.apache.org/licenses/LICENSE-2.0
[Mozilla Public License 2.0]: https://www.mozilla.org/media/MPL/2.0/index.txt
[cookiecutter-napari-plugin]: https://github.com/napari/cookiecutter-napari-plugin

[file an issue]: https://github.com/haesleinhuepf/napari-cupy-image-processing/issues

[napari]: https://github.com/napari/napari
[tox]: https://tox.readthedocs.io/en/latest/
[pip]: https://pypi.org/project/pip/
[PyPI]: https://pypi.org/
