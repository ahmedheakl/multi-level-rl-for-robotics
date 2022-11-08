from distutils.core import setup, Extension
from Cython.Build import cythonize

import numpy

package = Extension(
    "lidar2d_fast", ["lidar_setup/lidar2d_fast.pyx"], include_dirs=[numpy.get_include()]
)
setup(ext_modules=cythonize([package]))
