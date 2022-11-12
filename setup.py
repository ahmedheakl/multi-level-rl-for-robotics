from setuptools import setup, Extension
from Cython.Build import cythonize
import numpy
import os

setup(
    ext_modules=cythonize("lidar2d_fast.pyx", annotate=True),
    include_dirs=[numpy.get_include()],
)
