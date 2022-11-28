from setuptools import setup, Extension
from Cython.Build import cythonize
import numpy
setup(
    ext_modules=cythonize(["cython_packages/lidar2d_fast.pyx", "cython_packages/convex_hull.pyx"], annotate=True),
    include_dirs=[numpy.get_include()],
)
