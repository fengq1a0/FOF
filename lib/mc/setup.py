from setuptools import setup, Extension
from Cython.Build import cythonize
import numpy as np

setup(
    ext_modules=cythonize(Extension(
        name="_marching_cubes_lewiner_cy",
        sources=["_marching_cubes_lewiner_cy.pyx"],
        include_dirs=[np.get_include()],
        extra_compile_args=["-O3"]
    )))