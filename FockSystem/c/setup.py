from setuptools import setup, Extension
from Cython.Build import cythonize
import numpy
numpy_include = numpy.get_include()  # Get NumPy include path


setup(
    ext_modules=cythonize(
        Extension(
            "fermion_operations",
            ["fermion_operations.pyx"],
            include_dirs=[numpy_include],  # Explicitly add NumPy headers
        )
    ),
)
