from setuptools import setup
from Cython.Build import cythonize
from setuptools.extension import Extension

import numpy

extensions = [
    Extension("parser", sources = ["parser.pyx"]),
    Extension("utils", sources = ["utils.pyx"]),
    Extension("modeling", sources = ["modeling.pyx"]),
]


setup(
    ext_modules = cythonize(extensions, compiler_directives={"language_level": "3"}),
    include_dirs=[numpy.get_include()]
)

