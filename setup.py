from setuptools import setup, find_packages
from setuptools import Extension
from Cython.Build import cythonize
from Cython.Distutils import build_ext

import numpy

extensions = [
  Extension(
     "lib.lda.utils.util_funcs",
     ["lib/lda/utils/util_funcs.pyx"],
     include_dirs=[numpy.get_include()]
  )
]

setup(
    name="lib",
    version="0.1",
    packages=find_packages(),#['lib'],
    cmdclass = {'build_ext': build_ext},

    # Project uses reStructuredText, so ensure that the docutils get
    # installed or upgraded on the target machine
    install_requires=['numpy>=1.8', 'scipy>=0.10', 'nltk'],

    package_data={
        # If any package contains *.txt or *.rst files, include them:
        '': ['*.txt', '*.rst', '*.ini'],
        # And include any *.msg files found in the 'hello' package, too:
    },

    ext_modules=extensions,

    # metadata for upload to PyPI
    author="dslab topicmodel-lib team",
    author_email="",
    description="This is an LDA Package",
    license="MIT",
    keywords="",
    url="",   # project home page, if any

    # could also include long_description, download_url, classifiers, etc.
)


