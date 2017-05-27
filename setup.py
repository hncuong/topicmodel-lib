from setuptools import setup, find_packages
from setuptools import Extension
from Cython.Build import cythonize
from Cython.Distutils import build_ext

import numpy

extensions = [
  Extension(
     "tmlib.lda.utils.util_funcs",
     ["tmlib/lda/utils/util_funcs.pyx"],
     include_dirs=[numpy.get_include()]
  )
]

setup(
    name="tmlib",
    version="0.1",
    packages=find_packages(),#['lib'],
    cmdclass = {'build_ext': build_ext},

    # Project uses reStructuredText, so ensure that the docutils get
    # installed or upgraded on the target machine
    install_requires=['numpy>=1.8', 'scipy>=0.10', 'nltk', 'Cython'],

    package_data={
        # If any package contains *.txt or *.rst files, include them:
        '': ['*.txt', '*.rst', '*.ini'],
        # And include any *.msg files found in the 'hello' package, too:
    },

    ext_modules=extensions,

    # metadata for upload to PyPI
    author="dslab topicmodel-tmlib team",
    author_email="truongkhang95@gmail.com",
    description="This is a LDA Package",
    license="MIT",
    keywords=['ordination', 'LDA', 'topic modeling'],
    url="https://github.com/hncuong/topicmodel-lib",   # project home page, if any
    #classifiers=[
    #    'Development Status :: 1 - Planning',
    #    'Intended Audience ::  Science/Research',
    #    'License :: OSI Approved :: MIT License',
    #    'Programming Language :: Python :: 2.7',
    #],

    # could also include long_description, download_url, classifiers, etc.
)