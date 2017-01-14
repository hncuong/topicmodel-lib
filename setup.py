from setuptools import setup, find_packages
setup(
    name="tmlib",
    version="0.1",
    packages=find_packages(),

    # Project uses reStructuredText, so ensure that the docutils get
    # installed or upgraded on the target machine
    install_requires=['docutils>=0.3', 'numpy>=1.8', 'scipy>=0.10', 'nltk'],

    package_data={
        # If any package contains *.txt or *.rst files, include them:
        '': ['*.txt', '*.rst', '*.ini'],
        # And include any *.msg files found in the 'hello' package, too:
    },

    # metadata for upload to PyPI
    author="dslab tmlib team",
    author_email="",
    description="This is an LDA Package",
    license="MIT",
    keywords="",
    url="",   # project home page, if any

    # could also include long_description, download_url, classifiers, etc.
)
