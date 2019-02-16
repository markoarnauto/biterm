from setuptools import setup, find_packages
from setuptools.extension import Extension
from Cython.Build import cythonize
import numpy

extensions = [
    Extension(
        "biterm.cbtm",
        ["biterm/cbtm.pyx","biterm/cbtm.c"],
        include_dirs=[numpy.get_include()]
    )
]

# pypi setup
with open("README.md", "r") as fh:
    long_description = fh.read()

setup(
    name="biterm",
    packages = find_packages(),
    version="0.1.3",
    author="markoarnauto",
    author_email="markus.tretzmueller@cortecs.at",
    description="Biterm Topic Model",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/markoarnauto/biterm",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    install_requires=[
        'numpy',
        'tqdm',
        'cython',
        'nltk'
    ],
    ext_modules=cythonize(extensions)
)