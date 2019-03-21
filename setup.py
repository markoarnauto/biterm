from setuptools import setup, find_packages, dist
#dist.Distribution().fetch_build_eggs(['Cython>=0.15.1', 'numpy>=1.10'])
#from setuptools.extension import Extension
#from Cython.Build import cythonize
#import numpy

# extensions = [
#     Extension(
#         "biterm.cbtm",
#         ["biterm/cbtm.c"], #["biterm/cbtm.pyx"] #for compilation
#         include_dirs=[numpy.get_include()]
#     )
# ]

# pypi setup
with open("README.md", "r") as fh:
    long_description = fh.read()

setup(
    name="biterm",
    packages = find_packages(),
    version="0.2.0",
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
        #'cython',
        'nltk'
    ],
    #ext_modules=cythonize(extensions)
)