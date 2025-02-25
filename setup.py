import setuptools

setuptools.setup(
    name = "fastgp",
    version = "1.0",
    author = "Aleksei Sorokin",
    author_email = "asorokin@hawk.iit.edu",
    license = 'Apache license 2.0',
    description = "Fast Gaussian process regression",
    long_description_content_type = "text/markdown",
    url = "https://github.com/alegresor/FastGaussianProcesses",
    packages = [
        'fastgp',
    ],
    install_requires = [
        "qmcpy >= 1.6.2",
        "torch >= 2.0.0",
        "numpy >= 2.0.0",
        "scipy >= 2.0.0",
    ],
    classifiers = [
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: Apache Software License",
        "Operating System :: OS Independent"],
    keywords = [
        'fast',
        'Gaussian',
        'process',
        'regression',
        'low discrepancy',
        'quasi-random',
        'lattice',
        'digital net',
        'shift invariant',
        'digitally shift invariant'
    ],
    python_requires = ">= 3.5",
    include_package_data = True,
)