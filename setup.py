#!/usr/bin/env python3

import os, sys
from setuptools import setup, find_packages

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "mcc"))

setup(
    name="mcc",
    version="1.0",
    packages=find_packages(
        exclude=["tests", "*.tests", "*.tests.*", "tests.*"],
    ),

    install_requires=[
        'numpy',
        'matplotlib',
        'ortools',
        'scikit-learn',
        'tqdm',
        'pyyaml'
        'prettytable'
    ],
    include_package_data=True,
)