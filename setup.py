#!/usr/bin/env python3

from os import path

from setuptools import setup, find_packages

with open(path.join(path.abspath(path.dirname(__file__)), "README.md")) as f:
    long_description = f.read()

setup(
    name="Network Flow Generator",
    version="0.9.0",
    author="Andre Lehmann",
    author_email="aisberg@posteo.de",
    description="",
    long_description=long_description,
    long_description_content_type="text/markdown",
    classifiers=[
        "Development Status :: 4 - Beta",
        "Environment :: Console",
        "Intended Audience :: Developers",
        "Intended Audience :: Information Technology",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3 :: Only",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Scientific/Engineering :: Information Analysis",
    ],
    packages=find_packages(exclude=["tests"]),
    scripts=["bin/network-flow-generator"],
    install_requires=[
        "numpy==1.17.*",
        "pandas==0.25.*",
        "tensorflow==2.0.*",
        "holoviews==1.12.*",
    ],
    extras_require={
        "colorlog":  ["colorlog==4.0.*"],
    },
    include_package_data=True,
    zip_safe=False,
)
