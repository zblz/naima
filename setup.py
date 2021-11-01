#!/usr/bin/env python
# Licensed under a 3-clause BSD style license - see LICENSE.rst

from setuptools import find_packages, setup

setup(
    use_scm_version={
        "version_scheme": "post-release",
        "local_scheme": "dirty-tag",
    },
    setup_requires=["setuptools_scm"],
    packages=find_packages("src"),
    package_dir={"": "src"},
    package_data={"naima": ["data/*.npz"]},
    install_requires=[
        "astropy>=4.3",
        "emcee~=3.1",
        "corner",
        "matplotlib",
        "scipy",
        "h5py",
        "pyyaml",
    ],
    python_requires=">=3.7",
)
