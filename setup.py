#!/usr/bin/env python
# Licensed under a 3-clause BSD style license - see LICENSE.rst

from setuptools import setup, find_packages

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
        "astropy>=1.0.2",
        "emcee>=2.2.0,<3.0",
        "corner",
        "matplotlib",
        "scipy",
        "h5py",
        "pyyaml",
    ],
    python_requires=">=3.5",
)
