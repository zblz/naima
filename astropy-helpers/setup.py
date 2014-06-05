#!/usr/bin/env python
# Licensed under a 3-clause BSD style license - see LICENSE.rst

import ah_bootstrap
import pkg_resources
from setuptools import setup
from astropy_helpers.setup_helpers import register_commands, get_package_info
from astropy_helpers.version_helpers import generate_version_py

NAME = 'astropy_helpers'
VERSION = '0.4.dev'
RELEASE = 'dev' not in VERSION
DOWNLOAD_BASE_URL = 'http://pypi.python.org/packages/source/a/astropy-helpers'

generate_version_py(NAME, VERSION, RELEASE, False)

# Use the updated version including the git rev count
from astropy_helpers.version import version as VERSION

cmdclass = register_commands(NAME, VERSION, RELEASE)
# This package actually doesn't use the Astropy test command
del cmdclass['test']

setup(
    name=pkg_resources.safe_name(NAME),  # astropy_helpers -> astropy-helpers
    version=VERSION,
    description='',
    author='The Astropy Developers',
    author_email='astropy.team@gmail.com',
    license='BSD',
    url='http://astropy.org',
    long_description='',
    download_url='{0}/astropy-{1}.tar.gz'.format(DOWNLOAD_BASE_URL, VERSION),
    classifiers=[],
    cmdclass=cmdclass,
    zip_safe=False,
    **get_package_info(exclude=['astropy_helpers.tests'])
)
