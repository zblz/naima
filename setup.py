#!/usr/bin/env python
# -*- coding: utf8 -*-

from setuptools import setup

setup(name='emcee_specfit',
      version='0.1',
      description='MCMC spectral fitting with emcee',
      author='Víctor Zabalza',
      author_email='vzabalza@gmail.com',
      py_modules=["emcee_specfit"],
      long_description=open("README.rst").read(),
      requires = [
          'emcee (>=1.2.0)',
          'triangle_plot',
          'numpy',
          'matplotlib',
          'astropy (>=0.2)',
          'scipy',
          ]
     )
