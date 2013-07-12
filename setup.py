#!/usr/bin/env python
# -*- coding: utf8 -*-

from setuptools import setup

setup(name='gammafit',
      version='0.1',
      description='Derivation of non-thermal particle distributions through MCMC spectral fitting',
      author='Víctor Zabalza',
      author_email='vzabalza@gmail.com',
      py_modules=["gammafit"],
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
