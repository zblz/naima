Spectra fitting with emcee
==========================

`emcee_specfit` is a set of functions designed to make MCMC spectral fitting
easier. It is based on the powerful [`emcee`](http://dan.iel.fm/emcee) MCMC sampling package.

Installation
------------

Just run

::

    pip install emcee_specfit

or 

::

    pip install emcee_specfit.tar.gz


Usage
-----

The package consists of several convenience functions to which you must provide
a model function, a probability function for the parameter priors, and the
spectral data to be fit enclosed in a dictionary. An example is shown in the
file `velax_demo.py`.
