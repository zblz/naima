Spectra fitting with emcee
==========================

``emcee_specfit`` is a set of functions designed to make MCMC spectral fitting
easier. It is based on the powerful `emcee <http://dan.iel.fm/emcee>`_ MCMC sampling package.

Installation
------------

Just run

::

    python setup.py install

or, to pull also the needed dependencies, use pip as 

::

    python setup.py sdist
    pip install dist/emcee_specfit-0.1.tar.gz


Usage
-----

The package consists of several convenience functions to which you must provide
a model function, a probability function for the parameter priors, and the
spectral data to be fit enclosed in a dictionary. An example is shown in the
file `velax_demo.py`.
