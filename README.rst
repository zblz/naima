Non-thermal spectral fitting with emcee
=======================================

``gammafit`` uses MCMC fitting of non-thermal X-ray, GeV, and TeV spectra to
constrain the properties of their parent relativistic particle distributions. 

The workhorse of ``gammafit`` is the powerful `emcee
<http://dan.iel.fm/emcee>`_ affine-invariant ensemble sampler for Markov chain
Monte Carlo.


Installation
------------

Just run

::

    [sudo] pip install gammafit


Usage
-----

The package consists of several convenience functions to which you must provide
a model function, a probability function for the parameter priors, and the
spectral data to be fit enclosed in a dictionary. An example is shown in the
file ``velax_demo.py``.
