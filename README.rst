Derivation of non-thermal particle distribution
===============================================

``gammafit`` uses MCMC fitting of non-thermal X-ray, GeV, and TeV spectra to
constrain the properties of their parent relativistic particle distributions. 

The workhorse of ``gammafit`` is the powerful `emcee
<http://dan.iel.fm/emcee>`_ affine-invariant ensemble sampler for Markov chain
Monte Carlo.


Installation
------------

For a system-wide installation, run

::

    sudo pip install gammafit

or, for a user-only installation, run

::

    pip install --user gammafit


If you want to install from the latest development source (recommended until a
version stable is released), clone this repository

::

    git clone https://github.com/zblz/gammafit

enter directory ``gammafit``, and run

::

    python setup.py install --user


Usage
-----

The package consists of several convenience functions to which you must provide
a model function, a probability function for the parameter priors, and the
spectral data to be fit enclosed in a dictionary. An example is shown in the
file ``velax_demo.py``. Preliminary documentation can be found at
`gammafit.readthedocs.org <http://gammafit.readthedocs.org>`_.
