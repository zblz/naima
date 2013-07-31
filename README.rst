Derivation of non-thermal particle distribution
===============================================

``gammafit`` uses MCMC fitting of non-thermal X-ray, GeV, and TeV spectra to
constrain the properties of their parent relativistic particle distributions. 

The workhorse of ``gammafit`` is the powerful `emcee
<http://dan.iel.fm/emcee>`_ affine-invariant ensemble sampler for Markov chain
Monte Carlo.


Installation
------------

To install from the latest development source (recommended until a version
stable is released), clone this repository

::

    git clone http://github.com/zblz/gammafit

enter directory ``gammafit``, and run

::

    python setup.py install --user


Usage
-----

The package consists of several convenience functions to which you must provide
a model function, a probability function for the parameter priors, and the
spectral data to be fit enclosed in a dictionary. Three examples (derivation of
electron and proton distributions, as well as function fitting) are shown in the
directory ``examples``. Preliminary documentation can be found at
`gammafit.readthedocs.org <http://gammafit.readthedocs.org>`_.

Attribution
-----------

A publication describing the radiation models and the method for derivation of
the particle distribution is in preparation. In the meantime, if you use
``gammafit`` for your research, please link back to this webpage when mentioning
``gammafit`` in your publication.
