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

    python setup.py install

or, to pull also the needed dependencies, use pip as 

::

    python setup.py sdist
    pip install dist/gammafit-0.1.tar.gz


Usage
-----

The package consists of several convenience functions to which you must provide
a model function, a probability function for the parameter priors, and the
spectral data to be fit enclosed in a dictionary. An example is shown in the
file ``velax_demo.py``.

ToDo
----

- For X-ray (sync) and GeV/TeV (IC) data, estimate initial magnetic field from
  Fx/Fvhe ratio.
- Write Convenience function to compute decorrelation energy of data and the
  corresponding electron and/or proton energy to be used as ``norm_energy``.
- Write method to derive lower/upper limits to non-present features, such as
  energy cutoffs. See `arXiv:gr-qc/0504042
  <http://arxiv.org/abs/gr-qc/0504042v1>`_ for an example.

