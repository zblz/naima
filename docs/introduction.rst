Introduction
============

``gammafit`` uses MCMC fitting of non-thermal X-ray, GeV, and TeV spectra to
constrain the properties of their parent relativistic particle distributions. 
The workhorse of the MCMC fitting is the powerful `emcee
<http://dan.iel.fm/emcee>`_ affine-invariant ensemble sampler for Markov chain
Monte Carlo.

There are two main components of the package: a set of nonthermal radiative
models, and a set of utility functions that make it easier to fit a given model
to observed spectral data.

Nonthermal radiative models are available for Synchrotron, inverse Compton,
Bremsstrahlung, and neutral pion decay processes. All of the models allow the
use of an arbitrary shape of the particle energy distribution, and several
functional models are also available to be used as particle distribution
functions.


