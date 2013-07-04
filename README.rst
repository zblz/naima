Non-thermal spectral fitting with emcee
=======================================

``emcee_specfit`` is a set of functions designed to make MCMC spectral fitting
of non-thermal emission easier. It is based on the powerful `emcee
<http://dan.iel.fm/emcee>`_ MCMC sampling package.


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

ToDo
----

- ProtonOZM for pp interactions
- Write standard labels for parameters as a property of
  ``emcee.EnsembleSampler`` so that they can be accessed by other functions
  without knowing their order.
- For X-ray (sync) and GeV/TeV (IC) data, estimate initial magnetic field from
  Fx/Fvhe ratio.

