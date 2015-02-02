naima Documentation
======================

.. comment: split here

``naima`` uses MCMC fitting of non-thermal X-ray, GeV, and TeV spectra to
constrain the properties of their parent relativistic particle distributions.
The workhorse of ``naima`` is the powerful `emcee <http://dan.iel.fm/emcee>`_
affine-invariant ensemble sampler for Markov chain Monte Carlo.

.. comment: - Code: http://www.github.com/zblz/naima

.. comment: - Documentation: http://naima.readthedocs.org

.. comment: split here

There are two main components of the package: a set of nonthermal
:ref:`radiative`, and a set of utility functions that make it easier to fit a
given model to observed spectral data (see :ref:`MCMC`).

Nonthermal radiative models are available for Synchrotron, inverse Compton,
Bremsstrahlung, and neutral pion decay processes. All of the models allow the
use of an arbitrary shape of the particle energy distribution, and several
functional models are also available to be used as particle distribution
functions. See :ref:`radiative` for a detailed explanation of these.

User documentation
------------------

.. toctree::
   :maxdepth: 1
 
   installation.rst
   MCMC.rst
   radiative.rst
   tutorial.rst
   examples.rst

Appendices
----------


.. toctree::
   :maxdepth: 1
 
   dataformat.rst
   units.rst
   api.rst

Contributing
------------

Please report any issues with the package `here
<https://github.com/zblz/naima/issues>`_.

All development of ``naima`` is done through the `github repository`_, and
contributions to the code are welcome.  The development model is similar to that
of `astropy`_, so you can check the `astropy Developer Documentation
<https://astropy.readthedocs.org/en/latest/#developer-documentation>`_ if you
need information on how to make a code contribution.

.. _github repository: https://github.com/zblz/naima
