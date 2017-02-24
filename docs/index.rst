Welcome to Naima
================

.. comment: split here

Naima is a Python package for computation of non-thermal radiation from
relativistic particle populations. It includes tools to perform MCMC fitting of
radiative models to X-ray, GeV, and TeV spectra using `emcee
<http://dan.iel.fm/emcee>`_, an affine-invariant ensemble sampler for Markov
Chain Monte Carlo. Naima is an `Astropy`_ affiliated
package.

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

Use the sidebar on the left to access the documentation.


.. toctree::
   :hidden:
   :maxdepth: 2
 
   installation.rst
   radiative.rst
   mcmc.rst
   tutorial.rst
   sherpa.rst
   examples.rst


.. toctree::
   :hidden:
   :maxdepth: 2
 
   dataformat.rst
   units.rst
   api.rst

License & Attribution
---------------------

Naima is released under a 3-clause BSD style license - see the
`LICENSE.rst <https://github.com/zblz/naima/blob/master/LICENSE.rst>`_ for
details.

If you find Naima useful in your research, you can cite `Zabalza (2015)
<http://arxiv.org/abs/1509.03319>`_ (`arXiv <http://arxiv.org/abs/1509.03319>`_,
`ADS <http://adsabs.harvard.edu/abs/2015arXiv150903319Z>`_) to acknowledge its
use. The BibTeX entry for the paper is::

    @ARTICLE{naima,
       author = {{Zabalza}, V.},
        title = {naima: a Python package for inference of relativistic particle
                 energy distributions from observed nonthermal spectra},
         year = 2015,
      journal = {Proc.~of International Cosmic Ray Conference 2015},
        pages = "922",
       eprint = {1509.03319},
       adsurl = {http://adsabs.harvard.edu/abs/2015arXiv150903319Z},
    }


Contributing
------------

Please report any issues with the package `here
<https://github.com/zblz/naima/issues>`_.

All development of Naima is done through the `github repository`_, and
contributions to the code are welcome.  The development model is similar to that
of `astropy`_, so you can check the `astropy Developer Documentation
<https://astropy.readthedocs.org/en/latest/#developer-documentation>`_ if you
need information on how to make a code contribution.

.. _github repository: https://github.com/zblz/naima
