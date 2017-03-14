naima
=====

``naima`` is a Python package for computation of non-thermal radiation from
relativistic particle populations. It includes tools to perform MCMC fitting of
radiative models to X-ray, GeV, and TeV spectra using `emcee
<http://dan.iel.fm/emcee>`_, an affine-invariant ensemble sampler for Markov
Chain Monte Carlo.

``naima`` is named after a `ballad composed by John Coltrane in 1959
<https://en.wikipedia.org/wiki/Naima>`_ which appeared in the albums
`Giant Steps (1959) <https://www.youtube.com/watch?v=QTMqes6HDqU>`_ and
`Live at the Village Vanguard (1961) <https://www.youtube.com/watch?v=Tq3-99vbFt8>`_.

.. image:: http://img.shields.io/pypi/v/naima.svg
	:target: https://pypi.python.org/pypi/naima/
.. image:: http://img.shields.io/badge/license-BSD-green.svg
	:target: https://github.com/zblz/naima/blob/master/LICENSE.rst
.. image:: http://img.shields.io/badge/powered%20by-AstroPy-orange.svg
	:target: http://www.astropy.org
.. image:: http://img.shields.io/badge/arXiv-1509.03319-blue.svg
	:target: http://arxiv.org/abs/1509.03319

Documentation
^^^^^^^^^^^^^

Documentation is at `naima.readthedocs.org
<http://naima.readthedocs.org>`_.

Attribution
^^^^^^^^^^^

If you find ``naima`` useful in your research, you can cite `Zabalza (2015)
<http://arxiv.org/abs/1509.03319>`_ to acknowledge its use. The BibTeX entry for
the paper is::

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


License
^^^^^^^

Naima is released under a 3-clause BSD style license - see the
`LICENSE.rst <https://github.com/zblz/naima/blob/master/LICENSE.rst>`_ file.


Code status
^^^^^^^^^^^

.. image:: http://img.shields.io/travis/zblz/naima.svg
	:target: https://travis-ci.org/zblz/naima
.. image:: http://img.shields.io/coveralls/zblz/naima.svg
	:target: https://coveralls.io/r/zblz/naima
.. image:: http://img.shields.io/badge/benchmarked%20by-asv-green.svg
	:target: http://zblz.github.io/naima-benchmarks
