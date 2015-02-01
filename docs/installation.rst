Installation
============

Requirements
------------

naima requires Python 2.6, 2.7, 3.2, 3.3, or 3.4, and the following
packages to be installed:

* `Numpy <http://www.numpy.org>`_

* `Scipy <http://www.scipy.org>`_

* `Astropy 0.4 or later <http://www.astropy.org>`_

* `Matplotlib <http://www.matplotlib.org>`_

* `emcee <http://dan.iel.fm/emcee>`_

The package `triangle_plot <https://github.com/dfm/triangle.py>`_ is also
very useful to inspect the result of the MCMC run through a corner plot.

All of the above packages are available in a typical scientific python
installation (or in all-in-one Python installations such as the `Anaconda Python
Distribution <http://continuum.io/downloads>`_) or can be installed through
``pip``.

Installation
------------

Th lastest version can be installed through pip (You may need to use ``sudo`` if
you want to install it system-wide, or the flag ``--user`` to install only for
the current user)::

    pip install naima

To install from the latest development source (recommended until a stable
version is released), install ``naima`` from the `github repository`_ through pip::

    pip install git+http://github.com/zblz/naima.git#egg=naima

.. _github repository: https://github.com/zblz/naima
