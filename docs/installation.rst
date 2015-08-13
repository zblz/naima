Installation
============

Requirements
------------

naima requires Python 2.6, 2.7, 3.2, 3.3, or 3.4, and the following
packages to be installed:

* `Numpy <http://www.numpy.org>`_

* `Scipy <http://www.scipy.org>`_

* `Astropy`_

* `Matplotlib <http://www.matplotlib.org>`_

* `emcee <http://dan.iel.fm/emcee>`_

The package `triangle_plot <https://github.com/dfm/triangle.py>`_ is also
very useful to inspect the result of the MCMC run through a corner plot.

All of the above packages are available in a typical scientific python
installation (or in all-in-one Python installations such as the `Anaconda Python
Distribution <http://continuum.io/downloads>`_) or can be installed through
``pip``.

Installing naima
----------------

Anaconda python distribution
++++++++++++++++++++++++++++

The `Anaconda python distribution <http://continuum.io/downloads>`_ allows to
easily set up a fully working scientific Python distribution in any Linux or Mac
machine, even without root access. Once Anaconda is set up, ``naima`` and all of
its dependencies can be installed in an Anaconda distribution through the
Astropy conda channel::

    $ conda config --add channels astropy
    $ conda install naima

To update to the latest version, you can either update all packages in the conda
distribution::

    $ conda update --all

or only ``naima``::

    $ conda update naima

Note that if you want to use the :ref:`sherpamod` you have to use a Python 2
version of Anaconda, as sherpa is not yet compatible with Python 3. Otherwise,
all dependencies are available in both Python 2 and 3.

Using pip
+++++++++

You can install ``naima`` in an existing Python installation through pip (you
may need to use ``sudo`` if you want to install it system-wide, or the flag
``--user`` to install only for the current user)::

    $ pip install naima

Note that installing with pip means that all non-installed dependencies will be
downloaded as source and built in your machine. For pure Python packages such as
``naima`` or ``emcee`` that is not a problem, but if Numpy or matplotlib are
installed this way the build can take quite a long time.

Installing the development version
++++++++++++++++++++++++++++++++++

To install from the latest development source, install ``naima`` from the
`github repository`_ through pip::

    $ pip install git+http://github.com/zblz/naima.git#egg=naima

.. _github repository: https://github.com/zblz/naima
