Installation
============

Requirements
------------

naima is tested on Python 3.6 and 3.7 but also works in 3.4 and 3.5. It
also requires the following packages to be installed: `Numpy
<http://www.numpy.org>`_, `Scipy <http://www.scipy.org>`_, `Astropy`_,
`Matplotlib <http://www.matplotlib.org>`_, `emcee <http://dan.iel.fm/emcee>`_,
`corner <http://github.com/dfm/corner.py>`_, and `h5py <http://www.h5py.org>`_.
These will be installed automatically if you follow one of the installation
methods below.

The :ref:`sherpamod` can be used if the Sherpa package is installed.

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
machine, even without root access. Once Anaconda is set up, Naima and all of
its dependencies can be installed in an Anaconda distribution through the
Astropy conda channel

.. code-block:: shell

    $ conda config --add channels astropy
    $ conda install naima

To update to the latest version, you can either update all packages in the conda
distribution

.. code-block:: shell

    $ conda update --all

or only Naima

.. code-block:: shell

    $ conda update naima

Using pip
+++++++++

You can install Naima in an existing Python installation through pip. It is
recommended to install it in its own virtual environment, otherwise you may need
to use ``sudo`` if you want to install it system-wide, or the flag ``--user`` to
install only for the current user

.. code-block:: shell

    $ pip install naima

Note that installing with pip means that all non-installed dependencies will be
downloaded as source and built in your machine. For pure Python packages such as
Naima or ``emcee`` that is not a problem, but if Numpy or matplotlib are
installed this way the build can take quite a long time. You can bypass this
problem by installing these libraries through your systemwide package manager:
see below for how to do this for different systems.

Installing the development version
++++++++++++++++++++++++++++++++++

To install from the latest development source, install Naima from the
`github repository`_ through pip

.. code-block:: shell

    $ pip install git+http://github.com/zblz/naima.git#egg=naima

.. _github repository: https://github.com/zblz/naima
