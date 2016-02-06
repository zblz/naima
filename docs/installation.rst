Installation
============

Requirements
------------

naima requires Python 2.7, 3.4, or 3.5 and the following packages to be
installed: `Numpy <http://www.numpy.org>`_, `Scipy <http://www.scipy.org>`_,
`Astropy`_, `Matplotlib <http://www.matplotlib.org>`_, `emcee
<http://dan.iel.fm/emcee>`_, `corner <http://github.com/dfm/corner.py>`_, and
`h5py <http://www.h5py.org>`_. These will be installed automatically if you
follow one of the installation methods below.

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
Astropy conda channel::

    $ conda config --add channels astropy
    $ conda install naima

To update to the latest version, you can either update all packages in the conda
distribution::

    $ conda update --all

or only Naima::

    $ conda update naima

Note that if you want to use the :ref:`sherpamod` you have to use a Python 2
version of Anaconda, as sherpa is not yet compatible with Python 3. Otherwise,
all dependencies are available in both Python 2 and 3.

Using pip
+++++++++

You can install Naima in an existing Python installation through pip (you
may need to use ``sudo`` if you want to install it system-wide, or the flag
``--user`` to install only for the current user)::

    $ pip install naima

Note that installing with pip means that all non-installed dependencies will be
downloaded as source and built in your machine. For pure Python packages such as
Naima or ``emcee`` that is not a problem, but if Numpy or matplotlib are
installed this way the build can take quite a long time. You can bypass this
problem by installing these libraries through your systemwide package manager:
see below for how to do this for different systems.

apt-get (Debian based)
~~~~~~~~~~~~~~~~~~~~~~

For Debian-based systems (including Ubuntu), this can be done for Python 2.7
environments with::

    $ sudo apt-get install python-matplotlib python-scipy python-astropy python-h5py
    $ pip install naima

for Python 3 environments, the name of the packages should be::

    $ sudo apt-get install python3-matplotlib python-scipy python-astropy python-h5py
    $ pip3 install naima

Macports
~~~~~~~~

Macports can be used on Mac systems to install most dependencies::

    $ export PY=py34 # change to py27 for Python 2.7
    $ sudo port install $PY-pip $PY-scipy $PY-matplotlib $PY-emcee $PY-h5py \
        $PY-astropy
    $ pip install --user naima


Installing the development version
++++++++++++++++++++++++++++++++++

To install from the latest development source, install Naima from the
`github repository`_ through pip::

    $ pip install git+http://github.com/zblz/naima.git#egg=naima

.. _github repository: https://github.com/zblz/naima
