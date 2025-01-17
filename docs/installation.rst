Installation
============

Requirements
------------

Naima needs Python 3.9 or above. It also requires the following packages to be
installed: `Numpy <http://www.numpy.org>`_, `Scipy <http://www.scipy.org>`_, `Astropy`_,
`Matplotlib <http://www.matplotlib.org>`_, `emcee <https://emcee.readthedocs.io>`_,
`corner <http://github.com/dfm/corner.py>`_, and `h5py <http://www.h5py.org>`_.  These
will be installed automatically if you follow one of the installation methods below.

The :ref:`sherpamod` can be used if the Sherpa package is installed.

All of the above packages are available in a typical scientific python
installation (or in all-in-one Python installations such as the `Anaconda Python
Distribution <http://continuum.io/downloads>`_) or can be installed through
``pip``.

Installing naima
----------------

You can install Naima in an existing Python installation through pip. It is
recommended to install it in its own virtual environment, otherwise you may need
to use ``sudo`` if you want to install it system-wide, or the flag ``--user`` to
install only for the current user

.. code-block:: shell

    $ pip install naima

Installing the development version
++++++++++++++++++++++++++++++++++

To install from the latest development source, install Naima from the
`github repository`_ through pip

.. code-block:: shell

    $ pip install git+http://github.com/zblz/naima.git#egg=naima

.. _github repository: https://github.com/zblz/naima
