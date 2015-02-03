Sherpa models
=============

The `sherpa`_ package is a modeling and fitting application which can be used to
fit a variety of data from spatial morphology to forward-folding spectral
analysis. It is part of the Chandra analysis software pacakage (`CIAO`_), but it
can be easily installed standalone by using the `Anaconda Python
<http://continuum.io/downloads>`_ distribution. Once you have a Python 2.7
Anaconda environment setup, installing `sherpa`_ 4.7b is done as follows::

    $ export PATH=PATH_TO_ANACONDA/bin:$PATH
    $ conda config --add channels https://conda.binstar.org/cxc
    $ conda install sherpa

where ``PATH_TO_ANACONDA`` is the path where you have installed Anaconda. 


``naima`` provides wrappers for the :ref:`radiative` to make it easier to use
them in a sherpa session.  Note that you will also have to install ``naima`` in
the Anaconda environment::

    $ pip install naima

The models available for use in sherpa are the four radiative models available
in ``naima`` (see :ref:`radiative`) with a `~naima.models.PowerLaw` or
`~naima.models.ExponentialCutoffPowerLaw` particle distribution. Note that for
sherpa models, the parameters are not given as `~astropy.units.Quantity`
objects, but only as floats, and their units are fixed. 

Once within a python session or script, these models can be accesed through
`naima.sherpa_models`::

    >>> from sherpa.astro.ui import *
    >>> dataspace1d(0.1,10,0.1) # Data would be loaded at this step, here we fake it
    >>> from naima.sherpa_models import InverseCompton
    >>> set_model(InverseCompton('IC'))
    >>> show_model()
    Model: 1
    IC
       Param        Type          Value          Min          Max      Units
       -----        ----          -----          ---          ---      -----
       IC.index     thawed          2.1          -10           10
       IC.ref       frozen           60            0  3.40282e+38        TeV
       IC.ampl      thawed          100            0        1e+60    1e30/eV
       IC.cutoff    frozen            0            0  3.40282e+38        TeV
       IC.beta      frozen            1            0           10
       IC.TFIR      frozen           70            0  3.40282e+38          K
       IC.uFIR      frozen            0            0  3.40282e+38     eV/cm3
       IC.TNIR      frozen         3800            0  3.40282e+38          K
       IC.uNIR      frozen            0            0  3.40282e+38     eV/cm3
       IC.distance  frozen            1            0        1e+06        kpc
       IC.verbose   frozen            0            0  3.40282e+38




.. _sherpa: http://cxc.cfa.harvard.edu/contrib/sherpa/
