.. _sherpamod:

Sherpa models
=============

The `sherpa`_ package is a modeling and fitting application which can be used to
fit a variety of data from spatial morphology to forward-folding spectral
analysis. It is part of the Chandra analysis software pacakage (`CIAO
<http://cxc.cfa.harvard.edu/ciao/>`_), but it can be easily installed standalone
by using either the `Anaconda Python <http://continuum.io/downloads>`_ distribution
or with pip.

The `standalone Python version of sherpa
<https://sherpa.readthedocs.io/>`_  can be installed using Anaconda along with
Naima as follows

.. code-block:: shell

    $ export PATH=PATH_TO_ANACONDA/bin:$PATH
    $ conda config --add channels sherpa --add channels astropy
    $ conda install sherpa naima

where ``PATH_TO_ANACONDA`` is the path where you have installed Anaconda.

Sherpa can also be installed using pip, as long as NumPy is already
installed

.. code-block:: shell

    $ pip install sherpa

Naima provides wrappers for the :ref:`radiative` to make it easier to use
them in a sherpa session. The models available for use in sherpa are the four
radiative models available in Naima (see :ref:`radiative`) with a
`~naima.models.PowerLaw` or `~naima.models.ExponentialCutoffPowerLaw` particle
distribution:

    - `naima.sherpa_models.InverseCompton`: wrapper of `~naima.models.InverseCompton`
    - `naima.sherpa_models.Synchrotron`: wrapper of `~naima.models.Synchrotron`
    - `naima.sherpa_models.Bremsstrahlung`: wrapper of `~naima.models.Bremsstrahlung`
    - `naima.sherpa_models.PionDecay`: wrapper of `~naima.models.PionDecay`

Once within a python session or script, these models can be accesed through
`naima.sherpa_models` and added to an analysis session with the sherpa command
`set_model`. You can see the available parameters with `show_model`

.. code-block:: pycon

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

Initially, only the amplitude and index of the particle distribution are free
parameters, see the `sherpa`_ documentation for information of how to modify the
frozen parameters and thaw them. Note that sherpa models do not accept
parameters as `~astropy.units.Quantity` objects given that their units are
fixed. You can see the units for each of the parameters with the `show_model`
sherpa command.

.. _sherpa: http://cxc.cfa.harvard.edu/sherpa/
