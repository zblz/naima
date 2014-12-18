# Licensed under a 3-clause BSD style license - see LICENSE.rst
import numpy as np
from astropy.tests.helper import pytest

try:
    import sherpa
    HAS_SHERPA = True
except ImportError:
    HAS_SHERPA = False

@pytest.mark.skipif('not HAS_SHERPA')
def test_electron_models():
    """
    test import
    """

    from ..sherpamod import InverseCompton, Synchrotron, PionDecay

    energies = np.logspace(8,10,100) # 0.1 to 10 TeV in keV

    for modelclass in [InverseCompton, Synchrotron]:
        model = modelclass()

        model.ampl = 1e36
        model.index = 2.1

        # point calc
        output = model.calc([p.val for p in model.pars],energies)

        # integrated
        output = model.calc([p.val for p in model.pars],energies[:-1],xhi=energies[1:])


@pytest.mark.skipif('not HAS_SHERPA')
def test_proton_model():
    """
    test import
    """

    from ..sherpamod import PionDecay

    energies = np.logspace(8,10,10) # 0.1 to 10 TeV in keV

    model = PionDecay()

    model.ampl = 1e36
    model.index = 2.1

    # point calc
    output = model.calc([p.val for p in model.pars],energies)

    # integrated
    output = model.calc([p.val for p in model.pars],energies[:-1],xhi=energies[1:])


