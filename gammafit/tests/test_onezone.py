# Licensed under a 3-clause BSD style license - see LICENSE.rst
from astropy import units as u
import numpy as np
from numpy.testing import assert_approx_equal
from astropy.tests.helper import pytest

try:
    import emcee
    HAS_EMCEE = True
except ImportError:
    HAS_EMCEE = False


electronozmpars={
        'seedspec':'CMB',
        'index':2.0,
        'cutoff':1e13,
        'beta':1.0,
        'ngamd':100,
        'gmin':1e4,
        'gmax':1e10,
        }


@pytest.mark.skipif('not HAS_EMCEE')
def test_electronozm():
    from ..onezone import ElectronOZM

    ozm = ElectronOZM( np.logspace(0,15,1000), 1, **electronozmpars)
    ozm.calc_outspec()

    lsy=np.trapz(ozm.specsy*ozm.outspecene**2*u.eV.to('erg'),ozm.outspecene)
    assert_approx_equal(lsy.value,0.016769058688230903)
    lic=np.trapz(ozm.specic*ozm.outspecene**2*u.eV.to('erg'),ozm.outspecene)
    assert_approx_equal(lic.value,214080823.28721327)


#def test_electronozm_evolve():
    #from ..onezone import ElectronOZM

    #ozm = ElectronOZM( np.logspace(0,15,1000), 1, evolve_nelec=True, **electronozmpars)
    #ozm.calc_outspec()

    #lsy=np.trapz(ozm.specsy*ozm.outspecene**2*u.eV.to('erg'),ozm.outspecene)
    #assert_approx_equal(lsy,5718447729.5694494)
    #lic=np.trapz(ozm.specic*ozm.outspecene**2*u.eV.to('erg'),ozm.outspecene)
    #assert_approx_equal(lic,1.0514223815442389e+20)


@pytest.mark.skipif('not HAS_EMCEE')
def test_protonozm():
    from ..onezone import ProtonOZM

    ozm = ProtonOZM( np.logspace(8,15,100), 1, index=2.0,cutoff=1e13,beta=1.0)
    ozm.calc_outspec()

    lpp=np.trapz(ozm.specpp*ozm.outspecene**2*u.eV.to('erg'),ozm.outspecene)
    assert_approx_equal(lpp.value,3.2800253974151616e-4, significant=5)

