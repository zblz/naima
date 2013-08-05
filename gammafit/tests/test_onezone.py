from astropy import units as u
import numpy as np
from numpy.testing import assert_approx_equal

electronozmpars={
        'seedspec':'CMB',
        'bb':True,
        'nbb':10,
        'index':2.0,
        'cutoff':1e13,
        'beta':1.0,
        'ngamd':100,
        'gmin':1e4,
        'gmax':1e10,
        }



def test_electronozm():
    from ..onezone import ElectronOZM

    ozm = ElectronOZM( np.logspace(0,15,1000), 1, **electronozmpars)
    ozm.calc_outspec()

    lsy=np.trapz(ozm.specsy*ozm.outspecene**2*u.eV.to('erg'),ozm.outspecene)
    assert_approx_equal(lsy,0.016769058688230903)
    lic=np.trapz(ozm.specic*ozm.outspecene**2*u.eV.to('erg'),ozm.outspecene)
    assert_approx_equal(lic,212291612.87657347)

def test_electronozm_evolve():
    from ..onezone import ElectronOZM

    ozm = ElectronOZM( np.logspace(0,15,1000), 1, evolve_nelec=True, **electronozmpars)
    ozm.calc_outspec()

    lsy=np.trapz(ozm.specsy*ozm.outspecene**2*u.eV.to('erg'),ozm.outspecene)
    assert_approx_equal(lsy,5718447729.5694494)
    lic=np.trapz(ozm.specic*ozm.outspecene**2*u.eV.to('erg'),ozm.outspecene)
    assert_approx_equal(lic,1.0514223815442389e+20)

def test_protonozm():
    from ..onezone import ProtonOZM

    ozm = ProtonOZM( np.logspace(8,15,100), 1, index=2.0,cutoff=1e13,beta=1.0)
    ozm.calc_outspec()

    lpp=np.trapz(ozm.specpp*ozm.outspecene**2*u.eV.to('erg'),ozm.outspecene)
    assert_approx_equal(lpp,3.2800627079738687e+23, significant=5)

