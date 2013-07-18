from astropy import units as u
import numpy as np

def assert_almost_equal(x,y):
    assert abs(x-y)/y < 1e-6

def test_electronozm():
    from ..onezone import ElectronOZM

    ozm = ElectronOZM( np.logspace(0,15,1000), 1, seedspec=['CMB',], bb=True,nbb=10,index=2.0,cutoff=1e13,beta=1.0)
    ozm.calc_outspec()

    lsy=np.trapz(ozm.specsy*ozm.outspecene**2*u.eV.to('erg'),ozm.outspecene)
    assert_almost_equal(lsy,0.071974056347549428)
    lic=np.trapz(ozm.specic*ozm.outspecene**2*u.eV.to('erg'),ozm.outspecene)
    assert_almost_equal(lic,212255584.80106944)

def test_electronozm_evolve():
    from ..onezone import ElectronOZM

    ozm = ElectronOZM( np.logspace(0,15,1000), 1, seedspec=['CMB',], bb=True,nbb=10,index=2.0,cutoff=1e13,beta=1.0,evolve_nelec=True)
    ozm.calc_outspec()

    lsy=np.trapz(ozm.specsy*ozm.outspecene**2*u.eV.to('erg'),ozm.outspecene)
    assert_almost_equal(lsy,24600421942.782776)
    lic=np.trapz(ozm.specic*ozm.outspecene**2*u.eV.to('erg'),ozm.outspecene)
    assert_almost_equal(lic,1.0455302369906613e+20)

def test_protonozm():
    from ..onezone import ProtonOZM

    ozm = ProtonOZM( np.logspace(8,15,100), 1, index=2.0,cutoff=1e13,beta=1.0)
    ozm.calc_outspec()

    lpp=np.trapz(ozm.specpp*ozm.outspecene**2*u.eV.to('erg'),ozm.outspecene)
    assert_almost_equal(lpp,3.2800627079738687e+23)

