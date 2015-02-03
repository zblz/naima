# Licensed under a 3-clause BSD style license - see LICENSE.rst
from astropy import units as u
import numpy as np
from numpy.testing import assert_allclose
from astropy.tests.helper import pytest
from astropy.extern import six

from ..utils import trapz_loglog

try:
    import scipy
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False

e_0 = 20 * u.TeV
e_cutoff = 10 * u.TeV
alpha = 2.0
e_break = 1 * u.TeV
alpha_1 = 1.5
alpha_2 = 2.5

electron_properties = {'Eemin':100*u.GeV, 'Eemax':1*u.PeV}
proton_properties = {'Epmax':1*u.PeV}

energy = np.logspace(0, 15, 1000) * u.eV

from astropy.constants import m_e, c
pdist_unit = 1/u.Unit(m_e * c**2)

@pytest.fixture
def particle_dists():
    from ..models import ExponentialCutoffPowerLaw, PowerLaw, BrokenPowerLaw
    ECPL = ExponentialCutoffPowerLaw(amplitude=1*pdist_unit, e_0=e_0,
            alpha=alpha, e_cutoff=e_cutoff)
    PL = PowerLaw(amplitude=1*pdist_unit, e_0=e_0, alpha=alpha)
    BPL = BrokenPowerLaw(amplitude=1*pdist_unit, e_0=e_0, e_break=e_break,
            alpha_1=alpha_1, alpha_2=alpha_2)
    return ECPL,PL,BPL

@pytest.mark.skipif('not HAS_SCIPY')
def test_synchrotron_lum(particle_dists):
    """
    test sync calculation
    """
    from ..models import Synchrotron

    ECPL,PL,BPL = particle_dists

    lum_ref = [0.00025231299317576985, 0.03316716140790719, 0.00044597094281743937]

    We_ref = [5064124681.509977, 11551172186.500914, 926633862.864901]

    Wes = []
    lsys = []
    for pdist in particle_dists:
        sy = Synchrotron(pdist, **electron_properties)

        Wes.append(sy.We.to('erg').value)

        lsy = trapz_loglog(sy.spectrum(energy) * energy, energy).to('erg/s')
        assert(lsy.unit == u.erg / u.s)
        lsys.append(lsy.value)

    assert_allclose(lsys, lum_ref)
    assert_allclose(Wes, We_ref)

    sy = Synchrotron(ECPL,B=1*u.G, **electron_properties)

    lsy = trapz_loglog(sy.spectrum(energy) * energy, energy).to('erg/s')
    assert(lsy.unit == u.erg / u.s)
    assert_allclose(lsy.value, 31374135.447829477)

@pytest.mark.skipif('not HAS_SCIPY')
def test_bolometric_luminosity(particle_dists):
    """
    test sync calculation
    """
    from ..models import Synchrotron

    ECPL,PL,BPL = particle_dists

    sy = Synchrotron(ECPL,B=1*u.G, **electron_properties)
    sy.flux(energy, distance=0*u.kpc)
    sy.flux(energy, distance=0)
    sy.sed(energy, distance=0*u.kpc)
    sy.sed(energy, distance=0)

@pytest.mark.skipif('not HAS_SCIPY')
def test_compute_We(particle_dists):
    """
    test sync calculation
    """
    from ..models import Synchrotron

    ECPL,PL,BPL = particle_dists

    sy = Synchrotron(ECPL,B=1*u.G, **electron_properties)

    Eemax, Eemin = 10*u.GeV, 100*u.TeV

    sy.compute_We()
    sy.compute_We(Eemin=Eemin)
    sy.compute_We(Eemax=Eemax)
    sy.compute_We(Eemin=Eemin, Eemax=Eemax)
    assert sy.We == sy.compute_We(Eemin=sy.Eemin,Eemax=sy.Eemax)

@pytest.mark.skipif('not HAS_SCIPY')
def test_bremsstrahlung_lum(particle_dists):
    """
    test sync calculation
    """
    from ..models import Bremsstrahlung

    ECPL,PL,BPL = particle_dists

    # avoid low-energy (E<2MeV) where there are problems with cross-section
    energy2 = np.logspace(8,14,100) * u.eV

    brems = Bremsstrahlung(ECPL, n0 = 1 * u.cm**-3, Eemin = m_e*c**2)
    lbrems = trapz_loglog(brems.spectrum(energy2) * energy2, energy2).to('erg/s')

    lum_ref = 2.3064095039069847e-05
    assert_allclose(lbrems, lum_ref)

@pytest.mark.skipif('not HAS_SCIPY')
def test_inverse_compton_lum(particle_dists):
    """
    test IC calculation
    """
    from ..models import InverseCompton

    ECPL,PL,BPL = particle_dists

    lum_ref = [0.00027822017772343816, 0.004821189282097695, 0.00012916583207749083]

    We_ref = [5064124681.509977, 11551172186.500914, 926633862.864901]

    Wes = []
    lums = []
    for pdist in particle_dists:
        ic = InverseCompton(pdist, **electron_properties)
        Wes.append(ic.We.to('erg').value)
        lic = trapz_loglog(ic.spectrum(energy) * energy, energy).to('erg/s')
        assert(lic.unit == u.erg / u.s)
        lums.append(lic.value)

    assert_allclose(lums, lum_ref)
    assert_allclose(Wes, We_ref)

    ic = InverseCompton(ECPL,seed_photon_fields=['CMB','FIR','NIR'])

    lic = trapz_loglog(ic.spectrum(energy) * energy, energy).to('erg/s')
    assert_allclose(lic.value,0.00035803644460162635)

@pytest.mark.skipif('not HAS_SCIPY')
def test_anisotropic_inverse_compton_lum(particle_dists):
    """
    test IC calculation
    """
    from ..models import InverseCompton

    ECPL,PL,BPL = particle_dists

    angles = [45, 90, 135] * u.deg

    lum_ref = [48901.37566513,  111356.44973684,  149800.27022024]

    lums = []
    for angle in angles:
        ic = InverseCompton(PL,
                seed_photon_fields=[['Star',20000*u.K, 0.1*u.erg/u.cm**3, angle],],
                **electron_properties)
        lic = trapz_loglog(ic.spectrum(energy) * energy, energy).to('erg/s')
        assert(lic.unit == u.erg / u.s)
        lums.append(lic.value)

    assert_allclose(lums, lum_ref)


@pytest.mark.skipif('not HAS_SCIPY')
def test_flux_sed(particle_dists):
    """
    test IC calculation
    """
    from ..models import InverseCompton,Synchrotron,PionDecay

    ECPL,PL,BPL = particle_dists

    d1 = 2.5 * u.kpc
    d2 = 10. * u.kpc

    ic = InverseCompton(ECPL,seed_photon_fields=['CMB','FIR','NIR'], **electron_properties)

    luminosity = trapz_loglog(ic.spectrum(energy) * energy, energy).to('erg/s').value

    int_flux1 = trapz_loglog(ic.flux(energy,d1) * energy, energy).to('erg/(s cm2)').value
    int_flux2 = trapz_loglog(ic.flux(energy,d2) * energy, energy).to('erg/(s cm2)').value

    # check distance scaling
    assert_allclose(int_flux1/int_flux2,(d2/d1).value**2.)

    # check values
    assert_allclose(int_flux1,luminosity/(4*np.pi*(d1.to('cm').value)**2))

    # check SED
    sed1 = ic.sed(energy,d1).to('erg/(s cm2)').value
    sed0 = (ic.spectrum(energy) * energy ** 2).to('erg/s').value

    assert_allclose(sed1,sed0/(4*np.pi*(d1.to('cm').value)**2))

@pytest.mark.skipif('not HAS_SCIPY')
def test_ic_seed_input(particle_dists):
    """
    test initialization of different input formats for seed photon fields
    """
    from ..models import InverseCompton

    ECPL,PL,BPL = particle_dists

    ic = InverseCompton(PL, seed_photon_fields='CMB')

    ic = InverseCompton(PL, seed_photon_fields=['CMB', 'FIR', 'NIR'],)

    ic = InverseCompton(PL, seed_photon_fields=['CMB',
                        ['test', 5000 * u.K, 0], ],)

    ic = InverseCompton(PL, seed_photon_fields=['CMB',
                        ['test2', 5000 * u.K, 15 * u.eV / u.cm ** 3], ],)


@pytest.mark.skipif('not HAS_SCIPY')
def test_pion_decay(particle_dists):
    """
    test ProtonOZM
    """
    from ..models import PionDecay

    ECPL,PL,BPL = particle_dists

    for pdist in [ECPL,PL,BPL]:
        pdist.amplitude = 1*(1/u.TeV)

    lum_ref_LUT = [9.94070311e-13,   2.30256683e-12,   1.57263936e-13]

    lum_ref_noLUT = [9.94144387e-13,   2.30264140e-12,   1.57272216e-13]

    Wp_ref = [5406.36160963,  8727.55086557,   554.13864492]

    energy = np.logspace(-3, 3, 60) * u.TeV
    Wps = []
    lpps_LUT = []
    lpps_noLUT = []
    for pdist in particle_dists:
        pp = PionDecay(pdist,useLUT=True, **proton_properties)
        Wps.append(pp.Wp.to('erg').value)
        lpp = trapz_loglog(pp.spectrum(energy) * energy, energy).to('erg/s')
        assert(lpp.unit == u.erg / u.s)
        lpps_LUT.append(lpp.value)
        pp.useLUT=False
        lpp = trapz_loglog(pp.spectrum(energy) * energy, energy).to('erg/s')
        lpps_noLUT.append(lpp.value)

    assert_allclose(lpps_LUT, lum_ref_LUT)
    assert_allclose(lpps_noLUT, lum_ref_noLUT)
    assert_allclose(Wps, Wp_ref)

    # test LUT not found
    pp = PionDecay(PL,useLUT=True,hiEmodel='Geant4', **proton_properties)
    pp.spectrum(energy)

@pytest.mark.skipif('not HAS_SCIPY')
def test_pion_decay_no_nuc_enh(particle_dists):
    """
    test PionDecayKelner06
    """
    from ..radiative import PionDecay

    ECPL,PL,BPL = particle_dists

    for pdist in [ECPL,PL,BPL]:
        pdist.amplitude = 1*(1/u.TeV)

    lum_ref = [5.693100769654807e-13,]

    energy = np.logspace(9, 13, 20) * u.eV
    pp = PionDecay(ECPL,nuclear_enhancement=False, useLUT=False, **proton_properties)
    Wp = pp.Wp.to('erg').value
    lpp = trapz_loglog(pp.spectrum(energy) * energy, energy).to('erg/s')
    assert(lpp.unit == u.erg / u.s)

    assert_allclose(lpp.value, lum_ref[0])


@pytest.mark.skipif('not HAS_SCIPY')
def test_pion_decay_kelner(particle_dists):
    """
    test PionDecayKelner06
    """
    from ..radiative import PionDecayKelner06 as PionDecay

    ECPL,PL,BPL = particle_dists

    for pdist in [ECPL,PL,BPL]:
        pdist.amplitude = 1*(1/u.TeV)

    lum_ref = [5.54225481494e-13,
               1.21723084093e-12,
               7.35927471e-14]

    energy = np.logspace(9, 13, 20) * u.eV
    pp = PionDecay(ECPL, **proton_properties)
    Wp = pp.Wp.to('erg').value
    lpp = trapz_loglog(pp.spectrum(energy) * energy, energy).to('erg/s')
    assert(lpp.unit == u.erg / u.s)

    assert_allclose(lpp.value, lum_ref[0])

def test_inputs():
    """ test input validation with LogParabola and ExponentialCutoffBrokenPowerLaw
    """

    from ..models import LogParabola, ExponentialCutoffBrokenPowerLaw


    LP = LogParabola(1., e_0, 1.7, 0.2)

    LP(np.logspace(1,10,10)*u.TeV)
    LP(10*u.TeV)

    ECBPL = ExponentialCutoffBrokenPowerLaw(1., e_0, e_break, 1.5, 2.5, e_cutoff, 2.0)
    ECBPL(np.logspace(1,10,10)*u.TeV)

    with pytest.raises(TypeError):
        data = {'flux':[1,2,4]}
        LP(data)

