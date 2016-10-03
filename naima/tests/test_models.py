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

electron_properties = {'Eemin': 100 * u.GeV, 'Eemax': 1 * u.PeV}
proton_properties = {'Epmax': 1 * u.PeV}

energy = np.logspace(0, 15, 1000) * u.eV

from astropy.table import QTable, Table
data = QTable()
data['energy'] = energy
data2 = Table()
data2['energy'] = energy

from astropy.constants import m_e, c, sigma_sb, hbar
pdist_unit = 1 / u.Unit(m_e * c**2)


@pytest.fixture
def particle_dists():
    from ..models import ExponentialCutoffPowerLaw, PowerLaw, BrokenPowerLaw
    ECPL = ExponentialCutoffPowerLaw(amplitude=1 * pdist_unit,
                                     e_0=e_0,
                                     alpha=alpha,
                                     e_cutoff=e_cutoff)
    PL = PowerLaw(amplitude=1 * pdist_unit, e_0=e_0, alpha=alpha)
    BPL = BrokenPowerLaw(amplitude=1 * pdist_unit,
                         e_0=e_0,
                         e_break=e_break,
                         alpha_1=alpha_1,
                         alpha_2=alpha_2)
    return ECPL, PL, BPL


@pytest.mark.skipif('not HAS_SCIPY')
def test_synchrotron_lum(particle_dists):
    """
    test sync calculation
    """
    from ..models import Synchrotron

    ECPL, PL, BPL = particle_dists

    lum_ref = [0.00025231299317576985, 0.03316716140790719,
               0.00044597094281743937]

    We_ref = [5064124681.509977, 11551172186.500914, 926633862.864901]

    Wes = []
    lsys = []
    for pdist in particle_dists:
        sy = Synchrotron(pdist, **electron_properties)

        Wes.append(sy.We.to('erg').value)

        lsy = trapz_loglog(sy.flux(energy, 0) * energy, energy).to('erg/s')
        assert (lsy.unit == u.erg / u.s)
        lsys.append(lsy.value)

    assert_allclose(lsys, lum_ref)
    assert_allclose(Wes, We_ref)

    sy = Synchrotron(ECPL, B=1 * u.G, **electron_properties)
    sy.flux(data)
    sy.flux(data2)

    lsy = trapz_loglog(sy.flux(energy, 0) * energy, energy).to('erg/s')
    assert (lsy.unit == u.erg / u.s)
    assert_allclose(lsy.value, 31374135.447829477)


@pytest.mark.skipif('not HAS_SCIPY')
def test_bolometric_luminosity(particle_dists):
    """
    test sync calculation
    """
    from ..models import Synchrotron

    ECPL, PL, BPL = particle_dists

    sy = Synchrotron(ECPL, B=1 * u.G, **electron_properties)
    sy.flux(energy, distance=0 * u.kpc)
    sy.flux(energy, distance=0)
    sy.sed(energy, distance=0 * u.kpc)
    sy.sed(energy, distance=0)


@pytest.mark.skipif('not HAS_SCIPY')
def test_compute_We(particle_dists):
    """
    test sync calculation
    """
    from ..models import Synchrotron, PionDecay

    ECPL, PL, BPL = particle_dists

    sy = Synchrotron(ECPL, B=1 * u.G, **electron_properties)

    Eemin, Eemax = 10 * u.GeV, 100 * u.TeV

    sy.compute_We()
    sy.compute_We(Eemin=Eemin)
    sy.compute_We(Eemax=Eemax)
    sy.compute_We(Eemin=Eemin, Eemax=Eemax)
    assert sy.We == sy.compute_We(Eemin=sy.Eemin, Eemax=sy.Eemax)

    pp = PionDecay(ECPL)
    Epmin, Epmax = 10 * u.GeV, 100 * u.TeV
    pp.compute_Wp()
    pp.compute_Wp(Epmin=Epmin)
    pp.compute_Wp(Epmax=Epmax)
    pp.compute_Wp(Epmin=Epmin, Epmax=Epmax)


@pytest.mark.skipif('not HAS_SCIPY')
def test_set_We(particle_dists):
    """
    test sync calculation
    """
    from ..models import Synchrotron, PionDecay

    ECPL, PL, BPL = particle_dists

    sy = Synchrotron(ECPL, B=1 * u.G, **electron_properties)
    pp = PionDecay(ECPL)

    W = 1e49 * u.erg

    Eemax = 100 * u.TeV
    for Eemin in [1 * u.GeV, 10 * u.GeV, None]:
        for Eemax in [100 * u.TeV, None]:
            sy.set_We(W, Eemin, Eemax)
            assert_allclose(W, sy.compute_We(Eemin, Eemax))
            sy.set_We(W, Eemin, Eemax, amplitude_name='amplitude')
            assert_allclose(W, sy.compute_We(Eemin, Eemax))

            pp.set_Wp(W, Eemin, Eemax)
            assert_allclose(W, pp.compute_Wp(Eemin, Eemax))
            pp.set_Wp(W, Eemin, Eemax, amplitude_name='amplitude')
            assert_allclose(W, pp.compute_Wp(Eemin, Eemax))

    with pytest.raises(AttributeError):
        sy.set_We(W, amplitude_name='norm')

    with pytest.raises(AttributeError):
        pp.set_Wp(W, amplitude_name='norm')


@pytest.mark.skipif('not HAS_SCIPY')
def test_bremsstrahlung_lum(particle_dists):
    """
    test sync calculation
    """
    from ..models import Bremsstrahlung

    ECPL, PL, BPL = particle_dists

    # avoid low-energy (E<2MeV) where there are problems with cross-section
    energy2 = np.logspace(8, 14, 100) * u.eV

    brems = Bremsstrahlung(ECPL, n0=1 * u.cm** -3, Eemin=m_e * c**2)
    lbrems = trapz_loglog(brems.flux(energy2, 0) * energy2, energy2).to('erg/s')

    lum_ref = 2.3064095039069847e-05
    assert_allclose(lbrems.value, lum_ref)


@pytest.mark.skipif('not HAS_SCIPY')
def test_inverse_compton_lum(particle_dists):
    """
    test IC calculation
    """
    from ..models import InverseCompton

    ECPL, PL, BPL = particle_dists

    lum_ref = [0.00027822017772343816, 0.004821189282097695,
               0.00012916583207749083]

    lums = []
    for pdist in particle_dists:
        ic = InverseCompton(pdist, **electron_properties)
        lic = trapz_loglog(ic.flux(energy, 0) * energy, energy).to('erg/s')
        assert (lic.unit == u.erg / u.s)
        lums.append(lic.value)

    assert_allclose(lums, lum_ref)

    ic = InverseCompton(ECPL, seed_photon_fields=['CMB', 'FIR', 'NIR'])
    ic.flux(data)
    ic.flux(data2)

    lic = trapz_loglog(ic.flux(energy, 0) * energy, energy).to('erg/s')
    assert_allclose(lic.value, 0.0005833030865998417)


@pytest.mark.skipif('not HAS_SCIPY')
def test_anisotropic_inverse_compton_lum(particle_dists):
    """
    test IC calculation
    """
    from ..models import InverseCompton

    ECPL, PL, BPL = particle_dists

    angles = [45, 90, 135] * u.deg

    lum_ref = [48901.37566513, 111356.44973684, 149800.27022024]

    lums = []
    for angle in angles:
        ic = InverseCompton(PL,
                            seed_photon_fields=[['Star', 20000 * u.K, 0.1 *
                                                 u.erg / u.cm**3, angle],],
                            **electron_properties)
        lic = trapz_loglog(ic.flux(energy, 0) * energy, energy).to('erg/s')
        assert (lic.unit == u.erg / u.s)
        lums.append(lic.value)

    assert_allclose(lums, lum_ref)


@pytest.mark.skipif('not HAS_SCIPY')
def test_monochromatic_inverse_compton(particle_dists):
    """
    test IC monochromatic against khangulyan et al.
    """
    from ..models import InverseCompton, PowerLaw

    PL = PowerLaw(1 / u.eV, 1 * u.TeV, 3)

    # compute a blackbody spectrum with 1 eV/cm3 at 30K
    from astropy.analytic_functions import blackbody_nu
    Ephbb = np.logspace(-3.5, -1.5, 100) * u.eV
    lambdabb = Ephbb.to('AA', equivalencies=u.equivalencies.spectral())
    T = 30 * u.K
    w = 1 * u.eV / u.cm**3
    bb = (blackbody_nu(lambdabb, T) * 2 * u.sr / c.cgs
            / Ephbb / hbar).to('1/(cm3 eV)')
    Ebbmax = Ephbb[np.argmax(Ephbb**2 * bb)]

    ar = (4 * sigma_sb / c).to('erg/(cm3 K4)')
    bb *= (w / (ar * T**4)).decompose()

    eopts = {'Eemax': 10000 * u.GeV, 'Eemin': 10 * u.GeV, 'nEed': 1000}
    IC_khang = InverseCompton(PL, seed_photon_fields=[['bb', T, w]], **eopts)
    IC_mono = InverseCompton(PL,
                             seed_photon_fields=[['mono', Ebbmax, w]],
                             **eopts)
    IC_bb = InverseCompton(PL, seed_photon_fields=[['bb2', Ephbb, bb]], **eopts)
    IC_bb_ene = InverseCompton(PL,
                    seed_photon_fields=[['bb2', Ephbb, Ephbb**2 * bb]], **eopts)

    Eph = np.logspace(-1, 1, 3) * u.GeV

    assert_allclose(IC_khang.sed(Eph).value, IC_mono.sed(Eph).value, rtol=1e-2)
    assert_allclose(IC_khang.sed(Eph).value, IC_bb.sed(Eph).value, rtol=1e-2)
    assert_allclose(IC_khang.sed(Eph).value, IC_bb_ene.sed(Eph).value,
                    rtol=1e-2)


@pytest.mark.skipif('not HAS_SCIPY')
def test_flux_sed(particle_dists):
    """
    test IC calculation
    """
    from ..models import InverseCompton, Synchrotron, PionDecay

    ECPL, PL, BPL = particle_dists

    d1 = 2.5 * u.kpc
    d2 = 10. * u.kpc

    ic = InverseCompton(ECPL,
                        seed_photon_fields=['CMB', 'FIR', 'NIR'],
                        **electron_properties)

    luminosity = trapz_loglog(
        ic.flux(energy, 0) * energy, energy).to('erg/s').value

    int_flux1 = trapz_loglog(
        ic.flux(energy, d1) * energy, energy).to('erg/(s cm2)').value
    int_flux2 = trapz_loglog(
        ic.flux(energy, d2) * energy, energy).to('erg/(s cm2)').value

    # check distance scaling
    assert_allclose(int_flux1 / int_flux2, (d2 / d1).value**2.)

    # check values
    assert_allclose(int_flux1, luminosity / (4 * np.pi *
                                             (d1.to('cm').value)**2))

    # check SED
    sed1 = ic.sed(energy, d1).to('erg/(s cm2)').value
    sed0 = (ic.flux(energy, 0) * energy**2).to('erg/s').value

    assert_allclose(sed1, sed0 / (4 * np.pi * (d1.to('cm').value)**2))


@pytest.mark.skipif('not HAS_SCIPY')
def test_ic_seed_input(particle_dists):
    """
    test initialization of different input formats for seed photon fields
    """
    from ..models import InverseCompton

    ECPL, PL, BPL = particle_dists

    ic = InverseCompton(PL, seed_photon_fields='CMB')

    ic = InverseCompton(PL, seed_photon_fields=['CMB', 'FIR', 'NIR'],)

    Eph = (1, 10) * u.eV
    phn = (3, 1) * u.Unit('1/(eV cm3)')
    test_seeds = [['test', 5000 * u.K, 0],
                  ['array', Eph, phn],
                  ['array-energy', Eph, Eph**2 * phn],
                  ['mono', Eph[0], phn[0] * Eph[0]**2],
                  ['mono-array', Eph[:1], phn[:1] * Eph[:1]**2],
                  # from docs:
                  ['NIR', 50 * u.K, 1.5 * u.eV / u.cm**3],
                  ['star', 25000 * u.K, 3 * u.erg / u.cm**3, 120 * u.deg],
                  ['X-ray', [1, 10] * u.keV, [1, 1e-2] * 1 / (u.eV * u.cm**3)],
                  ['UV', 50 * u.eV, 15 * u.eV / u.cm**3],]

    for seed in test_seeds:
        ic = InverseCompton(PL, seed_photon_fields=['CMB', seed])


@pytest.mark.skipif('not HAS_SCIPY')
def test_ic_seed_fluxes(particle_dists):
    """
    test per seed flux computation
    """
    from ..models import InverseCompton

    _, PL, _ = particle_dists

    ic = InverseCompton(
        PL,
        seed_photon_fields=['CMB',
                            ['test', 5000 * u.K, 0],
                            ['test2', 5000 * u.K, 10 * u.eV / u.cm**3],
                            ['test3', 5000 * u.K, 10 * u.eV / u.cm**3, 90 *
                             u.deg],],)

    ene = np.logspace(-3, 0, 5) * u.TeV

    for idx, name in enumerate(['CMB', 'test', 'test2', 'test3',]):
        icname = ic.sed(ene, seed=name)
        icnumber = ic.sed(ene, seed=idx)
        assert_allclose(icname, icnumber)

    with pytest.raises(ValueError):
        _ = ic.sed(ene, seed='FIR')

    with pytest.raises(ValueError):
        _ = ic.sed(ene, seed=10)


@pytest.mark.skipif('not HAS_SCIPY')
def test_pion_decay(particle_dists):
    """
    test ProtonOZM
    """
    from ..models import PionDecay

    ECPL, PL, BPL = particle_dists

    for pdist in [ECPL, PL, BPL]:
        pdist.amplitude = 1 * (1 / u.TeV)

    lum_ref_LUT = [9.94070311e-13, 2.30256683e-12, 1.57263936e-13]

    lum_ref_noLUT = [9.94144387e-13, 2.30264140e-12, 1.57272216e-13]

    Wp_ref = [5406.36160963, 8727.55086557, 554.13864492]

    energy = np.logspace(-3, 3, 60) * u.TeV
    Wps = []
    lpps_LUT = []
    lpps_noLUT = []
    for pdist in particle_dists:
        pp = PionDecay(pdist, useLUT=True, **proton_properties)
        Wps.append(pp.Wp.to('erg').value)
        lpp = trapz_loglog(pp.flux(energy, 0) * energy, energy).to('erg/s')
        assert (lpp.unit == u.erg / u.s)
        lpps_LUT.append(lpp.value)
        pp.useLUT = False
        lpp = trapz_loglog(pp.flux(energy, 0) * energy, energy).to('erg/s')
        lpps_noLUT.append(lpp.value)

    assert_allclose(lpps_LUT, lum_ref_LUT)
    assert_allclose(lpps_noLUT, lum_ref_noLUT)
    assert_allclose(Wps, Wp_ref)

    # test LUT not found
    pp = PionDecay(PL, useLUT=True, hiEmodel='Geant4', **proton_properties)
    pp.flux(energy, 0)


@pytest.mark.skipif('not HAS_SCIPY')
def test_pion_decay_no_nuc_enh(particle_dists):
    """
    test PionDecayKelner06
    """
    from ..radiative import PionDecay

    ECPL, PL, BPL = particle_dists

    for pdist in [ECPL, PL, BPL]:
        pdist.amplitude = 1 * (1 / u.TeV)

    lum_ref = [5.693100769654807e-13,]

    energy = np.logspace(9, 13, 20) * u.eV
    pp = PionDecay(ECPL,
                   nuclear_enhancement=False,
                   useLUT=False,
                   **proton_properties)
    Wp = pp.Wp.to('erg').value
    lpp = trapz_loglog(pp.flux(energy, 0) * energy, energy).to('erg/s')
    assert (lpp.unit == u.erg / u.s)

    assert_allclose(lpp.value, lum_ref[0])


@pytest.mark.skipif('not HAS_SCIPY')
def test_pion_decay_kelner(particle_dists):
    """
    test PionDecayKelner06
    """
    from ..radiative import PionDecayKelner06 as PionDecay

    ECPL, PL, BPL = particle_dists

    for pdist in [ECPL, PL, BPL]:
        pdist.amplitude = 1 * (1 / u.TeV)

    lum_ref = [5.54225481494e-13, 1.21723084093e-12, 7.35927471e-14]

    energy = np.logspace(9, 13, 20) * u.eV
    pp = PionDecay(ECPL, **proton_properties)
    Wp = pp.Wp.to('erg').value
    lpp = trapz_loglog(pp.flux(energy, 0) * energy, energy).to('erg/s')
    assert (lpp.unit == u.erg / u.s)

    assert_allclose(lpp.value, lum_ref[0])


def test_inputs():
    """ test input validation with LogParabola and ExponentialCutoffBrokenPowerLaw
    """

    from ..models import LogParabola, ExponentialCutoffBrokenPowerLaw

    LP = LogParabola(1., e_0, 1.7, 0.2)
    LP._memoize = True

    # do twice for memoize
    LP(np.logspace(1, 10, 10) * u.TeV)
    LP(np.logspace(1, 10, 10) * u.TeV)
    LP(10 * u.TeV)
    LP(10 * u.TeV)

    ECBPL = ExponentialCutoffBrokenPowerLaw(1., e_0, e_break, 1.5, 2.5,
                                            e_cutoff, 2.0)
    ECBPL._memoize = True
    ECBPL(np.logspace(1, 10, 10) * u.TeV)

    with pytest.raises(TypeError):
        data = {'flux': [1, 2, 4]}
        LP(data)


def test_tablemodel():
    from ..models import TableModel

    lemin, lemax = -4, 2
    # test an exponential cutoff PL with index 2, cutoff at 10 TeV
    e = np.logspace(lemin, lemax, 50) * u.TeV
    n = (e.value)** -2 * np.exp(-e.value / 10) / u.eV
    tm = TableModel(e, n, amplitude=1)
    assert_allclose(n.to('1/eV').value, tm(e).to('1/eV').value)

    # test interpolation at low tolerance
    e2 = np.logspace(lemin, lemax, 1000) * u.TeV
    n2 = (e2.value)** -2 * np.exp(-e2.value / 10) / u.eV
    assert_allclose(n2.to('1/eV').value, tm(e2).to('1/eV').value, rtol=1e-1)

    # test TableModel without units in y
    tm2 = TableModel(e, n.value)
    assert_allclose(tm2(e2), n2.value, rtol=1e-1)

    # test that it returns 0 outside of bounds
    e3 = np.logspace(lemin - 4, lemin - 2, 100) * u.TeV
    assert_allclose(tm(e3).value, 0.0)

    # use tablemodel as pdist
    from ..radiative import Synchrotron, InverseCompton, PionDecay
    SY = Synchrotron(tm)
    _ = SY.flux(e / 10)
    IC = InverseCompton(tm)
    _ = IC.flux(e / 10)
    PP = PionDecay(tm)
    _ = PP.flux(e / 10)


def test_eblabsorptionmodel():
    """
    test EblAbsorptionModel
    """
    from ..models import EblAbsorptionModel, BrokenPowerLaw

    lemin, lemax = -4, 2

    EBL_zero = EblAbsorptionModel(0., 'Dominguez')
    EBL_moderate = EblAbsorptionModel(0.5, 'Dominguez')

    e = np.logspace(lemin, lemax, 50) * u.TeV

#   Test if the EBL absorption at z = 0 changes the test array filled with ones
    assert_allclose(np.ones_like(e).value, np.ones_like(e).value * 
                    EBL_zero.transmission(e), rtol=1e-1)
#   Make sure the transmission at z = 0. is always larger than the one at z = 0.5
    difference = EBL_zero.transmission(e) - EBL_moderate.transmission(e)
    assert(np.all(difference > -1E-10))
