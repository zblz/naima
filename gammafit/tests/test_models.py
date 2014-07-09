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

energy = np.logspace(0, 15, 1000) * u.eV

from astropy.constants import m_e, c
pdist_unit = 1/u.Unit(m_e * c**2)

@pytest.fixture
def particle_dists():
    from ..models import ExponentialCutoffPowerLaw, PowerLaw, BrokenPowerLaw
    ECPL = ExponentialCutoffPowerLaw(amplitude=1*pdist_unit, e_0=e_0,
            alpha=alpha, e_cutoff=e_cutoff)
    PL = PowerLaw(amplitude=1*pdist_unit, e_0=e_0, alpha=alpha)
    BPL = BrokenPowerLaw(amplitude=1*pdist_unit, e_break=e_break,
            alpha_1=alpha_1, alpha_2=alpha_2)
    return ECPL,PL,BPL

@pytest.mark.skipif('not HAS_SCIPY')
def test_synchrotron_lum(particle_dists):
    """
    test sync calculation
    """
    from ..models import Synchrotron

    ECPL,PL,BPL = particle_dists

    lums = [0.000251958779927,
            0.169718700532,
            1.16045322298e-05
           ]

    We_ref = [8781804580.93,
            17326758279.8,
            12005110.5343]

    Wes = []
    for pdist, lum in six.moves.zip(particle_dists, lums):
        sy = Synchrotron(pdist)

        Wes.append(sy.We.to('erg').value)

        lsy = trapz_loglog(sy.flux(energy) * energy, energy).to('erg/s')
        assert(lsy.unit == u.erg / u.s)
        assert_allclose(lsy.value, lum)

    assert_allclose(Wes, We_ref)

    sy = Synchrotron(ECPL,B=1*u.G)

    lsy = trapz_loglog(sy.flux(energy) * energy, energy).to('erg/s')
    assert(lsy.unit == u.erg / u.s)
    assert_allclose(lsy.value, 31629469.710301004)

@pytest.mark.skipif('not HAS_SCIPY')
def test_inverse_compton_lum(particle_dists):
    """
    test sync calculation
    """
    from ..models import InverseCompton

    ECPL,PL,BPL = particle_dists

    lums = [0.000283116903484,
            0.00545489789968,
            1.49293663874e-06,
            ]

    We_ref = [8.78208826e+09,
            1.87706548e+10,
            1.20438823e+07]

    Wes = []
    for pdist, lum in six.moves.zip(particle_dists, lums):
        ic = InverseCompton(pdist)

        Wes.append(ic.We.to('erg').value)
        lic = trapz_loglog(ic.flux(energy) * energy, energy).to('erg/s')
        assert(lic.unit == u.erg / u.s)
        assert_allclose(lic.value, lum)

    assert_allclose(Wes, We_ref)

    ic = InverseCompton(ECPL,seed_photon_fields=['CMB','FIR','NIR'])

    lic = trapz_loglog(ic.flux(energy) * energy, energy).to('erg/s')
    assert_allclose(lic.value, 0.000359750950957)


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

    lums = [5.54225481494e-13,
            1.21723084093e-12,
            8.22791925348e-16]

    Wp_ref = [5.40535654e+03,
            2.74631565e+04,
            6.29678421e+00]

    energy = np.logspace(9, 13, 20) * u.eV
    Wps = []
    for pdist, lum in six.moves.zip(particle_dists, lums):
        pp = PionDecay(pdist)

        Wps.append(pp.Wp.to('erg').value)

        lpp = trapz_loglog(pp.flux(energy) * energy, energy).to('erg/s')
        assert(lpp.unit == u.erg / u.s)
        assert_allclose(lpp.value, lum)

    assert_allclose(Wps, Wp_ref)

def test_inputs():
    """ test input validation with LogParabola
    """

    from ..models import LogParabola


    LP = LogParabola(1., e_0, 1.7, 0.2)

    LP(np.logspace(1,10,10)*u.TeV)
    LP(10*u.TeV)

    with pytest.raises(TypeError):
        data = {'flux':[1,2,4]}
        LP(data)

