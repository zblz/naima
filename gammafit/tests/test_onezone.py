# Licensed under a 3-clause BSD style license - see LICENSE.rst
from astropy import units as u
import numpy as np
from numpy.testing import assert_allclose
from astropy.tests.helper import pytest
from astropy.extern import six

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

@pytest.fixture
def particle_dists():
    from ..models import ExponentialCutoffPowerLaw, PowerLaw, BrokenPowerLaw
    ECPL = ExponentialCutoffPowerLaw(amplitude=1, e_0=e_0, alpha=alpha, e_cutoff=e_cutoff)
    PL = PowerLaw(amplitude=1, e_0=e_0, alpha=alpha)
    BPL = BrokenPowerLaw(amplitude=1, e_break=e_break, alpha_1=alpha_1, alpha_2=alpha_2)
    return ECPL,PL,BPL

@pytest.mark.skipif('not HAS_SCIPY')
def test_synchrotron_lum(particle_dists):
    """
    test sync calculation
    """
    from ..models import Synchrotron

    ECPL,PL,BPL = particle_dists

    lums = [0.0002525815099101462,
            0.16997228271344694,
            1.1623884971024219e-05]

    for pdist, lum in six.moves.zip(particle_dists, lums):
        sy = Synchrotron(pdist)

        lsy = np.trapz(sy.flux(energy) * energy, energy).to('erg/s')
        assert(lsy.unit == u.erg / u.s)
        assert_allclose(lsy.value, lum)

    sy = Synchrotron(ECPL,B=1*u.G)

    lsy = np.trapz(sy.flux(energy) * energy, energy).to('erg/s')
    assert(lsy.unit == u.erg / u.s)
    assert_allclose(lsy.value, 31700300.30988492)

@pytest.mark.skipif('not HAS_SCIPY')
def test_inverse_compton_lum(particle_dists):
    """
    test sync calculation
    """
    from ..models import InverseCompton

    ECPL,PL,BPL = particle_dists

    lums = [0.00028327087904549787,
            0.005459045188008858,
            1.4938685711445786e-06]

    for pdist, lum in six.moves.zip(particle_dists, lums):
        ic = InverseCompton(pdist)

        lic = np.trapz(ic.flux(energy) * energy, energy).to('erg/s')
        assert(lic.unit == u.erg / u.s)
        assert_allclose(lic.value, lum)

    ic = InverseCompton(ECPL,seedspec=['CMB','FIR','NIR'])

    lic = np.trapz(ic.flux(energy) * energy, energy).to('erg/s')
    assert_allclose(lic.value, 0.00035996458437447014)


@pytest.mark.skipif('not HAS_SCIPY')
def test_ic_seed_input(particle_dists):
    """
    test initialization of different input formats for seed photon fields
    """
    from ..models import InverseCompton

    ECPL,PL,BPL = particle_dists

    ic = InverseCompton(PL, seedspec='CMB')

    ic = InverseCompton(PL, seedspec=['CMB', 'FIR', 'NIR'],)

    ic = InverseCompton(PL, seedspec=['CMB', ['test', 5000 * u.K, 0], ],)

    ic = InverseCompton(PL,
            seedspec=['CMB', ['test2', 5000 * u.K, 15 * u.eV / u.cm ** 3], ],)


@pytest.mark.skipif('not HAS_SCIPY')
def test_pion_decay(particle_dists):
    """
    test ProtonOZM
    """
    from ..models import PionDecay

    ECPL,PL,BPL = particle_dists

    lums = [5.81597553001e-13,
            1.25944455136e-12,
            8.56815829515e-16]

    energy = np.logspace(9, 13, 20) * u.eV

    for pdist, lum in six.moves.zip(particle_dists, lums):
        pp = PionDecay(pdist)

        lpp = np.trapz(pp.flux(energy) * energy, energy).to('erg/s')
        assert(lpp.unit == u.erg / u.s)
        assert_allclose(lpp.value, lum)

