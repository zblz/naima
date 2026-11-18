# Licensed under a 3-clause BSD style license - see LICENSE.rst
import astropy.units as u
import numpy as np
import pytest

from naima.models import ExponentialCutoffPowerLaw
from naima.radiative import InverseCompton

FIR = ["FIR", 30 * u.K, 0.5 * u.eV / u.cm**3]


@pytest.fixture
def ic():
    ecpl = ExponentialCutoffPowerLaw(1e36 / u.eV, 1 * u.TeV, 2.0, 50 * u.TeV)
    return InverseCompton(ecpl, seed_photon_fields=["CMB", FIR], Eemin=1 * u.GeV)


E1 = np.logspace(-1, 1, 5) * u.TeV
E2 = np.logspace(-2, 2, 7) * u.TeV


def test_seed_flux_after_other_energies(ic):
    """flux(E, seed=...) must not depend on the last energies that were used."""
    ref = ic.flux(E1, seed="CMB")

    ic.flux(E1)
    ic.flux(E2)
    after = ic.flux(E1, seed="CMB")

    assert after.shape == E1.shape
    assert u.allclose(after, ref, rtol=1e-12)


def test_seed_fluxes_add_up_to_total(ic):
    total = ic.flux(E1)
    ic.flux(E2)
    parts = ic.flux(E1, seed=0) + ic.flux(E1, seed=1)
    assert u.allclose(parts, total, rtol=1e-10)


def test_seed_by_name_and_index_agree(ic):
    assert u.allclose(ic.flux(E1, seed="FIR"), ic.flux(E1, seed=1), rtol=1e-12)
    assert u.allclose(ic.sed(E1, seed="FIR"), ic.sed(E1, seed=1), rtol=1e-12)


def test_seed_numpy_integer(ic):
    assert u.allclose(ic.flux(E1, seed=np.int64(1)), ic.flux(E1, seed=1), rtol=1e-12)


def test_seed_zero_distance(ic):
    flux = ic.flux(E1, distance=0, seed=1)
    assert flux.unit.is_equivalent(1 / (u.s * u.eV))


@pytest.mark.parametrize("seed", [2, 5, -1])
def test_seed_index_out_of_range(ic, seed):
    with pytest.raises(ValueError):
        ic.flux(E1, seed=seed)


def test_seed_unknown_name(ic):
    with pytest.raises(ValueError):
        ic.flux(E1, seed="NIR")
