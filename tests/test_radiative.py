# Licensed under a 3-clause BSD style license - see LICENSE.rst
import astropy.units as u
import numpy as np
import pytest

from naima.models import ExponentialCutoffPowerLaw
from naima.radiative import (
    Bremsstrahlung,
    InverseCompton,
    PionDecay,
    PionDecayKelner06,
    Synchrotron,
)

ELECTRONS = {"Eemin": 100 * u.GeV, "Eemax": 1 * u.PeV}


# InverseCompton: seed photon fields


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


def test_seed_boolean_is_not_an_index(ic):
    with pytest.raises(ValueError):
        ic.flux(E1, seed=True)


@pytest.mark.parametrize("seed", [0, "FIR"])
def test_seed_scalar_energy(ic, seed):
    assert u.allclose(ic.flux(1 * u.TeV, seed=seed), ic.flux(E1, seed=seed)[2])
    assert u.allclose(ic.sed(1 * u.TeV, seed=seed), ic.sed(E1, seed=seed)[2])


def test_seed_photon_fields_accept_a_tuple():
    ecpl = ExponentialCutoffPowerLaw(1e36 / u.eV, 1 * u.TeV, 2.0, 50 * u.TeV)
    ic = InverseCompton(ecpl, seed_photon_fields=("CMB", "FIR"))
    assert list(ic.seed_photon_fields) == ["CMB", "FIR"]


def test_seed_photon_field_definition_accepts_a_tuple(ic):
    ecpl = ExponentialCutoffPowerLaw(1e36 / u.eV, 1 * u.TeV, 2.0, 50 * u.TeV)
    ic_tuple = InverseCompton(
        ecpl, seed_photon_fields=("CMB", tuple(FIR)), Eemin=1 * u.GeV
    )
    assert list(ic_tuple.seed_photon_fields) == ["CMB", "FIR"]
    assert u.allclose(ic_tuple.flux(E1), ic.flux(E1))


def test_seed_sed_only_computes_the_requested_field(ic, monkeypatch):
    calls = []
    calc_specic = ic._calc_specic

    def counting(seed, energy):
        calls.append(seed)
        return calc_specic(seed, energy)

    monkeypatch.setattr(ic, "_calc_specic", counting)
    ic.sed(E1, seed="FIR")
    assert calls == ["FIR"]


def test_unknown_seed_photon_field_raises_with_a_message():
    ecpl = ExponentialCutoffPowerLaw(1e36 / u.eV, 1 * u.TeV, 2.0, 50 * u.TeV)
    with pytest.raises(ValueError, match="XYZ"):
        InverseCompton(ecpl, seed_photon_fields=["CMB", "XYZ"])


# Results are not cached between calls


ENERGY = np.logspace(-3, 1, 5) * u.TeV
XRAY = np.logspace(0, 3, 4) * u.eV


def test_callable_distribution_parameters_are_not_cached():
    """A plain callable has no param_names, so it used to contribute nothing
    to the cache key and changing it returned the previous result."""
    amplitude = {"value": 1e36}

    def distribution(energy):
        return amplitude["value"] * (energy / (1 * u.TeV)) ** -2 / u.eV

    syn = Synchrotron(distribution, B=10 * u.uG)
    first = syn.sed(XRAY)
    assert np.all(first > 0)
    amplitude["value"] = 1e37
    second = syn.sed(XRAY)

    assert u.allclose(second, 10 * first, rtol=1e-8)


@pytest.mark.parametrize("cls", [Synchrotron, InverseCompton, Bremsstrahlung])
def test_distribution_change_in_place_is_seen(cls):
    ecpl = ExponentialCutoffPowerLaw(1e36 / u.eV, 1 * u.TeV, 2.0, 50 * u.TeV)
    model = cls(ecpl)
    first = model.sed(ENERGY)
    ecpl.amplitude = ecpl.amplitude * 3
    second = model.sed(ENERGY)
    assert u.allclose(second, 3 * first, rtol=1e-8)


def test_tiny_parameter_change_is_seen():
    """str(Quantity) truncates to a few digits, hiding small changes."""
    ecpl = ExponentialCutoffPowerLaw(1e36 / u.eV, 1 * u.TeV, 2.0, 50 * u.TeV)
    syn = Synchrotron(ecpl, B=10 * u.uG)
    first = syn.sed(XRAY)
    assert np.all(first > 0)
    syn.B = 10.00000001 * u.uG
    second = syn.sed(XRAY)
    assert np.all(second != first)


def test_pion_decay_kelner06_instances_do_not_share_results():
    energy = np.logspace(0, 2, 4) * u.TeV
    ecpl = ExponentialCutoffPowerLaw(1e36 / u.eV, 1 * u.TeV, 2.0, 50 * u.TeV)
    a = PionDecayKelner06(ecpl, nh=1 * u.cm**-3)
    b = PionDecayKelner06(ecpl, nh=2 * u.cm**-3)
    fa = a.flux(energy)
    fb = b.flux(energy)
    assert u.allclose(fb, 2 * fa, rtol=1e-8)


# Scalar energies


@pytest.mark.parametrize(
    "radiative_model",
    [
        lambda pdist: Synchrotron(pdist, B=10 * u.uG, **ELECTRONS),
        lambda pdist: InverseCompton(
            pdist, seed_photon_fields=["CMB", "FIR"], **ELECTRONS
        ),
        lambda pdist: Bremsstrahlung(pdist, **ELECTRONS),
        lambda pdist: PionDecay(pdist, Epmax=1 * u.PeV),
    ],
    ids=["Synchrotron", "InverseCompton", "Bremsstrahlung", "PionDecay"],
)
@pytest.mark.parametrize("distance", [1 * u.kpc, 0 * u.kpc])
def test_flux_and_sed_accept_scalar_energies(radiative_model, distance):
    pdist = ExponentialCutoffPowerLaw(1e36 / u.eV, 1 * u.TeV, 2.0, 50 * u.TeV)
    model = radiative_model(pdist)
    e = [0.5, 1, 2] * u.TeV

    scalar_flux = model.flux(1 * u.TeV, distance)
    scalar_sed = model.sed(1 * u.TeV, distance)

    assert scalar_flux.isscalar and scalar_sed.isscalar
    assert u.allclose(scalar_flux, model.flux(e, distance)[1], rtol=1e-6)
    assert u.allclose(scalar_sed, model.sed(e, distance)[1], rtol=1e-6)
