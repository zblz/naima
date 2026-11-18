# Licensed under a 3-clause BSD style license - see LICENSE.rst
import gc
from pathlib import Path

import astropy.units as u
import numpy as np
import pytest
from astropy.io import ascii

from naima.core import _initial_walkers, get_sampler
from naima.models import ExponentialCutoffPowerLaw
from naima.radiative import (
    Bremsstrahlung,
    InverseCompton,
    PionDecay,
    Synchrotron,
)

data_table = ascii.read(str(Path(__file__).parent / "data/CrabNebula_HESS_ipac.dat"))
UNIT = u.Unit("1/(cm2 s TeV)")

ECPL = ExponentialCutoffPowerLaw(1e36 / u.eV, 1 * u.TeV, 2.5, 50 * u.TeV)


def powerlaw(pars, data):
    x = data["energy"]
    x0 = np.sqrt(x[0] * x[-1])
    return pars[0] * (x / x0) ** -pars[1] * UNIT


def raises_above_index(pars, data):
    if pars[1] > 2.1:
        raise OverflowError("model blew up")
    return powerlaw(pars, data)


# C6: the model failures are also handled by the prefit


def test_prefit_survives_a_model_that_fails_in_part_of_the_space():
    sampler, pos = get_sampler(
        data_table=data_table,
        p0=np.array([1e-11, 2.0]),
        labels=["norm", "index"],
        model=raises_above_index,
        nwalkers=10,
        nburn=0,
        threads=1,
        prefit=True,
        guess=False,
    )
    assert np.all(np.isfinite(pos.coords))


# C7: get_sampler alone does not leak its pool


def test_pool_is_terminated_when_the_sampler_is_discarded():
    sampler, pos = get_sampler(
        data_table=data_table,
        p0=np.array([1e-11, 2.0]),
        labels=["norm", "index"],
        model=powerlaw,
        nwalkers=10,
        nburn=0,
        threads=2,
        prefit=False,
        guess=False,
    )
    pool = sampler._naima_pool
    finalizer = sampler._naima_finalizer
    assert pool is not None and finalizer.alive
    del sampler
    gc.collect()
    assert not finalizer.alive
    assert pool._state != 0  # not RUN: terminated


# C12: scalar energies


@pytest.mark.parametrize(
    "model",
    [
        Synchrotron(ECPL, B=10 * u.uG),
        InverseCompton(ECPL, seed_photon_fields=["CMB", "FIR"]),
        Bremsstrahlung(ECPL, n0=1 / u.cm**3),
        PionDecay(ECPL, nh=1 / u.cm**3),
    ],
    ids=["Synchrotron", "InverseCompton", "Bremsstrahlung", "PionDecay"],
)
def test_flux_and_sed_accept_scalar_energies(model):
    e = [0.5, 1, 2] * u.TeV
    flux = model.flux(e)
    sed = model.sed(e)
    scalar_flux = model.flux(1 * u.TeV)
    scalar_sed = model.sed(1 * u.TeV)
    assert scalar_flux.isscalar and scalar_sed.isscalar
    assert u.allclose(scalar_flux, flux[1], rtol=1e-6)
    assert u.allclose(scalar_sed, sed[1], rtol=1e-6)


def test_inverse_compton_seed_accepts_scalar_energies():
    ic = InverseCompton(ECPL, seed_photon_fields=["CMB", "FIR"])
    e = [0.5, 1, 2] * u.TeV
    for seed in (0, "FIR"):
        assert u.allclose(
            ic.flux(1 * u.TeV, seed=seed), ic.flux(e, seed=seed)[1], rtol=1e-6
        )
        assert u.allclose(
            ic.sed(1 * u.TeV, seed=seed), ic.sed(e, seed=seed)[1], rtol=1e-6
        )


def test_flux_with_distance_zero_accepts_scalar_energies():
    ic = InverseCompton(ECPL)
    assert ic.flux(1 * u.TeV, distance=0 * u.kpc).unit == u.Unit("1/(s eV)")


# C15: seed photon fields input


def test_seed_photon_fields_accept_a_tuple():
    ic = InverseCompton(ECPL, seed_photon_fields=("CMB", "FIR"))
    assert list(ic.seed_photon_fields) == ["CMB", "FIR"]


def test_unknown_seed_photon_field_raises_a_message():
    with pytest.raises(ValueError, match="XYZ"):
        InverseCompton(ECPL, seed_photon_fields=["CMB", "XYZ"])


def test_boolean_seed_is_not_an_index():
    ic = InverseCompton(ECPL, seed_photon_fields=["CMB", "FIR"])
    with pytest.raises(ValueError):
        ic.flux([1] * u.TeV, seed=True)


# Walkers: only labels of logarithmic parameters have an absolute spread


@pytest.mark.parametrize(
    "label,scale",
    [
        ("log10(cutoff)", 1.0),
        ("log(norm)", np.log(10)),
        ("log10_cutoff", 1.0),
        ("logistic_k", None),
        ("index", None),
    ],
)
def test_only_logarithmic_labels_get_an_absolute_spread(label, scale):
    p0 = np.array([5.0])
    walkers = _initial_walkers(p0, 20000, 0.1, [label], log_spread=1.0)
    expected = 0.1 * 5.0 if scale is None else scale
    assert np.isclose(walkers[:, 0].std(), expected, rtol=0.05)


# Failures of the model are reported at the first failure of every fit


def test_model_failure_warning_is_repeated_in_a_new_fit(caplog):
    import naima.core

    naima.core._model_failure_warned = True
    get_sampler(
        data_table=data_table,
        p0=np.array([1e-11, 2.0]),
        labels=["norm", "index"],
        model=powerlaw,
        nwalkers=10,
        nburn=0,
        threads=1,
        prefit=False,
        guess=False,
    )
    assert naima.core._model_failure_warned is False
