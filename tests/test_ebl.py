# Licensed under a 3-clause BSD style license - see LICENSE.rst
import astropy.units as u
import numpy as np
import pytest

from naima import models
from naima.models import EblAbsorptionModel

pytest.importorskip("scipy")

# Optical depth of the Dominguez 2011 table at z=0.1, from the data file
TAU_Z01_10TEV = 3.79552


def test_transmission_is_exp_minus_tau():
    ebl = EblAbsorptionModel(0.1)
    # 10.1863 TeV is a node of the table, where tau = 3.79552
    t = ebl.transmission(10.1863 * u.TeV)
    assert t == pytest.approx(np.exp(-TAU_Z01_10TEV), rel=1e-3)


def test_call_is_an_absorption_factor_not_an_amplification():
    ebl = EblAbsorptionModel(0.1)
    e = np.logspace(-4, 3, 30) * u.TeV
    out = ebl(e)
    assert out.unit == u.dimensionless_unscaled
    assert np.all((out.value >= 0) & (out.value <= 1))
    assert np.allclose(out.value, ebl.transmission(e))


def test_absorption_factor_vanishes_above_table_range():
    """Above 100 TeV the optical depth is >200, so the flux is fully absorbed."""
    ebl = EblAbsorptionModel(0.1)
    t = ebl.transmission([150, 200, 1000] * u.TeV)
    assert np.all(t < 1e-30)


def test_transmission_is_monotonic_with_energy():
    ebl = EblAbsorptionModel(0.5)
    e = np.logspace(-0.5, 2, 100) * u.TeV
    assert np.all(np.diff(ebl.transmission(e)) <= 1e-12)


def test_no_absorption_for_zero_redshift_at_all_energies():
    ebl = EblAbsorptionModel(0.0)
    t = ebl.transmission([1e-5, 1, 1000] * u.TeV)
    assert np.all(t == 1.0)


def test_unity_below_one_gev():
    ebl = EblAbsorptionModel(1.0)
    assert np.all(ebl.transmission([1, 10, 500] * u.MeV) == 1.0)


def test_scalar_energy():
    ebl = EblAbsorptionModel(0.1)
    assert ebl.transmission(1 * u.TeV) == pytest.approx(
        ebl.transmission([1] * u.TeV)[0]
    )


def test_table_is_loaded_once(monkeypatch):
    """The npz must not be reloaded from disk for every instance."""
    models._load_tau_dominguez11.cache_clear()
    calls = []
    real = np.load

    def counting_load(*args, **kwargs):
        calls.append(args)
        return real(*args, **kwargs)

    monkeypatch.setattr(models.np, "load", counting_load)
    for _ in range(5):
        EblAbsorptionModel(0.3)
    assert len(calls) == 1
