# Licensed under a 3-clause BSD style license - see LICENSE.rst
import astropy.units as u
import numpy as np
import pytest

from naima.models import ExponentialCutoffPowerLaw
from naima.radiative import (
    Bremsstrahlung,
    InverseCompton,
    PionDecayKelner06,
    Synchrotron,
)

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
