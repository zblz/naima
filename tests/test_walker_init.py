# Licensed under a 3-clause BSD style license - see LICENSE.rst
from pathlib import Path

import astropy.units as u
import numpy as np
import pytest
from astropy import log
from astropy.io import ascii

from naima.core import _initial_walkers, get_sampler, uniform_prior

data_table = ascii.read(str(Path(__file__).parent / "data/CrabNebula_HESS_ipac.dat"))


def powerlaw_extra(pars, data):
    """Power law with an unused third parameter."""
    x = data["energy"]
    x0 = np.sqrt(x[0] * x[-1])
    return pars[0] * (x / x0) ** -pars[1] * u.Unit("1/(cm2 s TeV)")


def test_walkers_spread_around_zero_parameter():
    """A parameter starting at exactly zero must still get a spread."""
    sampler, state = get_sampler(
        data_table=data_table,
        p0=np.array([1e-9, 2.0, 0.0]),
        labels=["norm", "index", "beta"],
        model=powerlaw_extra,
        nwalkers=20,
        nburn=0,
        guess=False,
        threads=1,
    )
    assert np.all(np.std(state.coords, axis=0) > 0)


def test_log_parameters_get_absolute_spread_in_dex():
    """A relative spread on a logarithmic value depends on the units: it is
    1.2 dex at -12 but vanishes at values close to zero. It is half an order of
    magnitude for any value."""
    np.random.seed(0)
    labels = ["log10(norm)", "log10(cutoff)", "log10(other)", "index"]
    p0 = np.array([-12.0, 0.01, 0.0, 2.0])
    walkers = _initial_walkers(p0, 4000, 0.1, labels)
    assert np.std(walkers[:, :3], axis=0) == pytest.approx(0.5, rel=0.05)
    # the others keep a relative spread
    assert np.std(walkers[:, 3]) == pytest.approx(0.2, rel=0.05)


def test_log_spread_is_configurable():
    np.random.seed(0)
    walkers = _initial_walkers(
        np.array([-12.0]), 4000, 0.005, ["log10(n)"], log_spread=0.1
    )
    assert np.std(walkers[:, 0]) == pytest.approx(0.1, rel=0.05)


def test_natural_log_parameters_use_the_same_factor():
    """0.5 dex is 0.5 ln(10) in natural logarithm units."""
    np.random.seed(0)
    walkers = _initial_walkers(np.array([-27.0]), 4000, 0.1, ["log(norm)"])
    assert np.std(walkers[:, 0]) == pytest.approx(0.5 * np.log(10), rel=0.05)


def test_small_linear_parameters_keep_a_relative_spread():
    """Only exact zeros get an absolute spread: a normalization of 1e-12 is a
    legitimate value that needs a relative spread."""
    np.random.seed(0)
    walkers = _initial_walkers(np.array([1e-12, 0.0]), 4000, 0.1, ["norm", "beta"])
    assert np.std(walkers[:, 0]) == pytest.approx(1e-13, rel=0.05)
    assert np.std(walkers[:, 1]) == pytest.approx(0.1, rel=0.05)


def test_sampler_uses_half_an_order_of_magnitude_for_log_parameters():
    sampler, state = get_sampler(
        data_table=data_table,
        p0=np.array([-10.0, 2.0]),
        labels=["log10(norm)", "index"],
        model=lambda pars, data: powerlaw_extra([10 ** pars[0], pars[1]], data),
        nwalkers=2000,
        nburn=0,
        guess=False,
        threads=1,
    )
    assert np.std(state.coords[:, 0]) == pytest.approx(0.5, rel=0.1)


def test_walkers_start_inside_prior():
    """Walkers drawn in a region forbidden by the prior must be redrawn."""

    def prior(pars):
        return uniform_prior(pars[0], 0.0, 1.0)

    np.random.seed(0)
    # p0 right at the prior bound: about half of the naive draws are outside
    walkers = _initial_walkers(np.array([1.0, 2.0]), 200, 0.1, ["a", "b"], prior)
    assert np.all(walkers[:, 0] <= 1.0)
    assert np.all(np.isfinite([prior(w) for w in walkers]))


def test_walkers_in_a_prior_much_narrower_than_the_ball_stay_distinct():
    """Collapsing the walkers onto p0 makes emcee fail with 'Initial state has
    a large condition number'."""

    def prior(pars):
        return uniform_prior(pars[1], 2.0, 2.0 + 1e-6)

    np.random.seed(1)
    walkers = _initial_walkers(
        np.array([1e-11, 2.0]), 16, 0.1, ["norm", "index"], prior
    )
    assert np.all(np.isfinite([prior(w) for w in walkers]))
    assert len(np.unique(walkers[:, 1])) == 16


def test_sampler_with_a_narrow_prior_runs():
    def prior(pars):
        return uniform_prior(pars[1], 2.0, 2.0 + 1e-6)

    np.random.seed(1)
    sampler, state = get_sampler(
        data_table=data_table,
        p0=np.array([1e-11, 2.0]),
        labels=["norm", "index"],
        model=powerlaw_extra,
        prior=prior,
        nwalkers=16,
        nburn=2,
        guess=False,
        threads=1,
    )
    assert np.all(np.isfinite(state.log_prob))


def test_p0_outside_the_prior_is_reported():
    p0 = np.array([1.0, 2.0])
    with log.log_to_list() as messages:
        walkers = _initial_walkers(p0, 5, 0.1, ["a", "b"], lambda pars: -np.inf)
    assert walkers.shape == (5, 2)
    assert any("not finite at the initial position" in m.message for m in messages)


def test_nan_prior_counts_as_forbidden():
    def prior(pars):
        return np.nan if pars[0] > 1.0 else 0.0

    np.random.seed(0)
    walkers = _initial_walkers(np.array([1.0, 2.0]), 100, 0.1, ["a", "b"], prior)
    assert np.all(walkers[:, 0] <= 1.0)


def test_ml_start_narrows_log_and_relative_spreads_in_the_same_proportion(monkeypatch):
    """With p0 at the ML point the relative spread goes from 10% to 0.5%, and
    the absolute spread of log parameters from 0.5 dex to 0.025 dex."""
    import naima.core

    monkeypatch.setattr(naima.core, "_prefit", lambda p0, *args: (p0, True))
    sampler, state = get_sampler(
        data_table=data_table,
        p0=np.array([-10.0, 2.0]),
        labels=["log10(norm)", "index"],
        model=lambda pars, data: powerlaw_extra([10 ** pars[0], pars[1]], data),
        nwalkers=4000,
        nburn=0,
        guess=False,
        prefit=True,
        threads=1,
    )
    assert np.std(state.coords[:, 0]) == pytest.approx(0.025, rel=0.05)
    assert np.std(state.coords[:, 1]) == pytest.approx(0.01, rel=0.05)
