# Licensed under a 3-clause BSD style license - see LICENSE.rst
from pathlib import Path

import astropy.units as u
import numpy as np
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


def test_log_parameters_get_absolute_spread():
    """10% of a log10 parameter at -12 is 1.2 dex; the spread must be absolute."""
    np.random.seed(0)
    walkers = _initial_walkers(
        np.array([-12.0, 2.0]), 2000, 0.1, ["log10(norm)", "index"]
    )
    assert np.std(walkers[:, 0]) < 0.2
    assert np.std(walkers[:, 1]) > 0.15


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
