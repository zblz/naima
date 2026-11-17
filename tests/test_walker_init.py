# Licensed under a 3-clause BSD style license - see LICENSE.rst
from pathlib import Path

import astropy.units as u
import numpy as np
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


def test_walkers_forbidden_p0_falls_back_to_p0():
    """If the prior forbids everything, walkers fall back to p0."""
    p0 = np.array([1.0, 2.0])
    walkers = _initial_walkers(
        p0, 5, 0.1, ["a", "b"], lambda pars: -np.inf, max_tries=2
    )
    assert np.all(walkers == p0)
