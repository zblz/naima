# Licensed under a 3-clause BSD style license - see LICENSE.rst
from pathlib import Path

import astropy.units as u
import numpy as np
from astropy.io import ascii

from naima.core import get_sampler

data_table = ascii.read(str(Path(__file__).parent / "data/CrabNebula_HESS_ipac.dat"))


def powerlaw(pars, data):
    x = data["energy"]
    x0 = np.sqrt(x[0] * x[-1])
    return pars[0] * (x / x0) ** -pars[1] * u.Unit("1/(cm2 s TeV)")


def zeros(pars, data):
    return np.zeros(len(data["energy"])) * u.Unit("1/(cm2 s TeV)")


def run(model=powerlaw, **kwargs):
    kwargs.setdefault("nwalkers", 10)
    return get_sampler(data_table=data_table, model=model, nburn=0, threads=1, **kwargs)


def test_p0_is_not_modified_by_normalization_guess():
    p0 = np.array([1e-12, 2.0])
    original = p0.copy()
    sampler, _ = run(p0=p0, labels=["norm", "index"])
    assert np.array_equal(p0, original)
    # the guess was applied to the sampler's own copy
    assert sampler.run_info["p0"][0] != original[0]


def test_p0_list_is_not_modified():
    p0 = [1e-12, 2.0]
    run(p0=p0, labels=["norm", "index"])
    assert p0 == [1e-12, 2.0]


def test_labels_are_not_extended_in_place():
    labels = ["norm"]
    sampler, _ = run(p0=np.array([1e-12, 2.0]), labels=labels)
    assert labels == ["norm"]
    assert sampler.labels == ["norm", "par1"]


def test_labels_tuple_is_accepted():
    sampler, _ = run(p0=np.array([1e-12, 2.0]), labels=("norm",))
    assert sampler.labels == ["norm", "par1"]


def test_norm_guess_skipped_for_zero_model_flux():
    """A model with zero flux gave an infinite ratio and an infinite p0."""
    sampler, state = run(model=zeros, p0=np.array([1e-12, 2.0]), labels=["norm", "i"])
    assert np.all(np.isfinite(sampler.run_info["p0"]))
    assert sampler.run_info["p0"] == [1e-12, 2.0]
    assert np.all(np.isfinite(state.coords))
