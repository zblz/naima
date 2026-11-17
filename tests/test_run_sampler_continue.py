# Licensed under a 3-clause BSD style license - see LICENSE.rst
from multiprocessing import Pool
from pathlib import Path

import astropy.units as u
import numpy as np
import pytest
from astropy.io import ascii

import naima.core
from naima.core import get_sampler, run_sampler

data_table = ascii.read(str(Path(__file__).parent / "data/CrabNebula_HESS_ipac.dat"))


def powerlaw(pars, data):
    x = data["energy"]
    x0 = np.sqrt(x[0] * x[-1])
    return pars[0] * (x / x0) ** -pars[1] * u.Unit("1/(cm2 s TeV)")


KWARGS = dict(
    data_table=data_table,
    p0=np.array([1e-11, 2.0]),
    labels=["norm", "index"],
    model=powerlaw,
    nwalkers=10,
    nburn=4,
    guess=False,
)


def test_fresh_run_discards_burn_in():
    sampler, pos = run_sampler(nrun=3, threads=1, **KWARGS)
    assert sampler.get_chain().shape[0] == 3


def test_run_of_get_sampler_discards_burn_in():
    sampler, pos = get_sampler(threads=1, **KWARGS)
    assert sampler.get_chain().shape[0] == 4
    sampler, pos = run_sampler(nrun=3, sampler=sampler, pos=pos)
    assert sampler.get_chain().shape[0] == 3


def test_run_can_be_continued():
    sampler, pos = run_sampler(nrun=3, threads=1, **KWARGS)
    first = sampler.get_chain().copy()

    sampler, pos = run_sampler(nrun=2, sampler=sampler, pos=pos)
    chain = sampler.get_chain()
    assert chain.shape[0] == 5
    # the first steps are preserved
    assert np.array_equal(chain[:3], first)
    assert sampler.run_info["n_run"] == 5


def test_run_can_be_continued_with_process_pool():
    sampler, pos = run_sampler(nrun=3, threads=2, **KWARGS)
    assert sampler._naima_pool is None
    assert sampler.pool is None

    sampler, pos = run_sampler(nrun=2, sampler=sampler, pos=pos)
    assert sampler.get_chain().shape[0] == 5
    assert sampler._naima_pool is None


def test_user_pool_is_used_and_left_open():
    with Pool(2) as pool:
        sampler, pos = run_sampler(nrun=2, pool=pool, **KWARGS)
        assert sampler.pool is pool
        assert sampler._naima_pool is None
        # still usable after the run
        assert pool.map(abs, [-1, -2]) == [1, 2]

        sampler, pos = run_sampler(nrun=2, sampler=sampler, pos=pos)
        assert sampler.get_chain().shape[0] == 4
        assert pool.map(abs, [-1]) == [1]


def test_pool_is_closed_when_the_run_fails(monkeypatch):
    created = []
    real_get_sampler = naima.core.get_sampler

    def spy(**kwargs):
        sampler, pos = real_get_sampler(**kwargs)
        created.append(sampler)
        return sampler, pos

    def failing_run(sampler, pos, nrun):
        raise RuntimeError("interrupted")

    monkeypatch.setattr(naima.core, "get_sampler", spy)
    # let the burn-in work, fail in the main run
    calls = []
    real_run = naima.core._run_mcmc

    def run_once_then_fail(sampler, pos, nrun):
        calls.append(nrun)
        if len(calls) > 1:
            return failing_run(sampler, pos, nrun)
        return real_run(sampler, pos, nrun)

    monkeypatch.setattr(naima.core, "_run_mcmc", run_once_then_fail)

    with pytest.raises(RuntimeError):
        run_sampler(nrun=2, threads=2, **KWARGS)

    assert created[0]._naima_pool is None
