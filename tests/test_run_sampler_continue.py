# Licensed under a 3-clause BSD style license - see LICENSE.rst
import multiprocessing.pool
import os
import signal
import subprocess
import sys
import time
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


@pytest.mark.parametrize("exception", [RuntimeError, KeyboardInterrupt])
def test_pool_is_terminated_when_the_run_is_interrupted(monkeypatch, exception):
    """Joining a pool with an interrupted map blocks forever, so after an
    interruption the workers must be terminated."""
    terminated = []
    real_terminate = multiprocessing.pool.Pool.terminate

    def spy(self):
        terminated.append(self)
        return real_terminate(self)

    def interrupted(sampler, pos, nrun):
        raise exception("interrupted")

    monkeypatch.setattr(multiprocessing.pool.Pool, "terminate", spy)
    sampler, pos = get_sampler(threads=2, **KWARGS)
    monkeypatch.setattr(naima.core, "_run_mcmc", interrupted)

    with pytest.raises(exception):
        run_sampler(nrun=2, sampler=sampler, pos=pos)

    assert sampler._naima_pool is None
    assert len(terminated) >= 1


def test_n_run_matches_the_chain_and_is_not_set_by_a_failed_run(monkeypatch):
    sampler, pos = get_sampler(threads=1, **KWARGS)

    def failing(sampler, pos, nrun):
        raise RuntimeError("interrupted")

    monkeypatch.setattr(naima.core, "_run_mcmc", failing)
    with pytest.raises(RuntimeError):
        run_sampler(nrun=2, sampler=sampler, pos=pos)
    assert "n_run" not in sampler.run_info
    monkeypatch.undo()

    sampler, pos = run_sampler(nrun=3, sampler=sampler, pos=pos)
    sampler, pos = run_sampler(nrun=2, sampler=sampler, pos=pos)
    assert sampler.run_info["n_run"] == sampler.get_chain().shape[0] == 5


SIGINT_SCRIPT = """
import sys, time
import astropy.units as u
import numpy as np
from astropy.io import ascii
from naima.core import run_sampler

def slow(pars, data):
    time.sleep(0.05)
    x = data["energy"]
    return pars[0] * (x / x[0]) ** -pars[1] * u.Unit("1/(cm2 s TeV)")

if __name__ == "__main__":
    data = ascii.read(sys.argv[1])
    print("started", flush=True)
    try:
        run_sampler(data_table=data, p0=np.array([1e-11, 2.0]), model=slow,
                    nwalkers=40, nburn=0, nrun=50, threads=2, guess=False)
    except KeyboardInterrupt:
        print("interrupted", flush=True)
"""


@pytest.mark.skipif(not hasattr(os, "killpg"), reason="needs POSIX process groups")
def test_ctrl_c_with_process_pool_does_not_hang(tmp_path):
    """A terminal Ctrl-C sends SIGINT to the workers as well as the main
    process; run_sampler used to block forever joining the pool."""
    script = tmp_path / "interrupt_run.py"
    script.write_text(SIGINT_SCRIPT)
    proc = subprocess.Popen(
        [
            sys.executable,
            str(script),
            str(Path(__file__).parent / "data/CrabNebula_HESS_ipac.dat"),
        ],
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
        text=True,
        start_new_session=True,
    )
    try:
        assert proc.stdout.readline().strip() == "started"
        time.sleep(1.5)
        os.killpg(proc.pid, signal.SIGINT)
        out, _ = proc.communicate(timeout=30)
    except subprocess.TimeoutExpired:
        os.killpg(proc.pid, signal.SIGKILL)
        proc.communicate()
        pytest.fail("run_sampler hung after SIGINT")
    assert "interrupted" in out
