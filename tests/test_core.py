# Licensed under a 3-clause BSD style license - see LICENSE.rst
import gc
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
from astropy import log
from astropy.io import ascii

import naima.core
from naima.core import _initial_walkers, get_sampler, lnprob, run_sampler, uniform_prior

data_table = ascii.read(str(Path(__file__).parent / "data/CrabNebula_HESS_ipac.dat"))
UNIT = u.Unit("1/(cm2 s TeV)")


def powerlaw(pars, data):
    x = data["energy"]
    x0 = np.sqrt(x[0] * x[-1])
    return pars[0] * (x / x0) ** -pars[1] * UNIT


# get_sampler: inputs of the caller


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


# get_sampler: initial walkers


def test_walkers_spread_around_zero_parameter():
    """A parameter starting at exactly zero must still get a spread."""
    sampler, state = get_sampler(
        data_table=data_table,
        p0=np.array([1e-9, 2.0, 0.0]),
        labels=["norm", "index", "beta"],
        model=powerlaw,
        nwalkers=20,
        nburn=0,
        guess=False,
        threads=1,
    )
    assert np.all(np.std(state.coords, axis=0) > 0)


def test_log_parameters_get_absolute_spread_in_dex():
    """A relative spread on a logarithmic value depends on the units: it is
    1.2 dex at -12 but vanishes at values close to zero. It is one order of
    magnitude for any value."""
    np.random.seed(0)
    labels = ["log10(norm)", "log10(cutoff)", "log10(other)", "index"]
    p0 = np.array([-12.0, 0.01, 0.0, 2.0])
    walkers = _initial_walkers(p0, 4000, 0.1, labels)
    assert np.std(walkers[:, :3], axis=0) == pytest.approx(1.0, rel=0.05)
    # the others keep a relative spread
    assert np.std(walkers[:, 3]) == pytest.approx(0.2, rel=0.05)


def test_log_spread_is_configurable():
    np.random.seed(0)
    walkers = _initial_walkers(
        np.array([-12.0]), 4000, 0.005, ["log10(n)"], log_spread=0.1
    )
    assert np.std(walkers[:, 0]) == pytest.approx(0.1, rel=0.05)


def test_natural_log_parameters_use_the_same_factor():
    """1 dex is ln(10) in natural logarithm units."""
    np.random.seed(0)
    walkers = _initial_walkers(np.array([-27.0]), 4000, 0.1, ["log(norm)"])
    assert np.std(walkers[:, 0]) == pytest.approx(np.log(10), rel=0.05)


def test_small_linear_parameters_keep_a_relative_spread():
    """Only exact zeros get an absolute spread: a normalization of 1e-12 is a
    legitimate value that needs a relative spread."""
    np.random.seed(0)
    walkers = _initial_walkers(np.array([1e-12, 0.0]), 4000, 0.1, ["norm", "beta"])
    assert np.std(walkers[:, 0]) == pytest.approx(1e-13, rel=0.05)
    assert np.std(walkers[:, 1]) == pytest.approx(0.1, rel=0.05)


def test_sampler_uses_one_order_of_magnitude_for_log_parameters():
    sampler, state = get_sampler(
        data_table=data_table,
        p0=np.array([-10.0, 2.0]),
        labels=["log10(norm)", "index"],
        model=lambda pars, data: powerlaw([10 ** pars[0], pars[1]], data),
        nwalkers=2000,
        nburn=0,
        guess=False,
        threads=1,
    )
    assert np.std(state.coords[:, 0]) == pytest.approx(1.0, rel=0.1)


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
        model=powerlaw,
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
    the absolute spread of log parameters from 1 dex to 0.05 dex."""
    import naima.core

    monkeypatch.setattr(naima.core, "_prefit", lambda p0, *args: (p0, True))
    sampler, state = get_sampler(
        data_table=data_table,
        p0=np.array([-10.0, 2.0]),
        labels=["log10(norm)", "index"],
        model=lambda pars, data: powerlaw([10 ** pars[0], pars[1]], data),
        nwalkers=4000,
        nburn=0,
        guess=False,
        prefit=True,
        threads=1,
    )
    assert np.std(state.coords[:, 0]) == pytest.approx(0.05, rel=0.05)
    assert np.std(state.coords[:, 1]) == pytest.approx(0.01, rel=0.05)


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
    walkers = _initial_walkers(np.array([5.0]), 20000, 0.1, [label], log_spread=1.0)
    expected = 0.1 * 5.0 if scale is None else scale
    assert np.isclose(walkers[:, 0].std(), expected, rtol=0.05)


# lnprob: failures of the model


def nan_above_index(pars, data):
    model = powerlaw(pars, data)
    if pars[1] > 2.1:
        model = model * np.nan
    return model


def raises_above_index(pars, data):
    if pars[1] > 2.1:
        raise OverflowError("model blew up")
    return powerlaw(pars, data)


def nan_with_blobs(pars, data):
    return nan_above_index(pars, data), np.sum(powerlaw(pars, data))


P0 = np.array([1e-11, 2.0])
LABELS = ["norm", "index"]


def test_lnprob_nan_model_is_neginf():
    out = lnprob(np.array([1e-11, 3.0]), data_table_validated(), nan_above_index, None)
    assert out[0] == -np.inf


def test_lnprob_nan_model_keeps_blobs():
    out = lnprob(np.array([1e-11, 3.0]), data_table_validated(), nan_with_blobs, None)
    assert out[0] == -np.inf
    assert len(out) == 3


def test_lnprob_exception_with_fallback_is_neginf():
    data = data_table_validated()
    fallback = (powerlaw(P0, data), np.nan)
    out = lnprob(
        np.array([1e-11, 3.0]), data, raises_above_index, None, fallback_blob=fallback
    )
    assert out[0] == -np.inf
    assert out[1:] == fallback


def test_lnprob_exception_without_fallback_is_raised():
    with pytest.raises(OverflowError):
        lnprob(np.array([1e-11, 3.0]), data_table_validated(), raises_above_index, None)


@pytest.mark.filterwarnings("ignore:invalid value encountered in scalar subtract")
@pytest.mark.parametrize("model", [nan_above_index, raises_above_index, nan_with_blobs])
def test_run_survives_failing_model(model):
    """emcee used to stop with 'Probability function returned NaN' or crash."""
    sampler, pos = run_sampler(
        data_table=data_table,
        p0=P0,
        labels=LABELS,
        model=model,
        nwalkers=20,
        nburn=3,
        nrun=3,
        threads=1,
        guess=False,
    )
    assert np.all(np.isfinite(pos.coords))


def test_exception_in_model_at_p0_is_still_raised():
    """A model that fails at the initial position is a user error."""
    with pytest.raises(OverflowError):
        get_sampler(
            data_table=data_table,
            p0=np.array([1e-11, 3.0]),
            labels=LABELS,
            model=raises_above_index,
            nwalkers=10,
            nburn=0,
            threads=1,
            guess=False,
        )


def data_table_validated():
    from naima.utils import validate_data_table

    return validate_data_table(data_table)


def buggy_above_index(pars, data):
    if pars[1] > 2.1:
        raise TypeError("bug in the model")
    return powerlaw(pars, data)


def invalid_above_index(pars, data):
    if pars[1] > 2.1:
        raise ValueError("parameter out of domain")
    return powerlaw(pars, data)


def test_bugs_in_the_model_are_not_swallowed():
    """Only numerical failures are mapped to -inf: a TypeError is a bug in the
    model and silently turning it into zero probability gives garbage."""
    data = data_table_validated()
    fallback = (powerlaw(P0, data), np.nan)
    with pytest.raises(TypeError):
        lnprob(
            np.array([1e-11, 3.0]),
            data,
            buggy_above_index,
            None,
            fallback_blob=fallback,
        )


def test_run_with_a_buggy_model_raises():
    with pytest.raises(TypeError):
        run_sampler(
            data_table=data_table,
            p0=P0,
            labels=LABELS,
            model=buggy_above_index,
            nwalkers=20,
            nburn=3,
            nrun=3,
            threads=1,
            guess=False,
        )


def test_first_model_failure_is_warned(monkeypatch):
    monkeypatch.setattr(naima.core, "_model_failure_warned", False)
    data = data_table_validated()
    fallback = (powerlaw(P0, data), np.nan)
    with log.log_to_list() as messages:
        for _ in range(3):
            lnprob(
                np.array([1e-11, 3.0]),
                data,
                invalid_above_index,
                None,
                fallback_blob=fallback,
            )
    warnings = [m for m in messages if m.levelname == "WARNING"]
    assert len(warnings) == 1
    assert "parameter out of domain" in warnings[0].message


@pytest.mark.filterwarnings("ignore:invalid value encountered in scalar subtract")
def test_no_finite_walker_is_warned():
    with log.log_to_list() as messages:
        run_sampler(
            data_table=data_table,
            p0=P0,
            labels=LABELS,
            model=powerlaw,
            prior=lambda pars: -np.inf,
            nwalkers=10,
            nburn=0,
            nrun=2,
            threads=1,
            guess=False,
        )
    assert any("finite log-probability" in m.message for m in messages)


def test_prefit_survives_a_model_that_fails_in_part_of_the_space():
    sampler, pos = get_sampler(
        data_table=data_table,
        p0=P0,
        labels=LABELS,
        model=raises_above_index,
        nwalkers=10,
        nburn=0,
        threads=1,
        prefit=True,
        guess=False,
    )
    assert np.all(np.isfinite(pos.coords))


def test_model_failure_warning_is_reset_in_a_new_fit(monkeypatch):
    monkeypatch.setattr(naima.core, "_model_failure_warned", True)
    get_sampler(
        data_table=data_table,
        p0=P0,
        labels=LABELS,
        model=powerlaw,
        nwalkers=10,
        nburn=0,
        threads=1,
        prefit=False,
        guess=False,
    )
    assert naima.core._model_failure_warned is False


# run_sampler: continued runs and process pools


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


def test_pool_is_terminated_when_the_sampler_is_discarded():
    sampler, pos = get_sampler(threads=2, prefit=False, **{**KWARGS, "nburn": 0})
    pool = sampler._naima_pool
    finalizer = sampler._naima_finalizer
    assert pool is not None and finalizer.alive
    del sampler
    gc.collect()
    assert not finalizer.alive
    assert pool._state != multiprocessing.pool.RUN
