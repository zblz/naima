# Licensed under a 3-clause BSD style license - see LICENSE.rst
from pathlib import Path

import astropy.units as u
import numpy as np
import pytest
from astropy.io import ascii

from naima.analysis import save_results_table
from naima.core import lnprobmodel, normal_prior, run_sampler

pytest.importorskip("matplotlib")

data_table = ascii.read(str(Path(__file__).parent / "data/CrabNebula_HESS_ipac.dat"))
UNIT = u.Unit("1/(cm2 s TeV)")


def powerlaw(pars, data):
    x = data["energy"]
    x0 = np.sqrt(x[0] * x[-1])
    return pars[0] * (x / x0) ** -pars[1] * UNIT


def informative_prior(pars):
    # a prior that adds a large, parameter-dependent term to the log-probability
    return normal_prior(pars[1], 2.5, 0.05) + 100.0


def run(data=data_table, **kwargs):
    return run_sampler(
        data_table=data,
        p0=np.array([3e-11, 2.6]),
        labels=["norm", "index"],
        model=powerlaw,
        nwalkers=16,
        nburn=5,
        nrun=5,
        threads=1,
        **kwargs,
    )


def test_max_loglikelihood_excludes_prior(tmp_path):
    sampler, _ = run(prior=informative_prior)
    t = save_results_table(str(tmp_path / "res"), sampler, include_blobs=False)

    meta = t.meta
    pars = np.array(meta["MAP_pars"])
    expected = lnprobmodel(powerlaw(pars, sampler.data), sampler.data)

    assert meta["MaxLogLikelihood"] == pytest.approx(float(expected))
    # the maximum log-probability includes the prior, which is >= 100 - 0.5*log(2 pi s^2)
    assert meta["MaxLogProb"] == pytest.approx(
        float(np.max(sampler.get_log_prob())), rel=1e-12
    )
    assert meta["MaxLogProb"] > meta["MaxLogLikelihood"] + 50
    assert meta["ML_pars"] == meta["MAP_pars"]


def test_bic_uses_loglikelihood_and_excludes_upper_limits(tmp_path):
    data = data_table.copy()
    # keep the upper limits of the table (their errors are zero, so they cannot
    # be treated as data points) and add more
    data["ul"] = np.asarray(data["ul"], dtype=bool)
    data["ul"][-3:] = True
    nreal = int(np.sum(~data["ul"]))
    assert 0 < nreal < len(data) - 3
    sampler, _ = run(data=data, prior=informative_prior)
    t = save_results_table(str(tmp_path / "res"), sampler, include_blobs=False)

    assert t.meta["n_data"] == nreal
    # the BIC must be finite, otherwise comparing it proves nothing
    assert np.isfinite(t.meta["MaxLogLikelihood"])
    expected = 2 * np.log(nreal) - 2 * t.meta["MaxLogLikelihood"]
    assert np.isfinite(expected)
    assert t.meta["BIC"] == pytest.approx(expected)


def test_ml_pars_is_not_a_yaml_alias_of_map_pars(tmp_path):
    """The same list object in two keys is written as a YAML anchor/alias,
    which readers other than astropy see as '*id001' instead of the values."""
    sampler, _ = run()
    save_results_table(str(tmp_path / "res"), sampler, include_blobs=False)
    text = next(tmp_path.glob("res_results*")).read_text()
    assert "&id" not in text
    assert "*id" not in text


def test_without_model_function_falls_back_to_logprob(tmp_path):
    sampler, _ = run()
    del sampler.modelfn
    t = save_results_table(str(tmp_path / "res"), sampler, include_blobs=False)
    assert t.meta["MaxLogLikelihood"] == pytest.approx(t.meta["MaxLogProb"])
