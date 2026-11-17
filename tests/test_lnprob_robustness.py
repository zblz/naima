# Licensed under a 3-clause BSD style license - see LICENSE.rst
from pathlib import Path

import astropy.units as u
import numpy as np
import pytest
from astropy.io import ascii

from naima.core import get_sampler, lnprob, run_sampler

data_table = ascii.read(str(Path(__file__).parent / "data/CrabNebula_HESS_ipac.dat"))
UNIT = u.Unit("1/(cm2 s TeV)")


def powerlaw(pars, data):
    x = data["energy"]
    x0 = np.sqrt(x[0] * x[-1])
    return pars[0] * (x / x0) ** -pars[1] * UNIT


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
