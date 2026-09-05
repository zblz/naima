# Licensed under a 3-clause BSD style license - see LICENSE.rst
import warnings
from importlib.util import find_spec
from pathlib import Path

import astropy.units as u
import numpy as np
from astropy.io import ascii
from astropy.table import QTable
from astropy.tests.helper import pytest

from naima.core import (
    get_sampler,
    lnprob,
    lnprobmodel,
    log_uniform_prior,
    normal_prior,
    run_sampler,
    uniform_prior,
)

HAS_EMCEE = find_spec("emcee") is not None

HAS_SCIPY = find_spec("scipy") is not None
HAS_MATPLOTLIB = find_spec("matplotlib") is not None

if HAS_MATPLOTLIB:
    import matplotlib

    matplotlib.use("Agg")

# Read data
fname = Path(__file__).parent / "data/CrabNebula_HESS_ipac.dat"
data_table = ascii.read(str(fname))

# Read fake SED
fname0 = Path(__file__).parent / "data/Fake_ipac_sed.dat"
data_table_sed = ascii.read(str(fname0))

# Read spectrum with symmetric flux errors
fname2 = Path(__file__).parent / "data/CrabNebula_HESS_ipac_symmetric.dat"
data_table2 = ascii.read(str(fname2))

# Model definition


def cutoffexp(pars, data):
    """
    Powerlaw with exponential cutoff

    Parameters:
        - 0: PL normalization
        - 1: PL index
        - 2: cutoff energy
        - 3: cutoff exponent (beta)
    """

    x = data["energy"]
    # take logarithmic mean of first and last data points as normalization
    # energy
    x0 = np.sqrt(x[0] * x[-1])

    N = pars[0]
    gamma = pars[1]
    ecut = pars[2] * u.TeV
    # beta  = pars[3]
    beta = 1.0

    return (
        N * (x / x0) ** -gamma * np.exp(-((x / ecut) ** beta)) * u.Unit("1/(cm2 s TeV)")
    )


def cutoffexp_sed(pars, data):
    x = data["energy"]
    x0 = np.sqrt(x[0] * x[-1])
    N = pars[0]
    gamma = pars[1]
    ecut = pars[2] * u.TeV
    return N * (x / x0) ** -gamma * np.exp(-(x / ecut)) * u.Unit("erg/(cm2 s)")


def cutoffexp_blob(pars, data):
    model = cutoffexp(pars, data)
    return model, np.sum(model)


def cutoffexp_wrong(pars, data):
    return data["energy"] * u.m


# Prior definition


def lnprior(pars):
    """
    Return probability of parameter values according to prior knowledge.
    Parameter limits should be done here through uniform prior ditributions
    """

    logprob = (
        uniform_prior(pars[0], 0.0, np.inf)
        + normal_prior(pars[1], 1.4, 0.5)
        + uniform_prior(pars[2], 0.0, np.inf)
    )

    return logprob


# Set initial parameters

p0 = np.array((1e-9, 1.4, 14.0))
labels = ["norm", "index", "cutoff"]

# Initialize in different ways to test argument validation


@pytest.mark.skipif("not HAS_EMCEE")
def test_init():
    sampler, pos = get_sampler(
        data_table=data_table,
        p0=p0,
        labels=labels,
        model=cutoffexp,
        prior=lnprior,
        nwalkers=10,
        nburn=2,
        threads=1,
    )

    sampler, pos = run_sampler(
        data_table=data_table,
        p0=p0,
        labels=labels,
        model=cutoffexp,
        prior=lnprior,
        nwalkers=10,
        nburn=2,
        nrun=2,
        threads=1,
    )

    # test that the CL keyword has been correctly read
    assert np.all(sampler.data["cl"] == 0.99)


@pytest.mark.skipif("not HAS_EMCEE")
def test_inf_prior():
    pars = p0
    pars[0] = -1e-9
    _ = lnprob(pars, data_table, cutoffexp, lnprior)


@pytest.mark.skipif("not HAS_EMCEE")
def test_sed_conversion_in_lnprobmodel():
    sampler, pos = get_sampler(
        data_table=data_table,
        p0=p0,
        labels=labels,
        model=cutoffexp_sed,
        prior=lnprior,
        nwalkers=10,
        nburn=2,
        threads=1,
    )


def test_lnprobmodel_upper_limits_with_varying_cl():
    """
    Regression test for a bug where lnprobmodel indexed data['cl'] with the
    *count* of violated upper limits rather than a mask of which rows were
    violated. This gave the wrong result whenever 'cl' varies across rows
    (e.g. after combining tables from different instruments/confidence
    levels) and raised IndexError whenever every upper limit in the table
    was violated.
    """
    flux_unit = u.Unit("1/(cm2 s TeV)")
    data = QTable(
        {
            "energy": [1, 2, 3] * u.TeV,
            "flux": [1e-12, 1e-12, 1e-12] * flux_unit,
            "flux_error_lo": [1e-13, 1e-13, 1e-13] * flux_unit,
            "flux_error_hi": [1e-13, 1e-13, 1e-13] * flux_unit,
            "ul": np.array([True, True, True]),
            "cl": np.array([0.99, 0.9, 0.99]),
        }
    )

    # model is above all three upper limits -> all are violated, and used
    # to raise IndexError because data['cl'] only has 3 rows
    model = [2e-12, 2e-12, 2e-12] * flux_unit

    logprob = lnprobmodel(model, data)

    expected = np.sum(np.log(1.0 - data["cl"]))
    assert np.isclose(logprob, expected)

    # only the middle upper limit (cl=0.9) is violated: the penalty must
    # use *that* row's cl, not whichever row the violation count happens
    # to match
    model = [5e-13, 2e-12, 5e-13] * flux_unit
    logprob = lnprobmodel(model, data)
    assert np.isclose(logprob, np.log(1.0 - 0.9))


@pytest.mark.skipif("not HAS_EMCEE")
def test_wrong_model_units():
    # test exception raised when model and data spectra cannot be compared
    with pytest.raises(u.UnitsError):
        sampler, pos = get_sampler(
            data_table=data_table,
            p0=p0,
            labels=labels,
            model=cutoffexp_wrong,
            prior=lnprior,
            nwalkers=10,
            nburn=2,
            threads=1,
        )


@pytest.mark.skipif("not HAS_EMCEE or not HAS_SCIPY")
def test_prefit():
    sampler, pos = get_sampler(
        data_table=data_table,
        p0=p0,
        labels=labels,
        model=cutoffexp,
        prior=lnprior,
        nwalkers=10,
        nburn=5,
        threads=1,
        prefit=True,
    )


@pytest.mark.skipif("not HAS_EMCEE or not HAS_SCIPY or not HAS_MATPLOTLIB")
@pytest.mark.xfail(reason="interactive to be deprecated")
def test_interactive():
    with warnings.catch_warnings():
        # Matplotlib warns a lot when unable to bring up the widget
        warnings.simplefilter("ignore")
        sampler, pos = get_sampler(
            data_table=data_table,
            p0=p0,
            labels=labels,
            model=cutoffexp,
            prior=lnprior,
            nwalkers=10,
            nburn=5,
            threads=1,
            interactive=True,
        )


@pytest.mark.skipif("not HAS_EMCEE")
def test_init_symmetric_dflux():
    # symmetric data_table errors
    sampler, pos = run_sampler(
        data_table=data_table2,
        p0=p0,
        labels=labels,
        model=cutoffexp,
        prior=lnprior,
        nwalkers=10,
        nburn=2,
        nrun=2,
        threads=1,
    )


@pytest.mark.skipif("not HAS_EMCEE")
def test_init_labels():
    # labels
    sampler, pos = run_sampler(
        data_table=data_table,
        p0=p0,
        labels=None,
        model=cutoffexp,
        prior=lnprior,
        nwalkers=10,
        nrun=2,
        nburn=2,
        threads=1,
    )
    sampler, pos = run_sampler(
        data_table=data_table,
        p0=p0,
        labels=labels[:2],
        model=cutoffexp,
        prior=lnprior,
        nwalkers=10,
        nrun=2,
        nburn=2,
        threads=1,
    )


@pytest.mark.skipif("not HAS_EMCEE")
def test_init_prior():
    # no prior
    sampler, pos = run_sampler(
        data_table=data_table,
        p0=p0,
        labels=labels,
        model=cutoffexp,
        prior=None,
        nwalkers=10,
        nrun=2,
        nburn=2,
        threads=1,
    )


@pytest.mark.skipif("not HAS_EMCEE")
def test_init_exception_model():
    # test exception raised when no model or data_table are provided
    with pytest.raises(TypeError):
        sampler, pos = get_sampler(
            data_table=data_table,
            p0=p0,
            labels=labels,
            prior=lnprior,
            nwalkers=10,
            nburn=2,
            threads=1,
        )


@pytest.mark.skipif("not HAS_EMCEE")
def test_init_exception_data():
    with pytest.raises(TypeError):
        sampler, pos = get_sampler(
            p0=p0,
            labels=labels,
            model=cutoffexp,
            prior=lnprior,
            nwalkers=10,
            nburn=2,
            threads=1,
        )


@pytest.mark.skipif("not HAS_EMCEE")
def test_multiple_data_tables():
    sampler, pos = get_sampler(
        data_table=[data_table_sed, data_table],
        p0=p0,
        labels=labels,
        model=cutoffexp,
        prior=lnprior,
        nwalkers=10,
        nburn=2,
        threads=1,
    )


@pytest.mark.skipif("not HAS_EMCEE")
def test_data_table_in_list():
    sampler, pos = get_sampler(
        data_table=[data_table],
        p0=p0,
        labels=labels,
        model=cutoffexp,
        prior=lnprior,
        nwalkers=10,
        nburn=2,
        threads=1,
    )


def test_blob_shape():
    kwargs = dict(
        data_table=data_table,
        p0=p0,
        labels=labels,
        prior=lnprior,
        nwalkers=10,
        nburn=5,
        threads=1,
    )

    sampler, _ = get_sampler(model=cutoffexp, **kwargs)
    sampler_blobs, _ = get_sampler(model=cutoffexp_blob, **kwargs)

    # The blobs should contain the model with the same shape in both samplers
    # as the first blob
    assert (
        sampler.get_blobs(flat=True)[0, 0].shape
        == sampler_blobs.get_blobs(flat=True)[0, 0].shape
    )


@pytest.mark.skipif("not HAS_EMCEE")
def test_multiprocessing_pool_shutdown():
    """Regression test for #245: pool cleanup on Python 3.13+."""
    sampler, pos = run_sampler(
        data_table=data_table,
        p0=p0,
        labels=labels,
        model=cutoffexp,
        prior=lnprior,
        nwalkers=10,
        nburn=2,
        nrun=2,
        threads=2,
    )
    # Pool should be cleaned up after run_sampler
    assert getattr(sampler, "_naima_pool", None) is None


def test_normal_prior():
    """normal_prior should match a Gaussian log-pdf."""
    for value, mean, sigma in [(0.0, 0.0, 2.0), (1.5, 1.0, 0.3), (-2.0, 1.0, 5.0)]:
        expected = -0.5 * np.log(2 * np.pi * sigma**2) - (value - mean) ** 2 / (
            2.0 * sigma**2
        )
        assert normal_prior(value, mean, sigma) == pytest.approx(expected)

    # peak of the distribution is at the mean, and decreases away from it
    assert normal_prior(0.0, 0.0, 1.0) > normal_prior(1.0, 0.0, 1.0)


def test_log_uniform_prior():
    """log_uniform_prior should return -log(value) within bounds, -inf outside."""
    assert log_uniform_prior(10.0) == pytest.approx(-np.log(10.0))
    assert log_uniform_prior(10.0, 1.0, 100.0) == pytest.approx(-np.log(10.0))
    assert log_uniform_prior(0.5, umin=1.0) == -np.inf
    assert log_uniform_prior(200.0, umin=1.0, umax=100.0) == -np.inf
    assert log_uniform_prior(-1.0) == -np.inf


@pytest.mark.skipif("not HAS_EMCEE")
def test_single_thread_no_pool():
    """Test that threads=1 does not create a multiprocessing pool."""
    sampler, pos = run_sampler(
        data_table=data_table,
        p0=p0,
        labels=labels,
        model=cutoffexp,
        prior=lnprior,
        nwalkers=10,
        nburn=2,
        nrun=2,
        threads=1,
    )
    assert getattr(sampler, "_naima_pool", None) is None
