# Licensed under a 3-clause BSD style license - see LICENSE.rst
from pathlib import Path
import warnings
from importlib.util import find_spec

import astropy.units as u
import numpy as np
from astropy.io import ascii
from astropy.tests.helper import pytest

from naima.core import (
    get_sampler,
    lnprob,
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
