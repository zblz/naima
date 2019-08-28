# Licensed under a 3-clause BSD style license - see LICENSE.rst
import astropy.units as u
import numpy as np
from astropy.io import ascii
from astropy.tests.helper import pytest
from astropy.utils.data import get_pkg_data_filename

from ..core import run_sampler, uniform_prior

# Read data
fname = get_pkg_data_filename("data/CrabNebula_HESS_ipac.dat")
data_table = ascii.read(fname)

# Model definition
def cutoffexp(pars, input_data):
    """
    Powerlaw with exponential cutoff

    Parameters:
        - 0: log(PL normalization)
        - 1: PL index
        - 2: log10(cutoff energy)
        - 3: cutoff exponent (beta)
    """

    data = input_data

    x = data["energy"].copy()
    # take logarithmic mean of first and last data points as normalization
    # energy
    x0 = np.sqrt(x[0] * x[-1])

    N = np.exp(pars[0])
    gamma = pars[1]
    ecut = (10 ** pars[2]) * u.TeV
    # beta  = pars[3]
    beta = 1.0

    flux = (
        N
        * (x / x0) ** -gamma
        * np.exp(-(x / ecut) ** beta)
        * u.Unit("1/(cm2 s TeV)")
    )

    # save a model with different energies than the data
    ene = (
        np.logspace(np.log10(x[0].value) - 1, np.log10(x[-1].value) + 1, 100)
        * x.unit
    )
    model = (
        N * (ene / x0) ** -gamma * np.exp(-(ene / ecut) ** beta)
    ) * u.Unit("1/(cm2 s TeV)")

    # save a particle energy distribution
    model_part = (
        N * (ene / x0) ** -gamma * np.exp(-(ene / ecut) ** beta)
    ) * u.Unit("1/(TeV)")

    # save a broken powerlaw in luminosity units
    _model1 = (
        N
        * np.where(
            x <= x0, (x / x0) ** -(gamma - 0.5), (x / x0) ** -(gamma + 0.5)
        )
        * u.Unit("1/(cm2 s TeV)")
    )

    model1 = (_model1 * (x ** 2) * 4 * np.pi * (2 * u.kpc) ** 2).to("erg/s")

    # save a model with no units to check that it is dealt with gracefully
    model2 = 1e-10 * np.ones(len(x))
    # save a model with wrong length to check that it is dealt with gracefully
    model3 = 1e-10 * np.ones(len(x) * 2) * u.Unit("erg/s")
    # add a scalar value to test plot_distribution
    model4 = np.trapz(model, ene).to("1/(cm2 s)")
    # and without units
    model5 = model4.value

    # save flux model as tuple with energies and without

    return (
        flux,
        (x, flux),
        (ene, model),
        (ene, model_part),
        model1,
        model2,
        model3,
        (x, model3),
        model4,
        model5,
    )


def simple_cutoffexp(pars, data):
    (flux, _, model, _, _, _, _, _, model4, model5) = cutoffexp(pars, data)
    return flux, model, model4, model5


# Prior definition
def lnprior(pars):
    """
    Return probability of parameter values according to prior knowledge.
    Parameter limits should be done here through uniform prior ditributions
    """

    # logprob = uniform_prior(np.exp(pars[0]), 0., np.inf) \
    logprob = uniform_prior(pars[1], -1, 5)

    return logprob


# Run sampler
@pytest.fixture(scope="module")
def sampler():
    p0 = np.array((np.log(1.8e-12), 2.4, np.log10(15.0)))
    labels = ["log(norm)", "index", "log10(cutoff)"]
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
    return sampler


@pytest.fixture(scope="module")
def simple_sampler():
    p0 = np.array((np.log(1.8e-12), 2.4, np.log10(15.0)))
    labels = ["log(norm)", "index", "log10(cutoff)"]
    sampler, pos = run_sampler(
        data_table=data_table,
        p0=p0,
        labels=labels,
        model=simple_cutoffexp,
        prior=lnprior,
        nwalkers=10,
        nburn=2,
        nrun=2,
        threads=1,
    )
    return sampler
