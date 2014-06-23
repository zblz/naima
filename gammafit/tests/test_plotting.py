
# Licensed under a 3-clause BSD style license - see LICENSE.rst
import numpy as np

import astropy.units as u
from astropy.tests.helper import pytest
from astropy.utils.data import get_pkg_data_filename
from astropy.extern import six

try:
    import matplotlib
    matplotlib.use('Agg')
    HAS_MATPLOTLIB = True
except:
    HAS_MATPLOTLIB = False

try:
    import emcee
    HAS_EMCEE = True
except:
    HAS_EMCEE = False

from ..utils import generate_diagnostic_plots
from ..core import run_sampler, uniform_prior
from ..plot import plot_chain, plot_fit, plot_data

# Read data
from astropy.io import ascii

fname = get_pkg_data_filename('data/CrabNebula_HESS_ipac.dat')
data_table = ascii.read(fname)

# Model definition

def cutoffexp(pars, data):
    """
    Powerlaw with exponential cutoff

    Parameters:
        - 0: PL normalization
        - 1: PL index
        - 2: log10(cutoff energy)
        - 3: cutoff exponent (beta)
    """

    x = data['energy'].copy()
    # take logarithmic mean of first and last data points as normalization
    # energy
    x0 = np.sqrt(x[0] * x[-1])

    N = pars[0]
    gamma = pars[1]
    ecut = (10**pars[2]) * u.TeV
    # beta  = pars[3]
    beta = 1.

    flux = N * (x / x0) ** -gamma * np.exp(
        -(x / ecut) ** beta) * u.Unit('1/(cm2 s TeV)')

    # save a model with different energies than the data
    ene = np.logspace(np.log10(x[0].value) - 1,
                      np.log10(x[-1].value) + 1, 100) * x.unit
    model = (N * (ene / x0) ** -gamma *
             np.exp(-(ene / ecut) ** beta)) * u.Unit('1/(cm2 s TeV)')

    # save a particle energy distribution
    model_part = (N * (ene / x0) ** -gamma *
                  np.exp(-(ene / ecut) ** beta)) * u.Unit('1/(TeV)')

    # save a broken powerlaw in luminosity units
    _model1 = N * np.where(x <= x0,
                          (x / x0) ** -(gamma - 0.5),
                          (x / x0) ** -(gamma + 0.5)) * u.Unit('1/(cm2 s TeV)')

    model1 = (_model1 * (x ** 2) * 4 * np.pi * (2 * u.kpc) ** 2).to('erg/s')

    # save a model with no units to check that it is dealt with gracefully
    model2 = 1e-10 * np.ones(len(x))
    # save a model with wrong length to check that it is dealt with gracefully
    model3 = 1e-10 * np.ones(len(x) * 2) * u.Unit('erg/s')

    # save flux model as tuple with energies and without

    return flux, flux, (x, flux), (ene, model), (ene, model_part), model1, model2, model3, (x, model3)

# Prior definition


def lnprior(pars):
    """
    Return probability of parameter values according to prior knowledge.
    Parameter limits should be done here through uniform prior ditributions
    """

    logprob = uniform_prior(pars[0], 0., np.inf) \
        + uniform_prior(pars[1], -1, 5)

    return logprob

# Set initial parameters

p0=np.array((1.8e-12,2.4,np.log10(15.0),))
labels=['norm','index','log10(cutoff)']

# Run sampler


@pytest.fixture
def sampler():
    sampler, pos = run_sampler(
        data_table=data_table, p0=p0, labels=labels, model=cutoffexp,
        prior=lnprior, nwalkers=10, nburn=2, nrun=2, threads=1)
    return sampler


@pytest.mark.skipif('not HAS_MATPLOTLIB or not HAS_EMCEE')
def test_chain_plots(sampler):

    f = plot_chain(sampler, last_step=True)
    f = plot_chain(sampler, last_step=False)
    f = plot_chain(sampler, p=1)

    del f


@pytest.mark.skipif('not HAS_MATPLOTLIB or not HAS_EMCEE')
def test_fit_plots(sampler):

    # plot models with correct format
    for idx in range(4):
        for sed in [True, False]:
            for last_step in [True, False]:
                f = plot_fit(sampler, modelidx=idx, sed=sed,
                             last_step=last_step, plotdata=True)
                del f


@pytest.mark.skipif('not HAS_MATPLOTLIB or not HAS_EMCEE')
def test_plot_data(sampler):
    # only plot data
    f = plot_data(sampler,)
    f = plot_data(sampler, sed=True)
    f = plot_data(sampler, sed=True, figure=f)
    # try to break it
    f = plot_data(sampler, plotdata=False)
    f = plot_data(sampler, confs=[3, 1, 0.5])
    del f


@pytest.mark.skipif('not HAS_MATPLOTLIB or not HAS_EMCEE')
def test_fit_data_units(sampler):

    plot_fit(sampler, modelidx=0, sed=None)


@pytest.mark.skipif('not HAS_MATPLOTLIB or not HAS_EMCEE')
def test_diagnostic_plots(sampler):
    # Diagnostic plots
    # try to plot all models, including those with wrong format/units

    generate_diagnostic_plots('test_function_1', sampler)
    generate_diagnostic_plots('test_function_2', sampler, sed=True)
    generate_diagnostic_plots(
        'test_function_3', sampler, sed=[True, True, False, ])
    generate_diagnostic_plots('test_function_4', sampler, sed=False)
    generate_diagnostic_plots('test_function_5', sampler, sed=True, pdf=True)
