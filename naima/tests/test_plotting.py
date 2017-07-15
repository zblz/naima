# Licensed under a 3-clause BSD style license - see LICENSE.rst
import os
from glob import glob
import numpy as np

import astropy.units as u
from astropy.tests.helper import pytest
from astropy.utils.data import get_pkg_data_filename
from astropy.io import ascii

try:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False

try:
    import emcee
    HAS_EMCEE = True
except ImportError:
    HAS_EMCEE = False

from ..analysis import save_diagnostic_plots, save_results_table
from ..core import run_sampler, uniform_prior
from ..plot import plot_chain, plot_fit, plot_data

# Read data
fname = get_pkg_data_filename('data/CrabNebula_HESS_ipac.dat')
data_table = ascii.read(fname)

fname2 = get_pkg_data_filename('data/CrabNebula_Fake_Xray.dat')
data_table2 = ascii.read(fname2)
data_list = [data_table2, data_table]

# Model definition


def cutoffexp(pars, data):
    """
    Powerlaw with exponential cutoff

    Parameters:
        - 0: log(PL normalization)
        - 1: PL index
        - 2: log10(cutoff energy)
        - 3: cutoff exponent (beta)
    """

    x = data['energy'].copy()
    # take logarithmic mean of first and last data points as normalization
    # energy
    x0 = np.sqrt(x[0] * x[-1])

    N = np.exp(pars[0])
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
                           (x / x0) ** -(gamma + 0.5)
                           ) * u.Unit('1/(cm2 s TeV)')

    model1 = (_model1 * (x ** 2) * 4 * np.pi * (2 * u.kpc) ** 2).to('erg/s')

    # save a model with no units to check that it is dealt with gracefully
    model2 = 1e-10 * np.ones(len(x))
    # save a model with wrong length to check that it is dealt with gracefully
    model3 = 1e-10 * np.ones(len(x) * 2) * u.Unit('erg/s')
    # add a scalar value to test plot_distribution
    model4 = np.trapz(model, ene).to('1/(cm2 s)')
    # and without units
    model5 = model4.value

    # save flux model as tuple with energies and without

    return (flux, (x, flux), (ene, model), (ene, model_part), model1, model2,
            model3, (x, model3), model4, model5)

# Prior definition
def lnprior(pars):
    """
    Return probability of parameter values according to prior knowledge.
    Parameter limits should be done here through uniform prior ditributions
    """

    # logprob = uniform_prior(np.exp(pars[0]), 0., np.inf) \
    logprob = uniform_prior(pars[1], -1, 5)

    return logprob

# Set initial parameters

p0=np.array((np.log(1.8e-12),2.4,np.log10(15.0),))
labels=['log(norm)','index','log10(cutoff)']

# Run sampler
@pytest.fixture
def sampler():
    sampler, pos = run_sampler(
        data_table=data_table, p0=p0, labels=labels, model=cutoffexp,
        prior=lnprior, nwalkers=10, nburn=2, nrun=2, threads=1)
    return sampler


@pytest.mark.skipif('not HAS_MATPLOTLIB or not HAS_EMCEE')
@pytest.mark.parametrize("last_step", [True, False])
@pytest.mark.parametrize("convert_log", [True, False])
@pytest.mark.parametrize("include_blobs", [True, False])
@pytest.mark.parametrize("format", ['ascii.ipac', 'ascii.ecsv', 'ascii'])
def test_results_table(sampler, last_step, convert_log, include_blobs, format):
    # set one keyword to a numpy array to try an break ecsv
    sampler.run_info['test'] = np.random.randn(3)
    save_results_table(
        'test_table', sampler,
        convert_log=convert_log, last_step=last_step,
        format=format, include_blobs=include_blobs)

    os.unlink(glob('test_table_results*')[0])


@pytest.mark.skipif('not HAS_MATPLOTLIB or not HAS_EMCEE')
@pytest.mark.parametrize("last_step", [True, False])
@pytest.mark.parametrize("p", [None, 1])
def test_chain_plots(sampler, last_step, p):
    fig = plot_chain(sampler, last_step=last_step, p=p)
    if fig is not None:
        plt.close(fig)


@pytest.mark.skipif('not HAS_MATPLOTLIB or not HAS_EMCEE')
@pytest.mark.parametrize("idx", range(4))
@pytest.mark.parametrize("sed", [True, False])
@pytest.mark.parametrize("last_step", [True, False])
@pytest.mark.parametrize("confs", [[1, 2], None])
@pytest.mark.parametrize("n_samples", [100, None])
@pytest.mark.parametrize("e_range", [[1 * u.GeV, 100 * u.TeV], None])
def test_fit_plots(sampler, idx, sed, last_step, confs, n_samples, e_range):
    # plot models with correct format
    fig = plot_fit(sampler, modelidx=idx, sed=sed,
                   last_step=last_step, plotdata=True,
                   confs=confs, n_samples=n_samples,
                   e_range=e_range)
    plt.close(fig)


@pytest.mark.skipif('not HAS_MATPLOTLIB or not HAS_EMCEE')
@pytest.mark.parametrize("threads", [None, 1, 4])
def test_threads_in_samples(sampler, threads):
    fig = plot_fit(sampler,
                   n_samples=100,
                   threads=threads,
                   e_range=[1 * u.GeV, 100 * u.TeV],
                   e_npoints=20)
    plt.close(fig)


@pytest.mark.skipif('not HAS_MATPLOTLIB or not HAS_EMCEE')
@pytest.mark.parametrize("sed", [True, False])
def test_plot_data(sampler, sed):
    fig = plot_data(sampler, sed=sed)
    plt.close(fig)


@pytest.mark.skipif('not HAS_MATPLOTLIB or not HAS_EMCEE')
def test_plot_data_reuse_fig(sampler):
    # change the energy units between calls
    data = sampler.data
    fig = plot_data(data, sed=True)
    data['energy'] = (data['energy']/1000).to('keV')
    plot_data(data, sed=True, figure=fig)
    plt.close(fig)


@pytest.mark.skipif('not HAS_MATPLOTLIB or not HAS_EMCEE')
@pytest.mark.parametrize("data_tables", [data_table, data_table2, data_list])
def test_plot_data_tables(sampler, data_tables):
    fig = plot_data(data_tables)
    plt.close(fig)


@pytest.mark.skipif('not HAS_MATPLOTLIB or not HAS_EMCEE')
def test_fit_data_units(sampler):
    fig = plot_fit(sampler, modelidx=0, sed=None)
    plt.close(fig)


@pytest.mark.skipif('not HAS_MATPLOTLIB or not HAS_EMCEE')
def test_diagnostic_plots(sampler):
    # Diagnostic plots
    # try to plot all models, including those with wrong format/units

    blob_labels = ['Model', 'Flux', 'Model', 'Particle Distribution',
                   'Broken PL', 'Wrong', 'Wrong', 'Wrong', 'Scalar',
                   'Scalar without units']

    save_diagnostic_plots('test_function_1', sampler, blob_labels=blob_labels)
    save_diagnostic_plots('test_function_2', sampler, sed=True,
                          blob_labels=blob_labels[:4])
    save_diagnostic_plots('test_function_3', sampler, pdf=True)
