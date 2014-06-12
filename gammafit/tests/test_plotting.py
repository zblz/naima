
# Licensed under a 3-clause BSD style license - see LICENSE.rst
import numpy as np

import astropy.units as u
from astropy.tests.helper import pytest
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

import gammafit
from gammafit.utils import build_data_dict, generate_diagnostic_plots
from gammafit.core import run_sampler, uniform_prior
from gammafit.plot import plot_chain, plot_fit, plot_data

# Read data
specfile = six.StringIO(
    """
#
#
# Energy: TeV
# Flux: cm^{-2}.s^{-1}.TeV^{-1}

0.5   1.5e-11   0   0
0.7185 1.055e-11  7.266e-12 1.383e-11
0.8684 1.304e-11  1.091e-11 1.517e-11
1.051 9.211e-12  7.81e-12 1.061e-11
1.274 8.515e-12  7.557e-12 9.476e-12
1.546 5.378e-12  4.671e-12 6.087e-12
1.877 4.455e-12  3.95e-12 4.962e-12
2.275 3.754e-12  3.424e-12 4.088e-12
2.759 2.418e-12  2.15e-12 2.688e-12
3.352 1.605e-12  1.425e-12 1.788e-12
4.078 1.445e-12  1.319e-12 1.574e-12
4.956 9.24e-13  8.291e-13 1.021e-12
6.008 7.348e-13  6.701e-13 8.019e-13
7.271 3.863e-13  3.409e-13 4.333e-13
8.795 3.579e-13  3.222e-13 3.954e-13
10.65 1.696e-13  1.447e-13 1.955e-13
12.91 1.549e-13  1.343e-13 1.765e-13
15.65 6.695e-14  5.561e-14 7.925e-14
18.88 2.105e-14  7.146e-15 3.425e-14
22.62 3.279e-14  2.596e-14 4.03e-14
26.87 3.026e-14  2.435e-14 3.692e-14
31.61 1.861e-14  1.423e-14 2.373e-14
36.97 5.653e-15  3.484e-15 8.57e-15
43.08 3.479e-15  1.838e-15 5.889e-15
52.37 1.002e-15  1.693e-16 2.617e-15
100.0 1.5e-15  0 0
""")
spec = np.loadtxt(specfile)
specfile.close()

ene = spec[:, 0] * u.TeV
flux = spec[:, 1] * u.Unit('1/(cm2 s TeV)')
merr = spec[:, 1] - spec[:, 2]
perr = spec[:, 3] - spec[:, 1]
dflux = np.array((merr, perr)) * u.Unit('1/(cm2 s TeV)')
ul = spec[:, 2] == 0

data = build_data_dict(ene, None, flux, dflux, ul=ul, cl=0.9)

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

    x = data['ene'].copy()
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
    ene_part = np.logspace(np.log10(x[0].value) - 1,
                           np.log10(x[-1].value) + 1, 100) * x.unit
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

    return flux, flux, (x, flux), (ene, model), model1, model2, model3, (x, model3)

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

p0 = np.array((1e-9, 1.4, np.log10(14.0),))
labels = ['norm', 'index', 'cutoff', 'beta']

# Run sampler


@pytest.fixture
def sampler():
    sampler, pos = run_sampler(
        data=data, p0=p0, labels=labels, model=cutoffexp,
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
