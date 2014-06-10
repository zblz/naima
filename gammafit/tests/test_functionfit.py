# Licensed under a 3-clause BSD style license - see LICENSE.rst
from StringIO import StringIO
import numpy as np
from astropy.tests.helper import pytest
from ..utils import build_data_dict, generate_diagnostic_plots
from ..core import run_sampler, get_sampler, uniform_prior, normal_prior
# Use batch backend to avoid $DISPLAY errors
import matplotlib
matplotlib.use("Agg")

try:
    import emcee
    HAS_EMCEE = True
except ImportError:
    HAS_EMCEE = False

## Read data
specfile=StringIO(
"""
# Crab Nebula spectrum measured by HESS taken from table 5 of
# Aharonian et al. 2006, A&A 457, 899
# ADS bibcode: 2006A&A...457..899A

# Column 1: Mean energy (TeV)
# Column 2: Excess events
# Column 3: Significance
# Column 4: Differential flux (1/cm2/s/TeV)
# Column 5: Upper 1-sigma flux error
# Column 6: Lower 1-sigma flux error

0.519  975   42.9  1.81e-10  0.06e-10  0.06e-10
0.729  1580  56.0  7.27e-11  0.20e-11  0.19e-11
1.06   1414  55.3  3.12e-11  0.09e-11  0.09e-11
1.55   1082  47.3  1.22e-11  0.04e-11  0.04e-11
2.26   762   39.5  4.60e-12  0.18e-12  0.18e-12
3.3    443   29.5  1.53e-12  0.08e-12  0.08e-12
4.89   311   24.9  6.35e-13  0.39e-13  0.38e-13
7.18   186   19.6  2.27e-13  0.18e-13  0.17e-13
10.4   86    13.1  6.49e-14  0.77e-14  0.72e-14
14.8   36    8.1   1.75e-14  0.33e-14  0.30e-14
20.9   23    7.5   7.26e-15  1.70e-15  1.50e-15
30.5   4     2.9   9.58e-16  5.60e-16  4.25e-16
""")
spec=np.loadtxt(specfile)
specfile.close()

ene=spec[:,0]*u.TeV
flux=spec[:,3]*u.Unit('1/(cm2 s TeV)')
merr=spec[:,4]
perr=spec[:,5]
dflux=np.array((merr,perr))*u.Unit('1/(cm2 s TeV)')

data=build_data_dict(ene,None,flux,dflux,)


@pytest.mark.skipif('not HAS_EMCEE')
def test_function_sampler():

## Model definition

    def cutoffexp(pars,data):
        """
        Powerlaw with exponential cutoff

        Parameters:
            - 0: PL normalization
            - 1: PL index
            - 2: cutoff energy
            - 3: cutoff exponent (beta)
        """

        x=data['ene']
        # take logarithmic mean of first and last data points as normalization energy
        x0=np.sqrt(x[0]*x[-1])

        N     = pars[0]
        gamma = pars[1]
        ecut  = pars[2]*u.TeV
        #beta  = pars[3]
        beta  = 1.

        return N*(x/x0)**-gamma*np.exp(-(x/ecut)**beta) * u.Unit('1/(cm2 s TeV)')

## Prior definition

    def lnprior(pars):
        """
        Return probability of parameter values according to prior knowledge.
        Parameter limits should be done here through uniform prior ditributions
        """

        logprob = uniform_prior(pars[0],0.,np.inf) \
                + normal_prior(pars[1],1.4,0.5) \
                + uniform_prior(pars[2],0.,np.inf)

        return logprob

## Set initial parameters

    p0=np.array((1e-9,1.4,14.0,))
    labels=['norm','index','cutoff','beta']

## Initialize in different ways to test argument validation

    sampler,pos = get_sampler(data=data, p0=p0, labels=labels, model=cutoffexp,
            prior=lnprior, nwalkers=10, nburn=0, threads=1)

    # labels
    sampler,pos = run_sampler(data=data, p0=p0, labels=None, model=cutoffexp,
            prior=lnprior, nwalkers=10, nrun=2, nburn=0, threads=1)
    sampler,pos = run_sampler(data=data, p0=p0, labels=labels[:2], model=cutoffexp,
            prior=lnprior, nwalkers=10, nrun=2, nburn=0, threads=1)

    # no prior
    sampler,pos = run_sampler(data=data, p0=p0, labels=labels, model=cutoffexp,
            prior=None, nwalkers=10, nrun=2, nburn=0, threads=1)

    sampler,pos = run_sampler(data=data, p0=p0, labels=labels, model=cutoffexp,
            prior=lnprior, nwalkers=10, nburn=2, nrun=2, threads=1)


