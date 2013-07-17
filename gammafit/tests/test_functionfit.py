#!/usr/bin/python
import numpy as np

import astropy.units as u

import gammafit
from ..utils import build_data_dict, generate_diagnostic_plots
from ..core import run_sampler, uniform_prior

## Read data
from StringIO import StringIO

specfile=StringIO(
"""
#
#
# Energy: TeV
# Flux: cm^{-2}.s^{-1}.TeV^{-1}

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
""")
spec=np.loadtxt(specfile)
specfile.close()

ene=spec[:,0]
dene=gammafit.generate_energy_edges(ene)

flux=spec[:,1]
merr=spec[:,1]-spec[:,2]
perr=spec[:,3]-spec[:,1]
dflux=np.array(zip(merr,perr))

ul=(dflux[:,0]==0.)
cl=0.99



def test_function_sampler():
    data=build_data_dict(ene,dene,flux,dflux,ul,cl)

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
        ecut  = pars[2]
        #beta  = pars[3]
        beta  = 1.

        return N*(x/x0)**-gamma*np.exp(-(x/ecut)**beta)

## Prior definition

    def lnprior(pars):
        """
        Return probability of parameter values according to prior knowledge.
        Parameter limits should be done here through uniform prior ditributions
        """

        logprob = uniform_prior(pars[0],0.,np.inf) \
                + uniform_prior(pars[1],-1,5) \
                + uniform_prior(pars[2],0.,np.inf) 

        return logprob

## Set initial parameters

    p0=np.array((1e-9,1.4,14.0,))
    labels=['norm','index','cutoff','beta']

## Run sampler

    sampler,pos = run_sampler(data=data, p0=p0, labels=labels, model=cutoffexp,
            prior=lnprior, nwalkers=10, nburn=2, nrun=2, threads=1)

## Diagnostic plots

    #generate_diagnostic_plots('velax_function',sampler)


