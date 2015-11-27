#!/usr/bin/env python
import numpy as np
import naima
import astropy.units as u
from astropy.io import ascii
from naima.models import InverseCompton, ExponentialCutoffPowerLaw

## Read data

data = ascii.read('RXJ1713_HESS_2007.dat')


def ElectronIC(pars, data):
    """
    Define particle distribution model, radiative model, and return model flux
    at data energy values
    """

    ECPL = ExponentialCutoffPowerLaw(pars[0] / u.eV, 10. * u.TeV, pars[1],
                                     10**pars[2] * u.TeV)
    IC = InverseCompton(ECPL, seed_photon_fields=['CMB'])

    return IC.flux(data, distance=1.0 * u.kpc)


def lnprior(pars):
    # Limit amplitude to positive domain
    logprob = naima.uniform_prior(pars[0], 0., np.inf)
    return logprob


if __name__ == '__main__':

    ## Set initial parameters and labels
    p0 = np.array((1e30, 3.0, np.log10(30),))
    labels = ['norm', 'index', 'log10(cutoff)']

    ## Run sampler
    sampler, pos = naima.run_sampler(data_table=data,
                                     p0=p0,
                                     labels=labels,
                                     model=ElectronIC,
                                     prior=lnprior,
                                     nwalkers=32,
                                     nburn=100,
                                     nrun=20,
                                     threads=4,
                                     prefit=True,
                                     interactive=False)
    ## Save run results
    out_root = 'RXJ1713_IC_minimal'
    naima.save_run(out_root, sampler)

    ## Save diagnostic plots and results table
    naima.save_diagnostic_plots(out_root, sampler, sed=False)
    naima.save_results_table(out_root, sampler)
