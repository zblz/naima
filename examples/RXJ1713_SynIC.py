#!/usr/bin/env python
import numpy as np
import astropy.units as u
from astropy.io import ascii

import naima

## Read data

# We only consider every fifth X-ray spectral point to speed-up calculations for this example
# DO NOT do this for a final analysis!
soft_xray = ascii.read('RXJ1713_Suzaku-XIS.dat')[::5]
vhe = ascii.read('RXJ1713_HESS_2007.dat')

## Model definition

from naima.models import InverseCompton, Synchrotron, ExponentialCutoffPowerLaw


def ElectronSynIC(pars, data):

    # Match parameters to ECPL properties, and give them the appropriate units
    amplitude = 10**pars[0] / u.eV
    alpha = pars[1]
    e_cutoff = (10**pars[2]) * u.TeV
    B = pars[3] * u.uG

    # Initialize instances of the particle distribution and radiative models
    ECPL = ExponentialCutoffPowerLaw(amplitude, 10. * u.TeV, alpha, e_cutoff)
    # Compute IC on CMB and on a FIR component with values from GALPROP for the
    # position of RXJ1713
    IC = InverseCompton(
        ECPL,
        seed_photon_fields=['CMB', ['FIR', 26.5 * u.K, 0.415 * u.eV / u.cm**3]],
        Eemin=100 * u.GeV)
    SYN = Synchrotron(ECPL, B=B)

    # compute flux at the energies given in data['energy']
    model = (IC.flux(data,
                     distance=1.0 * u.kpc) + SYN.flux(data,
                                                      distance=1.0 * u.kpc))

    # The first array returned will be compared to the observed spectrum for
    # fitting. All subsequent objects will be stored in the sampler metadata
    # blobs.
    return model, IC.compute_We(Eemin=1 * u.TeV)

## Prior definition


def lnprior(pars):
    """
    Return probability of parameter values according to prior knowledge.
    Parameter limits should be done here through uniform prior ditributions
    """
    # Limit norm and B to be positive
    logprob = naima.uniform_prior(pars[0], 0., np.inf) \
                + naima.uniform_prior(pars[1], -1, 5) \
                + naima.uniform_prior(pars[3], 0, np.inf)

    return logprob


if __name__ == '__main__':

    ## Set initial parameters and labels
    # Estimate initial magnetic field and get value in uG
    B0 = 2 * naima.estimate_B(soft_xray, vhe).to('uG').value

    p0 = np.array((33, 2.5, np.log10(48.0), B0))
    labels = ['log10(norm)', 'index', 'log10(cutoff)', 'B']

    ## Run sampler
    sampler, pos = naima.run_sampler(data_table=[soft_xray, vhe],
                                     p0=p0,
                                     labels=labels,
                                     model=ElectronSynIC,
                                     prior=lnprior,
                                     nwalkers=32,
                                     nburn=100,
                                     nrun=20,
                                     threads=4,
                                     prefit=True,
                                     interactive=False)

    ## Save run results to HDF5 file (can be read later with naima.read_run)
    naima.save_run('RXJ1713_SynIC', sampler)

    ## Diagnostic plots
    naima.save_diagnostic_plots('RXJ1713_SynIC',
                                sampler,
                                sed=True,
                                blob_labels=['Spectrum', '$W_e$($E_e>1$ TeV)'])
    naima.save_results_table('RXJ1713_SynIC', sampler)
