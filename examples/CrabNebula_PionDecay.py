#!/usr/bin/env python
import numpy as np
from astropy import units as u
from astropy.io import ascii

import naima

## Read data

data=ascii.read('CrabNebula_HESS_2006_ipac.dat')

## Model definition

from naima.models import PionDecay, ExponentialCutoffPowerLaw

# Prepare an energy array for saving the particle distribution
proton_energy = np.logspace(-3,2,50)*u.TeV

def ppgamma(pars,data):
    amplitude = pars[0] / u.TeV
    alpha = pars[1]
    e_cutoff = (10**pars[2])*u.TeV

    ECPL = ExponentialCutoffPowerLaw(amplitude, 30*u.TeV, alpha, e_cutoff)
    PP = PionDecay(ECPL)

    # convert to same units as observed differential spectrum
    model = PP.flux(data,distance=2.0*u.kpc).to(data['flux'].unit)

    # Save a realization of the particle distribution to the metadata blob
    proton_dist= PP.particle_distribution(proton_energy)

    return model, (proton_energy, proton_dist), PP.Wp

## Prior definition

def lnprior(pars):
	"""
	Return probability of parameter values according to prior knowledge.
	Parameter limits should be done here through uniform prior ditributions
	"""

	logprob = naima.uniform_prior(pars[0],0.,np.inf) \
                + naima.uniform_prior(pars[1],-1,5)

	return logprob

if __name__=='__main__':
    # Set initial parameters
    p0=np.array((1e46,2.34,np.log10(80.),))
    labels=['norm','index','log10(cutoff)']

    # Run sampler
    sampler,pos = naima.run_sampler(data_table=data, p0=p0, labels=labels,
            model=ppgamma, prior=lnprior, nwalkers=16, nburn=50, nrun=10,
            threads=4)

    # Save sampler
    from astropy.extern import six
    from six.moves import cPickle
    sampler.pool=None
    cPickle.dump(sampler,open('CrabNebula_proton_sampler.pickle','wb'))

    # Save plots and results
    naima.save_diagnostic_plots('CrabNebula_proton', sampler, sed=True,
            blob_labels=['Spectrum', 'Proton energy distribution', '$W_p (E_p>E_\mathrm{thr})$'])
    naima.save_results_table('CrabNebula_proton',sampler)

