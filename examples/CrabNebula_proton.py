#!/usr/bin/env python
import numpy as np
import gammafit

from astropy import units as u
from astropy.io import ascii

## Read data

data=ascii.read('CrabNebula_HESS_2006.dat')

## Set initial parameters

p0=np.array((474,2.34,np.log10(80.),))
labels=['norm','index','log10(cutoff)']

## Model definition

ph_energy = u.Quantity(data['energy'])
# peak gamma ph_energy production is ~0.1*Ep, so enemid corresponds to Ep=10*enemid
# If a cutoff is present, this should be reduced to reduce parameter correlation
e_0 = 5.*np.sqrt(ph_energy[0]*ph_energy[-1])

from gammafit.models import PionDecay, ExponentialCutoffPowerLaw

ECPL = ExponentialCutoffPowerLaw(1 / u.TeV, e_0, 2, 60. * u.TeV)
PP = PionDecay(ECPL)

Epmin = ph_energy[0]*1e-2
Epmax = ph_energy[-1]*1e3
proton_energy = np.logspace(np.log10(Epmin.value),
                               np.log10(Epmax.value),50)*ph_energy.unit

def ppgamma(pars,data):

    PP.particle_distribution.amplitude = pars[0] / u.TeV
    PP.particle_distribution.alpha = pars[1]
    PP.particle_distribution.e_cutoff = (10**pars[2])*u.TeV

    # convert to same units as observed differential spectrum
    model = PP.flux(data)
    model = model.to('1/(s TeV)')/u.cm**2

    # Save a realization of the particle distribution to the metadata blob
    proton_dist= PP.particle_distribution(proton_energy) * u.Unit('1/TeV')

    return model, model, (proton_energy, proton_dist)

## Prior definition

def lnprior(pars):
	"""
	Return probability of parameter values according to prior knowledge.
	Parameter limits should be done here through uniform prior ditributions
	"""

	logprob = gammafit.uniform_prior(pars[0],0.,np.inf) \
                + gammafit.uniform_prior(pars[1],-1,5)

	return logprob

if __name__=='__main__':

## Run sampler

    sampler,pos = gammafit.run_sampler(data_table=data, p0=p0, labels=labels,
            model=ppgamma, prior=lnprior, nwalkers=16, nburn=50, nrun=10,
            threads=4)

## Save sampler

    from astropy.extern import six
    from six.moves import cPickle
    sampler.pool=None
    cPickle.dump(sampler,open('CrabNebula_proton_sampler.pickle','wb'))

## Diagnostic plots

    gammafit.generate_diagnostic_plots('CrabNebula_proton',sampler,sed=True)

