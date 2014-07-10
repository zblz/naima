#!/usr/bin/env python
import numpy as np
import gammafit
import astropy.units as u
from astropy.constants import m_e,c
from astropy.io import ascii

## Read data

data=ascii.read('CrabNebula_HESS_2006.dat')

## Set initial parameters

p0=np.array((4.9,3.3,np.log10(48.0),))
labels=['norm','index','log10(cutoff)']

## Model definition

from gammafit.models import InverseCompton, ExponentialCutoffPowerLaw

ECPL = ExponentialCutoffPowerLaw(1 * u.Unit('1/TeV'),10.*u.TeV,2,10.*u.TeV)
IC = InverseCompton(ECPL,seed_photon_fields=['CMB'])

distance = 2.0 * u.kpc

mec2 = u.Unit(m_e*c**2)

def ElectronIC(pars,data):

    IC.particle_distribution.amplitude = pars[0] / u.TeV
    IC.particle_distribution.alpha = pars[1]
    IC.particle_distribution.e_cutoff = (10**pars[2])*u.TeV

    # convert to same units as observed differential spectrum
    model = IC.flux(data,distance).to('1/(s cm2 TeV)')

    # The electron particle distribution (nelec) is saved in units or particles
    # per unit lorentz factor (E/mc2).  We define a mec2 unit and give nelec and
    # elec_energy the corresponding units.
    nelec = IC._nelec * (1/mec2)
    elec_energy = IC._gam * mec2

    # The first array returned will be compared to the observed spectrum for
    # fitting. All subsequent objects will be stores in the sampler metadata
    # blobs.
    return model, model, (elec_energy,nelec), IC.We

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

    sampler,pos = gammafit.run_sampler(data_table=data, p0=p0, labels=labels, model=ElectronIC,
            prior=lnprior, nwalkers=50, nburn=50, nrun=10, threads=4)

## Save sampler
    from astropy.extern import six
    from six.moves import cPickle
    sampler.pool=None
    cPickle.dump(sampler,open('CrabNebula_IC_sampler.pickle','wb'))

## Diagnostic plots

    gammafit.generate_diagnostic_plots('CrabNebula_IC',sampler,sed=True)


