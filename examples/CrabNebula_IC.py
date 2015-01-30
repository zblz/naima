#!/usr/bin/env python
import numpy as np
import naima
import astropy.units as u
from astropy.constants import m_e,c
from astropy.io import ascii

## Read data

data=ascii.read('CrabNebula_HESS_2006_ipac.dat')

## Model definition

from naima.models import InverseCompton, ExponentialCutoffPowerLaw

def ElectronIC(pars,data):

    # Match parameters to ECPL properties, and give them the appropriate units
    amplitude = pars[0] / u.eV
    alpha = pars[1]
    e_cutoff = (10**pars[2])*u.TeV

    # Initialize instances of the particle distribution and radiative model
    ECPL = ExponentialCutoffPowerLaw(amplitude,10.*u.TeV, alpha, e_cutoff)
    IC = InverseCompton(ECPL,seed_photon_fields=['CMB'])

    # compute flux at the energies given in data['energy'], and convert to units
    # of flux data
    model = IC.flux(data,distance=2.0*u.kpc).to(data['flux'].unit)

    # The electron particle distribution (nelec) is saved in units or particles
    # per unit lorentz factor (E/mc2).  We define a mec2 unit and give nelec and
    # elec_energy the corresponding units.
    mec2 = u.Unit(m_e*c**2)
    nelec = IC._nelec * (1/mec2)
    elec_energy = IC._gam * mec2

    # The first array returned will be compared to the observed spectrum for
    # fitting. All subsequent objects will be stores in the sampler metadata
    # blobs.
    return model, (elec_energy,nelec), IC.We

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

## Set initial parameters and labels

    p0=np.array((4.9,3.3,np.log10(48.0),))
    labels=['norm','index','log10(cutoff)']

## Run sampler

    sampler,pos = naima.run_sampler(data_table=data, p0=p0, labels=labels, model=ElectronIC,
            prior=lnprior, nwalkers=50, nburn=50, nrun=10, threads=4)

## Save sampler
    from astropy.extern import six
    from six.moves import cPickle
    sampler.pool=None
    cPickle.dump(sampler,open('CrabNebula_IC_sampler.pickle','wb'))

## Diagnostic plots

    naima.save_diagnostic_plots('CrabNebula_IC',sampler,sed=True)
    naima.save_results_table('CrabNebula_IC',sampler)


