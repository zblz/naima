#!/usr/bin/env python
import numpy as np
import astropy.units as u
from astropy.constants import m_e,c
from astropy.io import ascii

import naima

## Read data

xray = ascii.read('CrabNebula_Fake_Xray.dat')
vhe = ascii.read('CrabNebula_HESS_2006_ipac.dat')

## Model definition

from naima.models import InverseCompton, Synchrotron, ExponentialCutoffPowerLaw

def ElectronSynIC(pars,data):

    # Match parameters to ECPL properties, and give them the appropriate units
    amplitude = pars[0] / u.eV
    alpha = pars[1]
    e_cutoff = (10**pars[2]) * u.TeV
    B = pars[3] * u.uG

    # Initialize instances of the particle distribution and radiative models
    ECPL = ExponentialCutoffPowerLaw(amplitude,10.*u.TeV, alpha, e_cutoff)
    IC = InverseCompton(ECPL, seed_photon_fields=['CMB'])
    SYN = Synchrotron(ECPL, B=B)

    # compute flux at the energies given in data['energy'], and convert to units
    # of flux data
    # Data contains the merged X-ray and VHE spectrum:
    # Select xray and vhe bands and only compute Sync and IC for these bands,
    # respectively
    model = np.zeros_like(data['flux'])

    xray_idx = np.where(data['energy'] < 1*u.MeV)
    model[xray_idx] = SYN.flux(data['energy'][xray_idx],
                               2.0*u.kpc).to(data['flux'].unit)

    vhe_idx = np.where(data['energy'] >= 1*u.MeV)
    model[vhe_idx] = IC.flux(data['energy'][vhe_idx],
                             2.0*u.kpc).to(data['flux'].unit)

    # An alternative, slower approach, is to compute both models for all the
    # energy range:
    # model = (IC.flux(data,distance=2.0*u.kpc).to(data['flux'].unit) +
    #          SYN.flux(data,distance=2.0*u.kpc).to(data['flux'].unit))

    # The first array returned will be compared to the observed spectrum for
    # fitting. All subsequent objects will be stores in the sampler metadata
    # blobs.
    return model, IC.compute_We(Eemin=1*u.GeV)

## Prior definition

def lnprior(pars):
	"""
	Return probability of parameter values according to prior knowledge.
	Parameter limits should be done here through uniform prior ditributions
	"""
        # Limit norm and B to be positive
	logprob = naima.uniform_prior(pars[0],0.,np.inf) \
                + naima.uniform_prior(pars[1],-1,5) \
                + naima.uniform_prior(pars[3],0,np.inf)

	return logprob

if __name__=='__main__':

## Set initial parameters and labels

    # Estimate initial magnetic field and get value in uG
    B0 = 2*naima.estimate_B(xray, vhe).to('uG').value

    p0=np.array((1e33,3.3,np.log10(48.0),B0))
    labels=['norm','index','log10(cutoff)','B']

## Run sampler

    # Simple guess does not usually work well in Sync+IC fits because of
    # degeneracy with B, set it to False (we need a good initial value for norm
    # in p0!)
    sampler,pos = naima.run_sampler(data_table=[xray,vhe], p0=p0, labels=labels,
            model=ElectronSynIC, prior=lnprior, nwalkers=32, nburn=100, nrun=20,
            threads=4, data_sed=False, guess=False, prefit=True)

## Save sampler
    from astropy.extern import six
    from six.moves import cPickle
    sampler.pool=None
    cPickle.dump(sampler,open('CrabNebula_SynIC_sampler.pickle','wb'))

## Diagnostic plots

    naima.save_diagnostic_plots('CrabNebula_SynIC', sampler, sed=True,
            blob_labels=['Spectrum', '$W_e$($E_e>1$ GeV)'])
    naima.save_results_table('CrabNebula_SynIC',sampler)


