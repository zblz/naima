import numpy as np
import naima

################################################################################
#
# This file shows a few example model functions (with associated priors, labels
# and p0 vector), that can be used as input for naima.run_sampler
#
################################################################################


#
# RADIATIVE MODELS
#

# Pion decay
# ==========

PionDecay_ECPL_p0=np.array((1e46,2.34,np.log10(80.),))
PionDecay_ECPL_labels=['norm','index','log10(cutoff)']

# Prepare an energy array for saving the particle distribution
proton_energy = np.logspace(-3,2,50)*u.TeV

def PionDecay_ECPL(pars,data):
    amplitude = pars[0] / u.TeV
    alpha = pars[1]
    e_cutoff = (10**pars[2])*u.TeV

    ECPL = ExponentialCutoffPowerLaw(amplitude, 30*u.TeV, alpha, e_cutoff)
    PP = PionDecay(ECPL)

    # convert to same units as observed differential spectrum
    model = PP.flux(data,distance=2.0*u.kpc).to(data['flux'].unit)

    # Save a realization of the particle distribution to the metadata blob
    proton_dist = PP.particle_distribution(proton_energy)

    return model, (proton_energy, proton_dist), PP.compute_Wp(Epmin=1*u.TeV)

## Prior definition

def PionDecay_ECPL_lnprior(pars):
    logprob = naima.uniform_prior(pars[0],0.,np.inf) \
                + naima.uniform_prior(pars[1],-1,5)
    return logprob

#
# FUNCTIONAL MODELS
#

# Exponential cutoff powerlaw
# ===========================


## Set initial parameters

ECPL_p0=np.array((1e-12,2.4,np.log10(15.0),))
ECPL_labels=['norm','index','log10(cutoff)']

# Get the units of the flux data and match them in the model amplitude
flux_unit = data['flux'].unit

def ECPL(pars,data):
    amplitude = pars[0] * flux_unit
    alpha = pars[1]
    e_cutoff = (10**pars[2])*u.TeV
    ECPL = ExponentialCutoffPowerLaw(amplitude, 1*u.TeV, alpha, e_cutoff)

    return ECPL(data)

## Prior definition

def ECPL_lnprior(pars):
    logprob = naima.uniform_prior(pars[0],0.,np.inf) \
            + naima.uniform_prior(pars[1],-1,5)
    return logprob
