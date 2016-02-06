import numpy as np
import astropy.units as u
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

PionDecay_ECPL_p0 = np.array((46, 2.34, np.log10(80.)))
PionDecay_ECPL_labels = ['log10(norm)', 'index', 'log10(cutoff)']

# Prepare an energy array for saving the particle distribution
proton_energy = np.logspace(-3, 2, 50) * u.TeV


def PionDecay_ECPL(pars, data):
    amplitude = 10**pars[0] / u.TeV
    alpha = pars[1]
    e_cutoff = 10**pars[2] * u.TeV

    ECPL = naima.models.ExponentialCutoffPowerLaw(amplitude, 30 * u.TeV, alpha,
                                                  e_cutoff)
    PP = naima.models.PionDecay(ECPL, nh=1.0 * u.cm** -3)

    model = PP.flux(data, distance=1.0 * u.kpc)
    # Save a realization of the particle distribution to the metadata blob
    proton_dist = PP.particle_distribution(proton_energy)
    # Compute the total energy in protons above 1 TeV for this realization
    Wp = PP.compute_Wp(Epmin=1 * u.TeV)

    # Return the model, proton distribution and energy in protons to be stored
    # in metadata blobs
    return model, (proton_energy, proton_dist), Wp


def PionDecay_ECPL_lnprior(pars):
    logprob = naima.uniform_prior(pars[1], -1, 5)
    return logprob

# Inverse Compton with the energy in electrons as the normalization parameter
# ===========================================================================

IC_We_p0 = np.array((40, 3.0, np.log10(30)))
IC_We_labels = ['log10(We)', 'index', 'log10(cutoff)']


def IC_We(pars, data):
    # Example of a model that is normalized though the total energy in electrons

    # Match parameters to ECPL properties, and give them the appropriate units
    We = 10**pars[0] * u.erg
    alpha = pars[1]
    e_cutoff = 10**pars[2] * u.TeV

    # Initialize instances of the particle distribution and radiative model
    # set a bogus normalization that will be changed in third line
    ECPL = naima.models.ExponentialCutoffPowerLaw(1 / u.eV, 10. * u.TeV,
                                                  alpha, e_cutoff)
    IC = naima.models.InverseCompton(ECPL, seed_photon_fields=['CMB'])
    IC.set_We(We, Eemin=1 * u.TeV)

    # compute flux at the energies given in data['energy']
    model = IC.flux(data, distance=1.0 * u.kpc)

    # Save this realization of the particle distribution function
    elec_energy = np.logspace(11, 15, 100) * u.eV
    nelec = ECPL(elec_energy)

    return model, (elec_energy, nelec)


def IC_We_lnprior(pars):
    logprob = naima.uniform_prior(pars[1], -1, 5)
    return logprob

#
# FUNCTIONAL MODELS
#
# Exponential cutoff powerlaw
# ===========================

ECPL_p0 = np.array((1e-12, 2.4, np.log10(15.0)))
ECPL_labels = ['norm', 'index', 'log10(cutoff)']


def ECPL(pars, data):
    # Get the units of the flux data and match them in the model amplitude
    amplitude = pars[0] * data['flux'].unit
    alpha = pars[1]
    e_cutoff = (10**pars[2]) * u.TeV
    ECPL = naima.models.ExponentialCutoffPowerLaw(amplitude, 1 * u.TeV, alpha,
                                                  e_cutoff)

    return ECPL(data)


def ECPL_lnprior(pars):
    logprob = naima.uniform_prior(pars[0], 0., np.inf) \
            + naima.uniform_prior(pars[1], -1, 5)
    return logprob

# Log-Parabola or Curved Powerlaw
# ===============================

LP_p0 = np.array((1.5e-12, 2.7, 0.12,))
LP_labels = ['norm', 'alpha', 'beta']


def LP(pars, data):
    amplitude = pars[0] * data['flux'].unit
    alpha = pars[1]
    beta = pars[2]
    LP = naima.models.LogParabola(amplitude, 1 * u.TeV, alpha, beta)
    return LP(data)


def LP_lnprior(pars):
    logprob = naima.uniform_prior(pars[0], 0., np.inf) \
                + naima.uniform_prior(pars[1], -1, 5)
    return logprob
