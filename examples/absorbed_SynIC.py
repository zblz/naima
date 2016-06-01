#!/usr/bin/env python
import numpy as np
import astropy.units as u
from astropy.io import ascii

import naima

## Model definition

from naima.models import InverseCompton, Synchrotron, ExponentialCutoffPowerLaw, BrokenPowerLaw, AbsorptionModel

# Define the redshift of the source, and absorption model
redshift = 0.06
EBL_transmitance = AbsorptionModel(redshift, 'Dominguez')

def ElectronSynIC(pars, data):

    # Match parameters to ECPL properties, and give them the appropriate units
    amplitude = 10**pars[0] / u.eV
    e_break = (10**pars[1]) * u.TeV
    alpha1 = pars[2]
    alpha2 = pars[3]
    B = pars[4] * u.uG

    # Initialize instances of the particle distribution and radiative models
    BPL = BrokenPowerLaw(amplitude, 1. * u.TeV, e_break, alpha1, alpha2)
    
    # Compute IC on CMB and on a FIR component
    IC = InverseCompton(
        BPL,
        seed_photon_fields=['CMB'],
        Eemin=10 * u.GeV)
    SYN = Synchrotron(BPL, B=B)

    # compute flux at the energies given in data['energy']
    model = (EBL_transmitance.transmission(data) * IC.flux(data, distance=1.0 * u.kpc) + SYN.flux(data,
                                                      distance=1.0 * u.kpc))

    # The first array returned will be compared to the observed spectrum for
    # fitting. All subsequent objects will be stored in the sampler metadata
    # blobs.
    return model, IC.compute_We(Eemin=1 * u.TeV)



if __name__ == '__main__':

    # Some random values for a "beautiful double peak structure
    p0 = np.array((33., 1., 0.35, 3., 5))

    labels = ['log10(norm)', 'log10(Energy_Break)', 'index1', 'index2', 'B', 'redshift']

    # Run interactive fitter, to show the very high energy absorption 
    imf = naima.InteractiveModelFitter(ElectronSynIC, p0, labels=labels, e_range=[1e-09*u.GeV, 1e05*u.GeV])


