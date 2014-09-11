Fitting a model to a spectrum
=============================

The first step in fitting a model to an observed spectrum is to read the
spectrum into the appropriate format. See :ref:`dataformat` for an explanation
of the format and an example, and :ref:`units` for a brief explanation of the
unit system used in `gammafit`.

Building the model and prior functions
--------------------------------------

The model function is be the function that will be called to compare with the
observed spectrum. It must take two parameters: an array of the free parameters
of the model, and the data table.

`gammafit` includes several models in the `gammafit.models` module that make it
easier to fit common functional forms for spectra (`models.PowerLaw`,
`models.ExponentialCutoffPowerLaw`, `models.BrokenPowerLaw`, and
`models.LogParabola`), as well as several radiative models (see :ref:`radiative`
for a detailed explanation of these). Once initialized with the relevant
parameters, all model instances can be called with an energy array to obtain the
flux of the model at the values of the energy array. If they are called with a
data table as argument, the energy values from the ``energy`` column will be
used.

Building the model function from one of the functional forms is easy. In the
following example, the three model parameters in the ``pars`` array are the
amplitude, the spectral index, and the cutoff energy::

    from gammafit.models import ExponentialCutoffPowerLaw
    import astropy.units as u

    def model(pars, data):
        amplitude = pars[0] * (1 / (u.cm**2 * u.s * u.TeV))
        alpha = pars[1]
        e_cutoff = (10**pars[2]) * u.TeV
        e_0 = 1 * u.TeV

        ECPL = ExponentialCutoffPowerLaw(amplitude, e_0, alpha, e_cutoff)

        return ECPL(data)

In addition, we must build a function to return the prior function, i.e., a
function that encodes any previous knowledge you have about the parameters, such
as previous measurements or physically acceptable ranges. Two simple priors
functions are included with gammafit: `normal_prior` and `uniform_prior`.
`uniform_prior` can be used to set parameter limits. Following the example
above, we might want to limit the amplitude to be positive,
and the spectral index to be between 0.5 and 3.5::

    from gammafit import uniform_prior

    def prior(pars):
        lnprior = uniform_prior(pars[0], 0., np.inf) \
                + uniform_prior(pars[1], 0.5, 3.5)


Saving additional information --- Metadata blobs
------------------------------------------------

If we wish to save additional information at each of the model computations,
extra information can be returned from the model call. This extra information
(known as metadata blobs) is stored in the sampler object returned from the
fitting and can be accessed later. There are three formats for the data
stored as a metadata blob that will be understood by the plotting routines of
`gammafit`:

- A `~astropy.units.Quantity` scalar. A histogram and distribution properties
  (median, 16th and 84th percentiles, etc.) will be plotted.
- A `~astropy.units.Quantity` array with the same length as the observed
  spectrum energy array. If it has a physical type of flux or luminosity, it
  will be interpreted as a photon spectrum and plotted against the observed
  spectrum energy array.
- A pair (tuple or list) of `~astropy.units.Quantity` arrays of equal length.
  They will be plotted against each other.

When fitting a particle distribution radiative output to a spectrum, information
on the particle distribution (e.g., the actual particle distribution and the
total energy in relativistic particles) can be saved as a metadata blob.  Below
is an example that does precisely this with an Inverse Compton emission model::

    from gammafit.models import ExponentialCutoffPowerLaw, InverseCompton
    import astropy.units as u
    import numpy as np

    def model(pars, data):
        amplitude = pars[0] * (1 / u.eV)
        alpha = pars[1]
        e_cutoff = (10**pars[2]) * u.TeV
        e_0 = 10 * u.TeV

        ECPL = ExponentialCutoffPowerLaw(amplitude, e_0, alpha, e_cutoff)
        IC = InverseCompton(ECPL, seed_photon_fields=['CMB','FIR'])

        # The total enegy in electrons of model IC can be accessed through the
        # attribute We
        We = IC.We

        # We can also save the particle distribution between 100 MeV and 100 TeV

        electron_e = np.logspace(8, 14, 100) * u.eV
        electron_dist = ECPL(electron_e)

        # The first object returned must be the model photon spectrum, and
        # subsequent objects will be stored as metadata blobs

        return IC(data), (electron_e, electron_dist), We




