Fitting a model to a spectrum
=============================

The first step in fitting a model to an observed spectrum is to read the
spectrum into the appropriate format. See :ref:`dataformat` for an explanation
of the format and an example.

Building the model and prior functions
--------------------------------------

The model function will be the function that will be called to compare with the
observed spectrum. It must take two parameters: an array of the parameters of
the model, and the data table. 

`gammafit` includes several models in the `gammafit.models` module that make it
easier to fit common functional forms for spectra (`models.PowerLaw`,
`models.ExponentialCutoffPowerLaw`, `models.BrokenPowerLaw`, and
`models.LogParabola`), as well as several radiative models (see :ref:`radiative`
for a detailed explanation of these). Once initialized with the
relevant parameters, all model instances can be called with an energy array to
obtain the flux of the model at the values of the energy array. If they are
called with a data table as argument, the energy values from the ``energy``
column will be used.

Building the model function from one of the functional forms is easy::

    from gammafit.models import ExponentialCutoffPowerLaw
    import astropy.units as u

    def model(pars, data):
        amplitude = pars[0] * (1 / (u.cm**2 * u.s * u.TeV))
        alpha = pars[1]
        e_cutoff = (10**pars[2]) * u.TeV

        PL = ExponentialCutoff(amplitude, 1*u.TeV, alpha, e_cutoff)

        return PL(data)

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





Metadata blobs
--------------
