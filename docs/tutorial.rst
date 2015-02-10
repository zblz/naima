Tutorial: Fitting a model to a spectrum
=======================================

The first step in fitting a model to an observed spectrum is to read the
spectrum into the appropriate format. See :ref:`dataformat`  for an explanation
of the format and an example, and :ref:`units`  for a brief explanation of the
unit system used in ``naima``.

Building the model and prior functions
--------------------------------------

The model function is be the function that will be called to compare with the
observed spectrum. It must take two parameters: an array of the free parameters
of the model, and the data table.

``naima`` includes several models in the `naima.models` module that make it easier
to fit common functional forms for spectra (`~naima.models.PowerLaw`,
`~naima.models.ExponentialCutoffPowerLaw`, `~naima.models.BrokenPowerLaw`, and
`~naima.models.LogParabola`), as well as several radiative models
(`~naima.models.InverseCompton`, `~naima.models.Synchrotron`,
`~naima.models.Bremsstrahlung`, and `~naima.models.PionDecay`; see
:ref:`radiative` for a detailed explanation of these). Once initialized with the
relevant parameters, all model instances can be called with an energy array to
obtain the flux of the model at the values of the energy array. If they are
called with a data table as argument, the energy values from the ``energy``
column will be used.

The model function to be used for fitting must take two arguments: 1) an array
of the free parameters, and 2) a data object which will include the energy and
flux of the observed spectrum in ``data['energy']`` and ``data['flux']``,
respectively.

Building the model function from one of the functional forms is easy. In the
following example, the three model parameters in the ``pars`` array are the
amplitude, the spectral index, and the cutoff energy::

    from naima.models import ExponentialCutoffPowerLaw
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
functions are included with ``naima``: `~naima.normal_prior` and `~naima.uniform_prior`.
`~naima.uniform_prior` can be used to set parameter limits. Following the example
above, we might want to limit the amplitude to be positive,
and the spectral index to be between 0.5 and 3.5::

    from naima import uniform_prior

    def lnprior(pars):
        lnprior = uniform_prior(pars[0], 0., np.inf) \
                + uniform_prior(pars[1], 0.5, 3.5)
        return lnprior


.. _blobs:

Saving additional information --- Metadata blobs
------------------------------------------------

If we wish to save additional information at each of the model computations,
extra information can be returned from the model call. This extra information
(known as metadata blobs; see details in the `emcee documentation
<http://dan.iel.fm/emcee/current/user/advanced/#arbitrary-metadata-blobs>`_) is
stored in the sampler object returned from the fitting and can be accessed
later. There are three formats for the data stored as a metadata blob that will
be understood by the plotting routines of ``naima``:

- A `~astropy.units.Quantity` scalar. A histogram and distribution properties
  (median, 16th and 84th percentiles, etc.) will be plotted.
- A `~astropy.units.Quantity` array with the same length as the observed
  spectrum energy array. If it has a physical type of flux or luminosity, it
  will be interpreted as a photon spectrum and plotted against the observed
  spectrum energy array.
- A pair (tuple or list) of `~astropy.units.Quantity` arrays of equal length.
  They will be plotted against each other.

When fitting a radiative output to a spectrum, information on the particle
distribution (e.g., the actual particle distribution, or the total energy in
relativistic particles) can be saved as a metadata blob.  Below is an example
that does precisely this with an Inverse Compton emission model::

    from naima.models import ExponentialCutoffPowerLaw, InverseCompton
    import astropy.units as u
    import numpy as np

    def model_function(pars, data):
        amplitude = pars[0] * (1 / u.eV)
        alpha = pars[1]
        e_cutoff = (10**pars[2]) * u.TeV
        e_0 = 10 * u.TeV

        ECPL = ExponentialCutoffPowerLaw(amplitude, e_0, alpha, e_cutoff)
        IC = InverseCompton(ECPL, seed_photon_fields=['CMB','FIR'])

        # The total enegy in electrons of model IC can be accessed through the
        # attribute We or obtained for a given range with compute_We
        We = IC.compute_We(Eemin = 1*u.TeV)

        # We can also save the particle distribution between 100 MeV and 100 TeV
        electron_e = np.logspace(8, 14, 100) * u.eV
        electron_dist = ECPL(electron_e)

        # The first object returned must be the model photon spectrum, and
        # subsequent objects will be stored as metadata blobs
        return IC(data), (electron_e, electron_dist), We

Sampling the posterior distribution function
--------------------------------------------

Before starting the MCMC run, we must provide the procedure with initial
estimates of the parameters and their names::

    p0 = np.array((1e36, 2.3, 1.1))
    labels = ['amplitude', 'alpha', 'log10(e_cutoff)']

All the objects above can then be provided to `~naima.run_sampler`, the main
fitting function in ``naima``::

    sampler, pos = naima.run_sampler(data_table = data, p0=p0, label=labels,
                    model=model_function, prior=lnprior,
                    nwalkers=128, nburn=50, nrun=10, threads=4)

The ``nwalkers`` parameter specifies how many *walkers* will be used in the
sampling procedure, ``nburn`` specifies how many steps to be run as *burn-in*,
and ``nrun`` specifies how many steps to run after the *burn-in* and save these
samples in the sampler object. For details on these parameters, see the
`documentation of the emcee package <http://dan.iel.fm/emcee/current/>`_.


.. _plotting:

Plotting and saving the results of the run
------------------------------------------

The results stored in the sampler object can be analysed through the plotting
procedures of ``naima``: `~naima.plot_chain`, `~naima.plot_fit`, and
`~naima.plot_data`. In addition, two convenience functions can be used to
generate a collection of plots that illustrate the results and the stability of
the fitting procedure. These are `~naima.save_diagnostic_plots`::

    naima.save_diagnostic_plots('CrabNebula_naima_fit', sampler,
        blob_labels=['Spectrum', 'Electron energy distribution', '$W_e; E>1$ TeV'])

and `~naima.save_results_table`::

    naima.save_results_table('CrabNebula_naima_fit', sampler)
