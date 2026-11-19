Changelog
---------

0.11.0 (2026-11-19)
^^^^^^^^^^^^^^^^^^^

API changes
~~~~~~~~~~~

These changes can alter the results of existing code without raising an error.
See the entries below for details.

- Calling ``run_sampler`` with a sampler that has already been run with
  ``run_sampler`` now continues it and appends the new steps to its chain,
  instead of resetting the chain and starting over.
- Calling an ``EblAbsorptionModel`` instance now returns the absorption factor
  ``exp(-tau)`` instead of ``10**tau``. Code that worked around the old value
  must be updated. ``transmission`` is 0 above 100 TeV instead of 0.023.
- The initial walkers of logarithmic parameters (labels starting with ``log10``
  or ``log``) are spread by 1 dex (0.05 dex at the maximum likelihood) instead
  of a fraction of their value, which changes the starting point of these fits.
- Models that return NaN or raise an ``ArithmeticError`` or ``ValueError`` for
  a parameter vector give it a probability of zero instead of stopping the
  run.
- ``save_results_table`` computes ``MaxLogLikelihood`` and ``BIC`` without the
  prior and without counting upper limits as data points.
- An unknown seed photon field name in ``InverseCompton`` raises a
  ``ValueError`` instead of a ``TypeError``.
- The ``naima.model_utils`` module and the ``_memoize`` attribute of the models
  have been removed.

Bug fixes and other changes
~~~~~~~~~~~~~~~~~~~~~~~~~~~

- ``run_sampler`` can now continue a sampler that has already been run,
  appending the new steps to its chain (a sampler fresh from ``get_sampler``
  still has its burn-in discarded), and ``n_run`` is set to the number of steps
  in the chain. The process pool created by naima is recreated when needed and
  is closed also if the run fails or is interrupted. ``get_sampler`` has a
  new ``pool`` argument to use a pool provided by the user, which is never
  closed by naima (``run_sampler`` passes it on when it creates the sampler).
- Fix the initial walkers of ``get_sampler`` for parameters that start at zero,
  which got no spread because it is relative to the value. Parameters whose
  label starts with ``log`` now use an absolute spread of 1 dex (0.05 dex
  when ``p0`` is already the maximum likelihood, in the same proportion as the
  relative spread of the other parameters), since a relative spread of a
  logarithmic value depends on the units and vanishes close to zero. Walkers
  that start in a region forbidden by the prior are drawn again in a shrinking
  ball, so that they remain distinct, and a warning is logged if ``p0`` is
  outside the prior.
- Fix ``InverseCompton.flux`` and ``InverseCompton.sed`` with the ``seed``
  argument returning the contribution computed for the energies of a previous
  call. A ``seed`` equal to the number of seed photon fields, or negative, now
  raises a ``ValueError`` instead of an ``IndexError`` or selecting the wrong
  field, and numpy integers are accepted.
- Remove the caching of results in the radiative models and particle
  distributions. The cache key ignored particle distributions without
  ``param_names`` (plain functions and ``TableModel``), so changing them
  returned the results for the previous parameters, and ``PionDecayKelner06``
  shared its cache between instances. The cache did not help when fitting, as
  the models are created again for each evaluation. The ``_memoize`` attribute
  no longer has any effect, and the ``naima.model_utils`` module, which only
  contained the ``memoize`` decorator, has been removed.
- Fix ``EblAbsorptionModel``. Calling the instance returned ``10**tau`` instead
  of the absorption factor ``exp(-tau)`` described in its docstring, which
  amplified the flux when multiplied to a model. The code read the optical
  depth table as if it contained ``log10(tau)``, when it contains ``tau``.
  ``transmission`` returned the correct factor, but a constant 0.023 above
  100 TeV where the factor is below 3e-9. It now returns 0 there, 1 for
  redshifts below 0.01 at any energy, and accepts scalar energies. The optical
  depth is limited to 700, the largest value for which ``exp(-tau)`` does not
  underflow, and the table is now read from disk once instead of for every
  instance.
- ``get_sampler`` no longer modifies the ``p0`` array and ``labels`` list given
  by the caller (the normalization guess was applied in place and short
  ``labels`` were extended in place). The normalization guess is skipped, with
  a warning, when the ratio between the data and model fluxes is not finite and
  positive, instead of setting ``p0`` to ``inf``.
- Models that return NaN, or raise an arithmetic error or ``ValueError`` (e.g.
  for parameters outside of the domain of a radiative model), now give a
  probability of zero for that parameter vector instead of stopping the run
  with emcee's "Probability function returned NaN". The first failure in each
  process is logged as a warning, and a warning is logged if no walker has a
  finite probability. This also applies to the maximum likelihood prefit of
  ``get_sampler``. Other exceptions are still raised. The blobs saved for those
  parameter vectors are NaN, and the plots ignore them. ``lnprob`` has a new
  ``fallback_blob`` argument for this. Note that ``ValueError`` also covers
  errors that are bugs in the model, such as an ``UnitConversionError``: they
  are only raised if they happen for the initial ``p0``.
- ``save_results_table`` reported the parameters with the highest posterior
  probability, including the prior, as ``ML_pars`` and their log-probability as
  ``MaxLogLikelihood``, which was also used for the BIC. The metadata now has
  ``MAP_pars`` and ``MaxLogProb`` for these, and ``MaxLogLikelihood`` and
  ``BIC`` are computed from the log-likelihood without the prior at that point.
  ``ML_pars`` is kept as an alias of ``MAP_pars``. Upper limits are no longer
  counted as data points in the BIC, and the number of data points is saved as
  ``n_data``. If the sampler has no model function, the log-probability is used
  with a warning.
- The process pool created by ``get_sampler`` is terminated when the sampler is
  garbage collected or the interpreter exits, instead of staying open if
  ``run_sampler`` is not used. The warning about the first failure of the model
  is shown again in every new fit of the same session, and ``labels`` that start
  with ``log`` are only taken as logarithmic parameters for the initial walkers
  if they start with ``log10`` or what follows ``log`` is not a letter or digit
  (``log10(cutoff)``, ``log10cutoff`` and ``log_cutoff``, but not ``logistic``
  or ``log2``).
- ``flux`` and ``sed`` of the radiative models (also with ``seed`` for
  ``InverseCompton``) accept scalar energies and return scalars. ``InverseCompton``
  accepts tuples both for ``seed_photon_fields`` and for the definition of each
  field (e.g. ``("star", 5000 * u.K, 1 * u.eV / u.cm**3)``), raises a ``ValueError`` with a
  message for an unknown seed photon field name instead of a bare ``TypeError``,
  and ``seed=True`` is no longer taken as the index 1. ``InverseCompton.sed``
  with ``seed`` only computes the requested field instead of all of them.
- ``trapz_loglog`` integrates each bin from the logarithms of its values, so
  that bins where the ratio of the ``y`` values or the product ``y * x``
  overflows no longer give ``0`` or ``nan``, and the integral is precise and
  continuous for a local power-law index close to -1 and for very narrow bins.
- Fix docstrings: the default ``nEed`` of ``InverseCompton`` is 100, the
  ``FIR`` and ``NIR`` seed photon fields were swapped, ``Epmax`` of ``PionDecay``
  is a maximum energy, and a table without an energy column mentions the
  ``energy`` column in the error. The ``run_sampler`` docstring describes
  continued runs and the ``**kwargs``, and ``get_sampler`` documents the
  ``log`` labels.

0.10.5 (2026-11-14)
^^^^^^^^^^^^^^^^^^^

- Fix the upper-limit likelihood in ``lnprobmodel`` indexing the per-row
  confidence level with the number of violated upper limits instead of a mask.
  This applied the wrong confidence level when ``cl`` varies between rows, and
  raised an ``IndexError`` when every upper limit was violated. See #261.
- The radiative model constructors (``Synchrotron``, ``InverseCompton``,
  ``Bremsstrahlung``, ``PionDecay`` and ``PionDecayKelner06``) now raise a
  ``TypeError`` for unknown keyword arguments instead of silently ignoring
  them, e.g. a misspelled ``Eemin``. See #262.
- Fix ``trapz_loglog`` overflowing to ``inf`` in bins with a very steep local
  power-law index, which made the inverse Compton flux jump to ``inf`` for some
  parameter and energy combinations. Fixes #263.

0.10.4 (2026-09-05)
^^^^^^^^^^^^^^^^^^^

- Fix incorrect log-prior formulas in ``normal_prior`` and ``log_uniform_prior``.
  Fixes #253.
- Fix a mutable default argument bug in ``plot_data`` and ``plot_fit`` that could
  leak upper-limit styling between calls.
- Add a ``label`` keyword to ``plot_data`` for setting a legend label on the
  plotted flux points. Fixes #237.
- ``plot_fit`` now recomputes the model on a fine energy grid by default
  instead of reusing the (possibly sparse) values stored in the sampler
  blobs, fixing inconsistent plots for models such as synchrotron
  self-Compton. Fixes #240.
- ``save_results_table`` now records the physical unit of each parameter in a
  new ``unit`` column. Fixes #243.
- Dropped support for Python 3.10, following astropy.

0.10.3 (2025-08-05)
^^^^^^^^^^^^^^^^^^^

- Add overwrite keyword to ``save_results_table`` - @HealthyPear

0.10.2 (2025-01-17)
^^^^^^^^^^^^^^^^^^^

- Fix bug when Eemin and Eemax where set too close. Fixes #225.

0.10.1 (2025-01-17)
^^^^^^^^^^^^^^^^^^^

- Dropped support for Python 3.9, updated to numpy v2, updated packaging.

0.10.0 (2021-11-02)
^^^^^^^^^^^^^^^^^^^

- Updated emcee dependency to version 3. The interface to access the chain,
  log-probability, and blobs has changed accordingly, see `emcee's documentation
  <https://emcee.readthedocs.io>`_.

0.9.1 (2020-01-31)
^^^^^^^^^^^^^^^^^^

- Fix exceptions thrown by `np.array_equal` for some combinations of model and
  blob outputs.

0.9.0 (2019-11-08)
^^^^^^^^^^^^^^^^^^

- Dropped Python 2 support. If you need to use naima in Python 2, please use
  version 0.8.4.

0.8.4 (2019-08-27)
^^^^^^^^^^^^^^^^^^

- Updated deprecated uses of numpy and astropy.Quantity logic
- Updated tests for latest versions of dependencies

0.8.3 (2018-11-27)
^^^^^^^^^^^^^^^^^^

Bug fixes
~~~~~~~~~

- Fixed plotting scalar blobs with units.
- Fixed plotting vector blobs with same length as data but incompatible units.

0.8.2 (2018-11-26)
^^^^^^^^^^^^^^^^^^

- Formatted source code with black.

Bug fixes
~~~~~~~~~

- Fixed deprecated use of np.all and normed argument to matplotlib's hist.

0.8.1 (2017-09-27)
^^^^^^^^^^^^^^^^^^

Bug fixes
~~~~~~~~~

- Fixed deprecated negative signs on numpy booleans.
- Fixed wrong sign in delta functional approximation of Kelner PionDecay.

0.8 (2016-12-21)
^^^^^^^^^^^^^^^^

- Added a `threads` parameter to `plot_fit` and `plot_samples` that set the
  number of cores to use in computing model samples for plotting.
- Added a new model for EBL absorption based on the tables of Dominguez et al.
  2011.

Bug fixes
~~~~~~~~~
- Updated to use new ``emcee`` autocorrelation API in version 2.2.
- Fixed sherpa models string representation.

0.7.1 (2016-02-04)
^^^^^^^^^^^^^^^^^^

- Packaging bugfix

0.7 (2016-02-04)
^^^^^^^^^^^^^^^^

- The ``InverseCompton`` class can now compute IC on arbitrary seed photon
  fields passed as arrays.
- ``plot_fit`` and ``plot_data`` have new options (``errorbar_opts`` and
  ``ulim_opts``) to control the properties of spectral flux points and
  upper-limits.
- There is a new table model class in ``naima.models.TableModel``.
- Added ``corner`` (former ``triangle_plot``) as a dependency.

0.6.1 (2015-10-29)
^^^^^^^^^^^^^^^^^^

- Performance improvements to the memoize logic that result in 10% or higher
  improvement in model execution time.

0.6 (2015-09-10)
^^^^^^^^^^^^^^^^

- Medians and associated errors are now shown with a precision corresponding to
  a single significant digit in the errors (except when the leading digit is 1,
  when two significant digits are shown). Note that they are still saved with
  full precision to the result tables.
- There is a new GUI tool for interactive model fitting:
  ``InteractiveModelFitter``, which can be accessed directly or through the
  ``interactive`` argument of ``get_sampler`` prior to a sampling run.
- Sampling run results can be saved and retrieved for later analysis or archival
  with the new functions `naima.save_run` and `naima.read_run`.
- The individual contributions to the total Inverse Compton spectrum of the
  different seed photon fields can now be accessed through the ``seed`` argument
  of the ``InverseCompton.flux`` and ``InverseCompton.sed`` functions.

Bug Fixes
~~~~~~~~~

- ``save_diagnostic_plots`` now turns matplotlib interactivity off, so the plots
  are only saved and not shown.
- The ``group`` column is now preserved if a data table is validated more than
  once.

0.5 (2015-08-05)
^^^^^^^^^^^^^^^^

- ``save_results_table`` now saves the maximum log likelihood and ML parameters
  of the sample.
- Update ``astropy_helpers`` to 1.0.2 and require astropy >= 1.0.2 to ensure
  pickleable Tables.
- Internal data is a QTable instead of a dict.
- When multiple input spectra are used, they are now plotted with different
  colors and markers.
- Now doing the prefit with a minimizer that allows for relative tolerance
  termination: in general prefit will be faster.
- Add ``e_range`` and ``e_npoints`` parameters to ``plot_fit`` to allow
  computing the model samples for a wider energy range that the observed
  spectrum (or at energies between data sets, such as X-ray and gamma-ray)
- Added ``plot_corner`` as a thin wrapper around ``triangle.corner`` with ML
  parameter plotting.

0.4 (2015-03-19)
^^^^^^^^^^^^^^^^

- All models have a cache of length 10 so that the output is not computed when
  the parameters have not changed. It can be turned off through the ``_memoize``
  attribute of the models.

Bug Fixes
~~~~~~~~~

- Fixed concatenation of UL and CL in ``validate_data_table``.

0.3 (2015-02-19)
^^^^^^^^^^^^^^^^

- Added an option to save the distribution properties of scalar blobs when
  saving results table: option ``include_blobs`` of ``save_results_table``.
- A new method for radiative classes allows to renormalize the particle
  distributions to a given energy content in particles. See ``set_We`` and
  ``set_Wp`` in leptonic and hadronic classes, respectively.
- The default FIR and NIR photon fields for `naima.InverseCompton` have been set
  to the GALPROP values at a galactic radius of 6.5 kpc.
- Require astropy >= 1.0.

Bug Fixes
~~~~~~~~~

- Fixed sed conversion of residuals [#69]
- Fixed ``plot_data`` so it can take lists of data_tables.

API Changes
~~~~~~~~~~~

- The name of the ``table_format`` parameter of ``save_results_table`` has been changed
  to ``format`` for consistency with `astropy.io.ascii`.

0.2 (2015-02-10)
^^^^^^^^^^^^^^^^

- Refactored sherpa models to use parent SherpaModelECPL class [#62]
- Added a data_sed flag to get_sampler to select whether to convert all data
  tables to SED or choose format of first data tables when providing multiple
  data tables.
- Added support for  a ``flux_ul`` column in input data tables.
- Added a method to estimate magnetic field: `naima.estimate_B`.
- Added the option to perform an optimization of the parameters before the MCMC
  run: see option ``prefit`` in `naima.get_sampler`.
- Convert between SED and differential fluxes automatically if the model and
  data physical types do not match.
- Add blob_labels parameter to save_diagnostic_plots.

Bug Fixes
~~~~~~~~~

- Fix sherpa models guess() for integrated datasets.
- Only complain about CL when there are ULs at a different CL.
- Fix parsing of string upper limit columns in Python 3.
- Use old energy unit when plotting a new data set onto a figure [#64]
- Show ordinate units when plotting blobs without spectral data.

API Changes
~~~~~~~~~~~

- module sherpamod is now sherpa_modules.

0.1 (2015-02-02)
^^^^^^^^^^^^^^^^

- Initial release
