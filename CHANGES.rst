Changelog
---------

0.11.0 (unreleased)
^^^^^^^^^^^^^^^^^^^

- ``run_sampler`` can now continue a sampler that has already been run,
  appending the new steps to its chain (a sampler fresh from ``get_sampler``
  still has its burn-in discarded), and ``n_run`` is set to the number of steps
  in the chain. The process pool created by naima is recreated when needed and
  is closed also if the run fails or is interrupted. ``get_sampler`` and
  ``run_sampler`` have a new ``pool`` argument to use a pool provided by the
  user, which is never closed by naima.
- Fix the initial walkers of ``get_sampler`` for parameters that start at zero,
  which got no spread because it is relative to the value. Parameters whose
  label starts with ``log`` now use an absolute spread of 1 dex (0.05 dex
  when ``p0`` is already the maximum likelihood, in the same proportion as the
  relative spread of the other parameters), since a relative spread of a
  logarithmic value depends on the units and vanishes close to zero. Walkers
  that start in a region forbidden by the prior are drawn again in a shrinking
  ball, so that they remain distinct, and a warning is logged if ``p0`` is
  outside the prior.
- ``run_sampler`` can now continue a sampler that has already been run,
  appending the new steps to its chain (a sampler fresh from ``get_sampler``
  still has its burn-in discarded), and ``n_run`` is set to the number of steps
  in the chain. The process pool created by naima is recreated when needed and
  is closed also if the run fails or is interrupted. ``get_sampler`` and
  ``run_sampler`` have a new ``pool`` argument to use a pool provided by the
  user, which is never closed by naima.
- Fix ``InverseCompton.flux`` and ``InverseCompton.sed`` with the ``seed``
  argument returning the contribution computed for the energies of a previous
  call. A ``seed`` equal to the number of seed photon fields, or negative, now
  raises a ``ValueError`` instead of an ``IndexError`` or selecting the wrong
  field, and numpy integers are accepted.
- ``run_sampler`` can now continue a sampler that has already been run,
  appending the new steps to its chain (a sampler fresh from ``get_sampler``
  still has its burn-in discarded), and ``n_run`` is set to the number of steps
  in the chain. The process pool created by naima is recreated when needed and
  is closed also if the run fails or is interrupted. ``get_sampler`` and
  ``run_sampler`` have a new ``pool`` argument to use a pool provided by the
  user, which is never closed by naima.
- Remove the caching of results in the radiative models and particle
  distributions. The cache key ignored particle distributions without
  ``param_names`` (plain functions and ``TableModel``), so changing them
  returned the results for the previous parameters, and ``PionDecayKelner06``
  shared its cache between instances. The cache did not help when fitting, as
  the models are created again for each evaluation. The ``_memoize`` attribute
  no longer has any effect.
- ``run_sampler`` can now continue a sampler that has already been run,
  appending the new steps to its chain (a sampler fresh from ``get_sampler``
  still has its burn-in discarded), and ``n_run`` is set to the number of steps
  in the chain. The process pool created by naima is recreated when needed and
  is closed also if the run fails or is interrupted. ``get_sampler`` and
  ``run_sampler`` have a new ``pool`` argument to use a pool provided by the
  user, which is never closed by naima.
- Fix ``EblAbsorptionModel``. Calling the instance returned ``10**tau`` instead
  of the absorption factor ``exp(-tau)`` described in its docstring, which
  amplified the flux when multiplied to a model. ``transmission`` returned the
  same factor, but a constant 0.023 above 100 TeV where the optical depth is
  above 200 and the factor is 0. It now returns 0 there, 1 for redshifts below
  0.01 at any energy, and accepts scalar energies. The optical depth table is
  now read from disk once instead of for every instance.

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
