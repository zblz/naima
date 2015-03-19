0.4 (unreleased)
----------------

- All models have a cache of length 10 so that the output is not computed when
  the parameters have not changed. It can be turned off through the ``_memoize``
  attribute of the models.
- Fixed concatenation of UL and CL in ``validate_data_table``.

0.3 (2015-02-19)
----------------

- Added an option to save the distribution properties of scalar blobs when
  saving results table: option ``include_blobs`` of ``save_results_table``.
- A new method for radiative classes allows to renormalize the particle
  distributions to a given energy content in particles. See ``set_We`` and
  ``set_Wp`` in leptonic and hadronic classes, respectively.
- The default FIR and NIR photon fields for `naima.InverseCompton` have been set
  to the GALPROP values at a galactic radius of 6.5 kpc.
- Require astropy >= 1.0.

Bug Fixes
^^^^^^^^^

- Fixed sed conversion of residuals [#69]
- Fixed ``plot_data`` so it can take lists of data_tables.

API Changes
^^^^^^^^^^^

- The name of the ``table_format`` parameter of ``save_results_table`` has been changed
  to ``format`` for consistency with `astropy.io.ascii`.

0.2 (2015-02-10)
----------------

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
^^^^^^^^^

- Fix sherpa models guess() for integrated datasets.
- Only complain about CL when there are ULs at a different CL.
- Fix parsing of string upper limit columns in Python 3.
- Use old energy unit when plotting a new data set onto a figure [#64]
- Show ordinate units when plotting blobs without spectral data.

API Changes
^^^^^^^^^^^

- module sherpamod is now sherpa_modules.


0.1 (2015-02-02)
----------------

- Initial release
