0.3 (unreleased)
----------------

Bug Fixes
^^^^^^^^^

- Fixed sed conversion of residuals [#69]

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
