0.2 (unreleased)
----------------

- Refactored sherpa models to use parent SherpaModelECPL class [#62]
- Added a data_sed flag to get_sampler to select whether to convert all data
  tables to SED or choose format of first data tables when providing multiple
  data tables.

Bug Fixes
^^^^^^^^^

- Fix sherpa models guess() for integrated datasets.
- Only complain about CL when there are ULs at a different CL.

API Changes
^^^^^^^^^^^

- module sherpamod is now sherpa_modules.


0.1 (2015-02-02)
----------------

- Initial release
