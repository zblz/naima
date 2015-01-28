.. _dataformat:

Data format
===========

The observed spectra to be used as constraints for the particle distribution
have to be provided to the `get_sampler` and `run_sampler` functions in the form
of an `astropy.table.Table` object. More information on creating, reading and
manipulating `~astropy.table.Table` can be found in the astropy documentation.

The table needs at least these columns, with the appropriate associated units
(with the physical type indicated in brackets below):

- ``energy``: Observed photon energy [energy]
- ``flux``: Observed fluxes [flux or differential flux]
- ``flux_error``: 68% CL gaussian uncertainty of the flux [flux or
  differential flux]. It can also be provided as ``flux_error_lo``
  and ``flux_error_hi`` (see below).

Optional columns:

- ``energy_width``: Width of the energy bin [``energy``], or
- ``energy_error``: Half-width of the energy bin [``energy``], or
- ``energy_error_lo`` and ``energy_error_hi``: Distance from bin center to lower
  and upper bin edges [``energy``], or
- ``energy_lo`` and ``energy_hi``: Energy edges of the corresponding
  energy bin [``energy``]
- ``flux_error_lo`` and ``flux_error_hi``: 68% CL gaussian lower and
  upper uncertainties of the flux.
- ``ul``: Flag to indicate that a flux measurement is an upper limit.

The ``keywords`` metadata field of the table can be used to provide the
confidence level of the upper limits with the keyword ``cl``, which defaults to
90%. The `astropy.io.ascii` reader can recover all the needed information from
ASCII tables in the :class:`~astropy.io.ascii.Ipac` and
:class:`~astropy.io.ascii.Daophot` formats, and everything except the ``cl``
keyword from tables in the :class:`~astropy.io.ascii.SExtractor`. Below you can
see an example of a file in :class:`~astropy.io.ascii.Ipac` format that includes
all the necessary fields.  This format is focused on being human readable.::


    \ Crab Nebula spectrum measured by HESS taken from table 5 of
    \ Aharonian et al. 2006, A&A 457, 899
    \ ADS bibcode: 2006A&A...457..899A
    \ 
    \cl = 0.9
    | energy | flux          | flux_error_hi | flux_error_lo | ul  |
    | float  | float         | float         | float         | int |
    | TeV    | 1/(cm2 s TeV) | 1/(cm2 s TeV) | 1/(cm2 s TeV) |     |
      0.519    1.81e-10        0.06e-10        0.06e-10        0
      0.729    7.27e-11        0.20e-11        0.19e-11        0
      1.06     3.12e-11        0.09e-11        0.09e-11        0
      1.55     1.22e-11        0.04e-11        0.04e-11        0
      2.26     4.60e-12        0.18e-12        0.18e-12        0
      3.3      1.53e-12        0.08e-12        0.08e-12        0
      4.89     6.35e-13        0.39e-13        0.38e-13        0
      7.18     2.27e-13        0.18e-13        0.17e-13        0
      10.4     6.49e-14        0.77e-14        0.72e-14        0
      14.8     1.75e-14        0.33e-14        0.30e-14        0
      20.9     7.26e-15        1.70e-15        1.50e-15        0
      30.5     9.58e-16        5.60e-16        4.25e-16        0

And the same table shown in the :class:`~astropy.io.ascii.SExtractor` format::

    # Crab Nebula spectrum measured by HESS taken from table 5 of
    # Aharonian et al. 2006, A&A 457, 899
    # ADS bibcode: 2006A&A...457..899A
    #
    # 1 energy [TeV]
    # 2 excess
    # 3 significance
    # 4 flux [cm-2 s-1 TeV-1]
    # 5 flux_error_hi [cm-2 s-1 TeV-1]
    # 6 flux_error_lo [cm-2 s-1 TeV-1]
    # 7 ul
    0.519   975.0  42.9  1.81e-10    6e-12     6e-12  0
    0.729  1580.0  56.0  7.27e-11    2e-12   1.9e-12  0
     1.06  1414.0  55.3  3.12e-11    9e-13     9e-13  0
     1.55  1082.0  47.3  1.22e-11    4e-13     4e-13  0
     2.26   762.0  39.5   4.6e-12  1.8e-13   1.8e-13  0
      3.3   443.0  29.5  1.53e-12    8e-14     8e-14  0
     4.89   311.0  24.9  6.35e-13  3.9e-14   3.8e-14  0
     7.18   186.0  19.6  2.27e-13  1.8e-14   1.7e-14  0
     10.4    86.0  13.1  6.49e-14  7.7e-15   7.2e-15  0
     14.8    36.0   8.1  1.75e-14  3.3e-15     3e-15  0
     20.9    23.0   7.5  7.26e-15  1.7e-15   1.5e-15  0
     30.5     4.0   2.9  9.58e-16  5.6e-16  4.25e-16  0


A data table to be used with naima can then be read with the
`astropy.io.ascii` reader::

    >>> from astropy.io import ascii
    >>> data_table = ascii.read('CrabNebula_HESS_2006.dat')

The table column names, types, and units, will be read automatically from the
file. If the table is in :class:`~astropy.io.ascii.SExtractor` format, the
confidence level of the upper limits can be added after reading the table as a
keyword::

    >>> data_table.meta['keywords'] = {'cl':{'value':0.9}}


A note on physical types
------------------------

Units defined through `astropy.units.Unit` have an associtaed physical type. naima defines a few additional physical types to those defined in
`astropy.units`. They are used internally to check that the inputs have the
approaprite physical type and can be converted to the appropriate units. These are:

- ``flux``: convertible to :math:`\mathrm{erg\,cm^{-2}\,s^{-1}}`
- ``differential flux``: convertible to :math:`\mathrm{1/(s\,cm^2\,eV)}`
- ``differential power``: convertible to :math:`\mathrm{1/(s\,eV)}`
- ``differential energy``: convertible to :math:`\mathrm{1/eV}`
