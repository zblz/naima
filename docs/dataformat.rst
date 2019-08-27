.. _dataformat:

Data format
===========

The observed spectra to be used as constraints for the particle distribution
have to be provided to the `get_sampler` and `run_sampler` functions in the form
of an `astropy.table.Table` object. More information on creating, reading and
manipulating `~astropy.table.Table` can be found in the `astropy`_ documentation.

The table needs at least these columns, with the appropriate associated units
(with the physical type indicated in brackets below):

- ``energy``: Observed photon energy [``energy``]
- ``flux``: Observed fluxes [``flux`` or ``differential flux``]
- ``flux_error``: 68% CL gaussian uncertainty of the flux [``flux`` or
  ``differential flux``]. It can also be provided as ``flux_error_lo``
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
- ``ul``: Flag to indicate that a flux measurement is an upper limit. The flux
  error values for this measurement will be disregarded.
- ``flux_ul``: Upper limit to the flux. If not present, the ``flux``
  column will be taken as an upper limit for those measurements with the
  ``ul`` flag.

The ``keywords`` metadata field of the table can be used to provide the
confidence level of the upper limits with the keyword ``cl``, which defaults to
90%. The `astropy.io.ascii` reader can recover all the needed information from
ASCII tables in the :class:`~astropy.io.ascii.Ecsv`,
:class:`~astropy.io.ascii.Ipac`, and :class:`~astropy.io.ascii.Daophot` formats,
and everything except the ``cl`` keyword from tables in the
:class:`~astropy.io.ascii.SExtractor` format.  A data table to be used with
naima can then be read with the `astropy.io.ascii` reader::

    >>> from astropy.io import ascii
    >>> data_table = ascii.read('RXJ1713_HESS_2007.dat')

The table column names, types, and units, will be read automatically from the
file.

Multiple data tables
--------------------

Multiple data tables can be provided to `get_sampler` and `run_sampler` as a
list. Each of them has to fulfill the requirements above, but they don't have to
be in the same format, as `naima` will concatenate them as appropriate. If some
of the tables are in differential flux and some others in energy flux, they will
all be converted to the format of the first table in the list. However, this can
be controlled with the ``data_sed`` argument of `get_sampler`, which will
control whether all data tables are converted to an SED (``data_sed=True``) or
to differential fluxes (``data_sed=False``).

Data table examples
-------------------

Ipac
++++

Below you can see an example of a file in :class:`~astropy.io.ascii.Ipac` format
that includes all the necessary fields.  This format is focused on being human
readable. Everything starting with a slash and a space is a comment, and
keywords are given after a slash without a space.

.. literalinclude:: ../examples/RXJ1713_HESS_2007.dat

Ecsv
++++

The same table shown in :class:`~astropy.io.ascii.Ecsv` format.

.. literalinclude:: ../examples/RXJ1713_HESS_2007.ecsv

SExtractor
++++++++++

And the same table shown in the :class:`~astropy.io.ascii.SExtractor` format

.. literalinclude:: ../examples/RXJ1713_HESS_2007_sextractor.dat

If the table is in :class:`~astropy.io.ascii.SExtractor` format, the
confidence level of the upper limits can be added after reading the table as a
keyword::

    >>> data_table.meta['keywords'] = {'cl':{'value':0.95}}
