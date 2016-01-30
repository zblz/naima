.. _units:

Units system
============

The package makes use of the :mod:`astropy.units` package to handle units and
unit conversions. Several of the arguments of the functions and classes of
Naima require :class:`~astropy.units.quantity.Quantity` instances. Defining
quantities is straightforward::

    from astropy import units as u

    # Define scalar quantity
    q1 = 3. * u.kpc

    # Define array quantity using a list
    q2 = [1., 2., 3.] * u.arcsec

    # Define array quantity using a Numpy array
    q3 = np.array([1., 2., 3.]) * u.cm ** 2 / u.g


A note on physical types
------------------------

Units defined through `astropy.units.Unit` have an associated physical type.
Naima defines a few additional physical types to those defined in
`astropy.units`. They are used internally to check that the inputs have the
correct physical type and can be converted to the appropriate units. These are:

- ``flux``: convertible to :math:`\mathrm{erg\,cm^{-2}\,s^{-1}}`
- ``differential flux``: convertible to :math:`\mathrm{1/(s\,cm^2\,eV)}`
- ``differential power``: convertible to :math:`\mathrm{1/(s\,eV)}`
- ``differential energy``: convertible to :math:`\mathrm{1/eV}`
