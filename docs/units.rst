.. _units:

Units system used
=================

The package makes use of the :mod:`astropy.units` package to handle units and
unit conversions. Several of the options that need to be specified in the
functions described below require :class:`~astropy.units.quantity.Quantity`
instances. Defining quantities is straightforward::

    from astropy import units as u

    # Define scalar quantity
    q1 = 3. * u.kpc

    # Define array quantity using a list
    q2 = [1., 2., 3.] * u.arcsec

    # Define array quantity using a Numpy array
    q3 = np.array([1., 2., 3.]) * u.cm ** 2 / u.g

