# -*- coding: utf-8 -*-
# Licensed under a 3-clause BSD style license - see LICENSE.rst
import functools
import os

import astropy.units as u
import numpy as np
from astropy.table import Table
from astropy.utils.data import get_pkg_data_filename

from .extern.validator import (
    validate_array,
    validate_physical_type,
    validate_scalar,
)
from .radiative import Bremsstrahlung, InverseCompton, PionDecay, Synchrotron

__all__ = [
    "Synchrotron",
    "InverseCompton",
    "PionDecay",
    "Bremsstrahlung",
    "BrokenPowerLaw",
    "ExponentialCutoffPowerLaw",
    "PowerLaw",
    "LogParabola",
    "ExponentialCutoffBrokenPowerLaw",
    "TableModel",
    "EblAbsorptionModel",
]


def _validate_ene(ene):
    if isinstance(ene, dict) or isinstance(ene, Table):
        try:
            ene = validate_array(
                "energy", u.Quantity(ene["energy"]), physical_type="energy"
            )
        except KeyError:
            raise TypeError("Table or dict does not have 'ene' column")
    else:
        if not isinstance(ene, u.Quantity):
            ene = u.Quantity(ene)
        validate_physical_type("energy", ene, physical_type="energy")

    return ene


class PowerLaw:
    """
    One dimensional power law model.

    Parameters
    ----------
    amplitude : float
        Model amplitude.
    e_0 : `~astropy.units.Quantity` float
        Reference energy
    alpha : float
        Power law index

    See Also
    --------
    PowerLaw, BrokenPowerLaw, LogParabola

    Notes
    -----
    Model formula (with :math:`A` for ``amplitude``, :math:`\\alpha` for
    ``alpha``):

        .. math:: f(E) = A (E / E_0) ^ {-\\alpha}

    """

    param_names = ["amplitude", "e_0", "alpha"]

    def __init__(self, amplitude, e_0, alpha):
        self.amplitude = amplitude
        self.e_0 = validate_scalar(
            "e_0", e_0, domain="positive", physical_type="energy"
        )
        self.alpha = alpha

    @staticmethod
    def eval(e, amplitude, e_0, alpha):
        """One dimensional power law model function"""

        xx = e / e_0
        return amplitude * xx ** (-alpha)

    def _calc(self, e):
        return self.eval(
            e.to("eV").value,
            self.amplitude,
            self.e_0.to("eV").value,
            self.alpha,
        )

    def __call__(self, e):
        """One dimensional power law model function"""
        e = _validate_ene(e)
        return self._calc(e)


class ExponentialCutoffPowerLaw:
    """
    One dimensional power law model with an exponential cutoff.

    Parameters
    ----------
    amplitude : float
        Model amplitude
    e_0 : `~astropy.units.Quantity` float
        Reference point
    alpha : float
        Power law index
    e_cutoff : `~astropy.units.Quantity` float
        Cutoff point
    beta : float
        Cutoff exponent

    See Also
    --------
    PowerLaw, BrokenPowerLaw, LogParabola

    Notes
    -----
    Model formula (with :math:`A` for ``amplitude``, :math:`\\alpha` for
    ``alpha``, and :math:`\\beta` for ``beta``):

        .. math:: f(E) = A (E / E_0) ^ {-\\alpha}
                    \\exp (- (E / E_{cutoff}) ^ \\beta)

    """

    param_names = ["amplitude", "e_0", "alpha", "e_cutoff", "beta"]

    def __init__(self, amplitude, e_0, alpha, e_cutoff, beta=1.0):
        self.amplitude = amplitude
        self.e_0 = validate_scalar(
            "e_0", e_0, domain="positive", physical_type="energy"
        )
        self.alpha = alpha
        self.e_cutoff = validate_scalar(
            "e_cutoff", e_cutoff, domain="positive", physical_type="energy"
        )
        self.beta = beta

    @staticmethod
    def eval(e, amplitude, e_0, alpha, e_cutoff, beta):
        "One dimensional power law with an exponential cutoff model function"

        xx = e / e_0
        return amplitude * xx ** (-alpha) * np.exp(-((e / e_cutoff) ** beta))

    def _calc(self, e):
        return self.eval(
            e.to("eV").value,
            self.amplitude,
            self.e_0.to("eV").value,
            self.alpha,
            self.e_cutoff.to("eV").value,
            self.beta,
        )

    def __call__(self, e):
        "One dimensional power law with an exponential cutoff model function"
        e = _validate_ene(e)
        return self._calc(e)


class BrokenPowerLaw:
    """
    One dimensional power law model with a break.

    Parameters
    ----------
    amplitude : float
        Model amplitude at the break energy
    e_0 : `~astropy.units.Quantity` float
        Reference point
    e_break : `~astropy.units.Quantity` float
        Break energy
    alpha_1 : float
        Power law index for x < x_break
    alpha_2 : float
        Power law index for x > x_break

    See Also
    --------
    PowerLaw, ExponentialCutoffPowerLaw, LogParabola

    Notes
    -----
    Model formula (with :math:`A` for ``amplitude``, :math:`E_0` for ``e_0``,
    :math:`\\alpha_1` for ``alpha_1`` and :math:`\\alpha_2` for ``alpha_2``):

        .. math::

            f(E) = \\left \\{
                     \\begin{array}{ll}
                       A (E / E_0) ^ {-\\alpha_1} & : E < E_{break} \\\\
                       A (E_{break}/E_0) ^ {\\alpha_2-\\alpha_1}
                           (E / E_0) ^ {-\\alpha_2} & :  E > E_{break} \\\\
                     \\end{array}
                   \\right.
    """

    param_names = ["amplitude", "e_0", "e_break", "alpha_1", "alpha_2"]

    def __init__(self, amplitude, e_0, e_break, alpha_1, alpha_2):
        self.amplitude = amplitude
        self.e_0 = validate_scalar(
            "e_0", e_0, domain="positive", physical_type="energy"
        )
        self.e_break = validate_scalar(
            "e_break", e_break, domain="positive", physical_type="energy"
        )
        self.alpha_1 = alpha_1
        self.alpha_2 = alpha_2

    @staticmethod
    def eval(e, amplitude, e_0, e_break, alpha_1, alpha_2):
        """One dimensional broken power law model function"""
        K = np.where(e < e_break, 1, (e_break / e_0) ** (alpha_2 - alpha_1))
        alpha = np.where(e < e_break, alpha_1, alpha_2)
        return amplitude * K * (e / e_0) ** -alpha

    def _calc(self, e):
        return self.eval(
            e.to("eV").value,
            self.amplitude,
            self.e_0.to("eV").value,
            self.e_break.to("eV").value,
            self.alpha_1,
            self.alpha_2,
        )

    def __call__(self, e):
        """One dimensional broken power law model function"""
        e = _validate_ene(e)
        return self._calc(e)


class ExponentialCutoffBrokenPowerLaw:
    """
    One dimensional power law model with a break.

    Parameters
    ----------
    amplitude : float
        Model amplitude at the break point
    e_0 : `~astropy.units.Quantity` float
        Reference point
    e_break : `~astropy.units.Quantity` float
        Break energy
    alpha_1 : float
        Power law index for x < x_break
    alpha_2 : float
        Power law index for x > x_break
    e_cutoff : `~astropy.units.Quantity` float
        Exponential Cutoff energy
    beta : float, optional
        Exponential cutoff rapidity. Default is 1.

    See Also
    --------
    PowerLaw, ExponentialCutoffPowerLaw, LogParabola

    Notes
    -----
    Model formula (with :math:`A` for ``amplitude``, :math:`E_0` for ``e_0``,
    :math:`\\alpha_1` for ``alpha_1``, :math:`\\alpha_2` for ``alpha_2``,
    :math:`E_{cutoff}` for ``e_cutoff``, and :math:`\\beta` for ``beta``):

        .. math::


            f(E) = \\exp(-(E / E_{cutoff})^\\beta)\\left \\{
                     \\begin{array}{ll}
                       A (E / E_0) ^ {-\\alpha_1}    & : E < E_{break} \\\\
                       A (E_{break}/E_0) ^ {\\alpha_2-\\alpha_1}
                            (E / E_0) ^ {-\\alpha_2} & : E > E_{break} \\\\
                     \\end{array}
                   \\right.

    """

    param_names = [
        "amplitude",
        "e_0",
        "e_break",
        "alpha_1",
        "alpha_2",
        "e_cutoff",
        "beta",
    ]

    def __init__(self, amplitude, e_0, e_break, alpha_1, alpha_2, e_cutoff, beta=1.0):
        self.amplitude = amplitude
        self.e_0 = validate_scalar(
            "e_0", e_0, domain="positive", physical_type="energy"
        )
        self.e_break = validate_scalar(
            "e_break", e_break, domain="positive", physical_type="energy"
        )
        self.alpha_1 = alpha_1
        self.alpha_2 = alpha_2
        self.e_cutoff = validate_scalar(
            "e_cutoff", e_cutoff, domain="positive", physical_type="energy"
        )
        self.beta = beta

    @staticmethod
    def eval(e, amplitude, e_0, e_break, alpha_1, alpha_2, e_cutoff, beta):
        """One dimensional broken power law model function"""
        K = np.where(e < e_break, 1, (e_break / e_0) ** (alpha_2 - alpha_1))
        alpha = np.where(e < e_break, alpha_1, alpha_2)
        ee2 = e / e_cutoff
        return amplitude * K * (e / e_0) ** -alpha * np.exp(-(ee2**beta))

    def _calc(self, e):
        return self.eval(
            e.to("eV").value,
            self.amplitude,
            self.e_0.to("eV").value,
            self.e_break.to("eV").value,
            self.alpha_1,
            self.alpha_2,
            self.e_cutoff.to("eV").value,
            self.beta,
        )

    def __call__(self, e):
        """One dimensional broken power law model with exponential cutoff
        function"""
        e = _validate_ene(e)
        return self._calc(e)


class LogParabola:
    """
    One dimensional log parabola model (sometimes called curved power law).

    Parameters
    ----------
    amplitude : float
        Model amplitude
    e_0 : `~astropy.units.Quantity` float
        Reference point
    alpha : float
        Power law index
    beta : float
        Power law curvature

    See Also
    --------
    PowerLaw, BrokenPowerLaw, ExponentialCutoffPowerLaw

    Notes
    -----
    Model formula (with :math:`A` for ``amplitude`` and :math:`\\alpha` for
    ``alpha`` and :math:`\\beta` for ``beta``):

        .. math::

            f(e) = A \\left(\\frac{E}{E_{0}}\\right)^
                {- \\alpha - \\beta \\log{\\left (\\frac{E}{E_{0}} \\right )}}

    """

    param_names = ["amplitude", "e_0", "alpha", "beta"]

    def __init__(self, amplitude, e_0, alpha, beta):
        self.amplitude = amplitude
        self.e_0 = validate_scalar(
            "e_0", e_0, domain="positive", physical_type="energy"
        )
        self.alpha = alpha
        self.beta = beta

    @staticmethod
    def eval(e, amplitude, e_0, alpha, beta):
        """One dimenional log parabola model function"""

        ee = e / e_0
        eeponent = -alpha - beta * np.log(ee)
        return amplitude * ee**eeponent

    def _calc(self, e):
        return self.eval(
            e.to("eV").value,
            self.amplitude,
            self.e_0.to("eV").value,
            self.alpha,
            self.beta,
        )

    def __call__(self, e):
        """One dimensional curved power law function"""
        e = _validate_ene(e)
        return self._calc(e)


class TableModel:
    """
    A model generated from a table of energy and value arrays.

    The units returned will be the units of the values array provided at
    initialization. The model will return values interpolated in
    log-space, returning 0 for energies outside of the limits of the provided
    energy array.

    Parameters
    ----------
    energy : `~astropy.units.Quantity` array
        Array of energies at which the model values are given
    values : array
        Array with the values of the model at energies ``energy``.
    amplitude : float
        Model amplitude that is multiplied to the supplied arrays. Defaults to
        1.
    """

    def __init__(self, energy, values, amplitude=1):
        from scipy.interpolate import interp1d

        self._energy = validate_array(
            "energy", energy, domain="positive", physical_type="energy"
        )
        self._values = values
        self.amplitude = amplitude

        loge = np.log10(self._energy.to("eV").value)
        try:
            self.unit = self._values.unit
            logy = np.log10(self._values.value)
        except AttributeError:
            self.unit = u.Unit("")
            logy = np.log10(self._values)

        self._interplogy = interp1d(
            loge, logy, fill_value=-np.inf, bounds_error=False, kind="cubic"
        )

    def __call__(self, e):
        e = _validate_ene(e)
        interpy = np.power(10, self._interplogy(np.log10(e.to("eV").value)))
        return self.amplitude * interpy * self.unit


@functools.lru_cache(maxsize=None)
def _load_tau_dominguez11():
    filename = get_pkg_data_filename(os.path.join("data", "tau_dominguez11.npz"))
    return np.load(filename)["arr_0"]


class EblAbsorptionModel(TableModel):
    """
    A TableModel containing the EBL absorption factor (``exp(-tau)``, where
    ``tau`` is the optical depth) from a specific model.

    Calling the instance returns the dimensionless absorption factor, which is
    1 for no absorption and decreases towards 0 for stronger absorption. It can
    be multiplied to any model to obtain the absorbed one. Note that the optical
    depth ``tau`` itself is not returned.

    Parameters
    ----------
    redshift : float
        Redshift considered for the absorption evaluation.
    ebl_absorption_model : {'Dominguez'}
        Name of the EBL absorption model to use (Dominguez by default).

    Notes
    -----
    Dominguez model refers to the Dominguez 2011 EBL model. Current
    implementation does NOT perform an interpolation in the redshift, so it
    just uses the closest z value from the finely binned tau_dominguez11.npz
    file (delta_z=0.01).

    See Also
    --------
    TableModel
    """

    def __init__(self, redshift, ebl_absorption_model="Dominguez"):
        # check that the redshift is a positive scalar
        if not isinstance(redshift, u.Quantity):
            redshift *= u.dimensionless_unscaled

        self.redshift = validate_scalar(
            "redshift",
            redshift,
            domain="positive",
            physical_type="dimensionless",
        )

        self.model = ebl_absorption_model

        if self.model == "Dominguez":
            # Table generated by Alberto Dominguez containing tau vs energy
            # [TeV] vs redshift.  Energy is defined between 1 GeV and 100 TeV,
            # in 500 bins uniform in log(E).  Redshift is defined between 0.01
            # and 4, in steps of 0.01.
            taus_table = _load_tau_dominguez11()
            redshift_list = np.arange(0.01, 4, 0.01)
            energy = taus_table["energy"] * u.TeV
            if self.redshift >= 0.01:
                colname = "col%s" % (
                    2 + (np.abs(redshift_list - self.redshift)).argmin()
                )
                # Set maximum value of tau to 150, which is high enough that
                # the transmission is zero for all practical purposes. This
                # avoids later underflow problems.
                taus = np.minimum(taus_table[colname], 150.0)
            else:
                taus = np.zeros(len(energy))
        else:
            raise ValueError('Model should be one of: ["Dominguez"]')

        # The table is stored as the transmission exp(-tau), which is
        # positive and so can be interpolated in log space by TableModel
        super().__init__(energy, np.exp(-taus) * u.dimensionless_unscaled)
        self._emin = energy[0]

    def __call__(self, e):
        """Absorption factor ``exp(-tau)`` at energies ``e``.

        It is 1 below the lower energy limit of the table (1 GeV) and 0 above
        the upper limit (100 TeV), where the optical depth is above 200 for all
        redshifts of the table.
        """
        e = _validate_ene(e)
        if self.redshift < 0.01:
            # no absorption, also outside of the energy range of the table
            return np.ones(e.shape) * u.dimensionless_unscaled
        transmission = super().__call__(e).to_value("")
        transmission = np.where(e < self._emin, 1.0, np.clip(transmission, 0.0, 1.0))
        return transmission * u.dimensionless_unscaled

    def transmission(self, e):
        """Absorption factor ``exp(-tau)`` at energies ``e``, without units.

        Same values as calling the instance, returned as a dimensionless array
        (or scalar) that can be multiplied to any model.
        """
        return self(e).to_value("")
