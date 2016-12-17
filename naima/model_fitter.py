# Licensed under a 3-clause BSD style license - see LICENSE.rst
from __future__ import (absolute_import, division, print_function,
                        unicode_literals)

import numpy as np
import astropy.units as u
from astropy.extern import six

from .core import lnprobmodel, _prefit
from .plot import color_cycle, _plot_data_to_ax
from .utils import sed_conversion, validate_data_table
from .extern.validator import validate_array

__all__ = ['InteractiveModelFitter']


def _process_model(model):
    if ((isinstance(model, tuple) or isinstance(model, list)) and
            not isinstance(model, np.ndarray)):
        return model[0]
    else:
        return model


class InteractiveModelFitter(object):
    """
    Interactive model fitter using matplotlib widgets


    Parameters
    ----------
    modelfn : function
        A function that takes a vector in the parameter space and the data
        table, and returns the expected fluxes at the energies in the
        spectrum.
    p0 : array
        Initial position vector.
    data : `~astropy.table.Table` or list of `~astropy.table.Table`
        Table or tables with observed spectra. Must follow the format described
        in `naima.run_sampler`.
    e_range : list of `~astropy.units.Quantity`, length 2, optional
        Limits in energy for the computation of the model.
        Note that setting this parameter will mean that the model output is
        computed twice when `data` is provided: once for display using
        `e_range` and once for computation of the log-probability using the
        energy values of the spectra.
    e_npoints : int, optional
        How many points to compute for the model if `e_range` is set. Default
        is 100.
    labels : iterable of strings, optional
        Labels for the parameters included in the position vector ``p0``. If
        not provided ``['par1','par2', ... ,'parN']`` will be used.
    sed : bool, optional
        Whether to plot SED or differential spectrum.
    auto_update : bool, optional
        Whether to update the model plot when parameter sliders are changed.
        Default is True and can also be changed through the GUI.
    """

    def __init__(self,
                 modelfn,
                 p0,
                 data=None,
                 e_range=None,
                 e_npoints=100,
                 labels=None,
                 sed=True,
                 auto_update=True):

        import matplotlib.pyplot as plt
        from matplotlib.widgets import Button, Slider, CheckButtons

        self.pars = p0
        self.P0_IS_ML = False
        npars = len(p0)
        if labels is None:
            labels = ['par{0}'.format(i) for i in range(npars)]
        elif len(labels) < npars:
            labels += ['par{0}'.format(i) for i in range(len(labels), npars)]

        self.hasdata = data is not None
        self.data = None
        if self.hasdata:
            self.data = validate_data_table(data, sed=sed)

        self.modelfn = modelfn
        self.fig = plt.figure()
        modelax = plt.subplot2grid((10, 4), (0, 0), rowspan=4, colspan=4)

        if e_range:
            e_range = validate_array(
                'e_range', u.Quantity(e_range), physical_type='energy')
            energy = np.logspace(
                np.log10(e_range[0].value),
                np.log10(e_range[1].value), e_npoints) * e_range.unit
            if self.hasdata:
                energy = energy.to(self.data['energy'].unit)
            else:
                e_unit = e_range.unit
        else:
            energy = np.logspace(-4, 2, e_npoints) * u.TeV
            e_unit = u.TeV

        # Bogus flux array to send to model if not using data
        if sed:
            flux = np.zeros(e_npoints) * u.Unit('erg/(cm2 s)')
        else:
            flux = np.zeros(e_npoints) * u.Unit('1/(TeV cm2 s)')

        if self.hasdata:
            e_unit = self.data['energy'].unit
            _plot_data_to_ax(self.data, modelax, sed=sed, e_unit=e_unit)
            if not e_range:
                # use data for model
                energy = self.data['energy']
                flux = self.data['flux']

        self.data_for_model = {'energy': energy, 'flux': flux}

        model = _process_model(self.modelfn(p0, self.data_for_model))

        if self.hasdata:
            if not np.all(
                    self.data_for_model['energy'] == self.data['energy']):
                # this will be slow, maybe interpolate already computed model?
                model_for_lnprob = _process_model(
                    self.modelfn(self.pars, self.data))
            else:
                model_for_lnprob = model
            lnprob = lnprobmodel(model_for_lnprob, self.data)
            if isinstance(lnprob, u.Quantity):
                lnprob = lnprob.decompose().value
            self.lnprobtxt = modelax.text(
                0.05,
                0.05,
                r'',
                ha='left',
                va='bottom',
                transform=modelax.transAxes,
                size=20)
            self.lnprobtxt.set_text(r'$\ln\mathcal{{L}} = {0:.1f}$'.format(
                lnprob))

        self.f_unit, self.sedf = sed_conversion(energy, model.unit, sed)

        if self.hasdata:
            datamin = (self.data['energy'][0] - self.data['energy_error_lo'][0]
                       ).to(e_unit).value / 3
            xmin = min(datamin, energy[0].to(e_unit).value)
            datamax = (
                self.data['energy'][-1] + self.data['energy_error_hi'][-1]
            ).to(e_unit).value * 3
            xmax = max(datamax, energy[-1].to(e_unit).value)
            modelax.set_xlim(xmin, xmax)
        else:
            # plot_data_to_ax has not set ylabel
            unit = self.f_unit.to_string('latex_inline')
            if sed:
                modelax.set_ylabel(r'$E^2 dN/dE$ [{0}]'.format(unit))
            else:
                modelax.set_ylabel(r'$dN/dE$ [{0}]'.format(unit))
            modelax.set_xlim(energy[0].value, energy[-1].value)

        self.line, = modelax.loglog(
            energy.to(e_unit), (model * self.sedf).to(self.f_unit),
            lw=2,
            c='k',
            zorder=10)

        modelax.set_xlabel('Energy [{0}]'.format(
            energy.unit.to_string('latex_inline')))

        paraxes = []
        for n in range(npars):
            paraxes.append(
                plt.subplot2grid(
                    (2 * npars, 10), (npars + n, 0), colspan=7))
        self.parsliders = []
        slider_props = {'facecolor': color_cycle[-1], 'alpha': 0.5}
        for label, parax, valinit in six.moves.zip(labels, paraxes, p0):
            # Attempt to estimate reasonable parameter ranges from label
            pmin, pmax = valinit / 10, valinit * 3
            if 'log' in label:
                span = 2
                if 'norm' in label or 'amplitude' in label:
                    # give more range for normalization parameters
                    span *= 2
                pmin, pmax = valinit - span, valinit + span
            elif ('index' in label) or ('alpha' in label):
                if valinit > 0.:
                    pmin, pmax = 0, 5
                else:
                    pmin, pmax = -5, 0
            elif 'norm' in label or 'amplitude' in label:
                # norm without log, it will not be pretty because sliders are
                # only linear
                pmin, pmax = valinit / 100, valinit * 100

            slider = Slider(
                parax,
                label,
                pmin,
                pmax,
                valinit=valinit,
                valfmt='%.4g',
                **slider_props)
            slider.on_changed(self.update_if_auto)
            self.parsliders.append(slider)

        autoupdateax = plt.subplot2grid((8, 4), (4, 3), colspan=1, rowspan=1)
        auto_update_check = CheckButtons(autoupdateax, ('Auto update',),
                                         (auto_update,))
        auto_update_check.on_clicked(self.update_autoupdate)
        self.autoupdate = auto_update

        updateax = plt.subplot2grid((8, 4), (5, 3), colspan=1, rowspan=1)
        update_button = Button(updateax, 'Update model')
        update_button.on_clicked(self.update)

        if self.hasdata:
            fitax = plt.subplot2grid((8, 4), (6, 3), colspan=1, rowspan=1)
            fit_button = Button(fitax, 'Do Nelder-Mead fit')
            fit_button.on_clicked(self.do_fit)

        closeax = plt.subplot2grid((8, 4), (7, 3), colspan=1, rowspan=1)
        close_button = Button(closeax, 'Close window')
        close_button.on_clicked(self.close_fig)

        self.fig.subplots_adjust(top=0.98, right=0.98, bottom=0.02, hspace=0.2)

        plt.show()

    def update_autoupdate(self, label):
        self.autoupdate = not self.autoupdate

    def update_if_auto(self, val):
        if self.autoupdate:
            self.update(val)

    def update(self, event):
        # If we update, values have changed and P0 is not ML anymore
        self.P0_IS_ML = False
        self.pars = [slider.val for slider in self.parsliders]
        model = _process_model(self.modelfn(self.pars, self.data_for_model))
        self.line.set_ydata((model * self.sedf).to(self.f_unit))
        if self.hasdata:
            if not np.all(
                    self.data_for_model['energy'] == self.data['energy']):
                # this will be slow, maybe interpolate already computed model?
                model = _process_model(self.modelfn(self.pars, self.data))
            lnprob = lnprobmodel(model, self.data)
            if isinstance(lnprob, u.Quantity):
                lnprob = lnprob.decompose().value
            self.lnprobtxt.set_text(r'$\ln\mathcal{{L}} = {0:.1f}$'.format(
                lnprob))
        self.fig.canvas.draw_idle()

    def do_fit(self, event):
        self.pars = [slider.val for slider in self.parsliders]
        self.pars, P0_IS_ML = _prefit(self.pars, self.data, self.modelfn, None)
        autoupdate = self.autoupdate
        self.autoupdate = False
        if P0_IS_ML:
            for slider, val in zip(self.parsliders, self.pars):
                slider.set_val(val)
            self.update('after_fit')
        self.autoupdate = autoupdate
        self.P0_IS_ML = P0_IS_ML

    def close_fig(self, event):
        import matplotlib.pyplot as plt
        plt.close(self.fig)
