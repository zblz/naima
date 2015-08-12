# Licensed under a 3-clause BSD style license - see LICENSE.rst
from __future__ import (absolute_import, division, print_function,
                        unicode_literals)

import numpy as np
import astropy.units as u
from astropy.extern import six
from astropy import log

import matplotlib.pyplot as plt
from matplotlib.widgets import Button, Slider, CheckButtons

from .plot import color_cycle, plot_data, _plot_data_to_ax
from .utils import sed_conversion, validate_data_table
from .extern.validator import validate_array

class ModelWidget(object):
    def __init__(self, modelfn, p0, data=None, e_range=None,
            labels=None, sed=True, auto_update=True, e_npoints=100):

        npars = len(p0)
        if labels is None:
            labels = ['par{0}'.format(i) for i in range(npars)]
        elif len(labels) < npars:
            labels += ['par{0}'.format(i) for i in range(len(labels),npars)]

        hasdata = data is not None

        self.modelfn = modelfn

        self.fig = plt.figure()
        modelax = plt.subplot2grid((2*npars+1,4),(0,0),rowspan=npars,colspan=4)

        if e_range:
            e_range = validate_array('e_range', u.Quantity(e_range),
                    physical_type='energy')
            e_unit = e_range.unit
            energy = np.logspace(np.log10(e_range[0].value),
                    np.log10(e_range[1].value), e_npoints) * e_unit
        else:
            energy = np.logspace(-4,2,e_npoints)*u.TeV
        if sed:
            flux = np.zeros(e_npoints) * u.Unit('erg/(cm2 s)')
        else:
            flux = np.zeros(e_npoints) * u.Unit('1/(TeV cm2 s)')


        if hasdata:
            data = validate_data_table(data)
            _plot_data_to_ax(data, modelax, sed=sed)
            if not e_range:
                energy = data['energy']
                flux = data['flux']

        self.data_for_model = {'energy': energy,
                'flux': flux}
        model = modelfn(p0, self.data_for_model)[0]

        self.f_unit, self.sedf = sed_conversion(energy, model.unit, sed)
        if hasdata:
            modelax.set_xlim(
                    (data['energy'][0] - data['energy_error_lo'][0]).value / 3,
                    (data['energy'][-1] + data['energy_error_hi'][-1]).value * 3)
        else:
            # plot_data_to_ax has not set ylabel
            unit = self.f_unit.to_string('latex_inline')
            if sed:
                modelax.set_ylabel(r'$E^2 dN/dE [{0}]$'.format(unit))
            else:
                modelax.set_ylabel(r'$dN/dE [{0}]$'.format(unit))
            modelax.set_xlim(data['energy'][0].value/3,
                    data['energy'][-1].value*3)

        self.line, = modelax.loglog(energy,
                (model*self.sedf).to(self.f_unit), lw=2,
                c='k', zorder=10)

        modelax.set_xlabel(
                'Energy [{0}]'.format(energy.unit.to_string('latex_inline')))


        paraxes = []
        for n in range(npars):
            paraxes.append(plt.subplot2grid((2*npars+1,4),(1+npars+n,0),colspan=2))
        self.parsliders = []
        for label,parax,valinit in six.moves.zip(labels, paraxes, p0):
            # Attempt to estimate reasonable parameter ranges from label
            pmin, pmax = valinit/10, valinit*3
            if 'log' in label:
                pmin = valinit - 2
                pmax = valinit + 2
            elif ('index' in label) or ('alpha' in label):
                if valinit > 0.:
                    pmin = -2
                    pmax = 5
                else:
                    pmin = -5
                    pmax = 2

            slider = Slider(parax, label, pmin, pmax,
                valinit=valinit, valfmt='%g',closedmin=False, closedmax=False)
            slider.on_changed(self.update_if_auto)
            self.parsliders.append(slider)

        updateax = plt.subplot2grid((2*npars+1,4),(npars+1,3),colspan=1,
                rowspan=min(int(npars/2),2))
        update_button = Button(updateax, 'Update model')
        update_button.on_clicked(self.update)

        autoupdateax = plt.subplot2grid((2*npars+1,4),
                (npars+1+min(int(npars/2),2),3),colspan=1,
                rowspan=min(int(npars/2),2))
        auto_update_check = CheckButtons(autoupdateax,
                ('Auto update',), (auto_update,))
        auto_update_check.on_clicked(self.update_autoupdate)
        self.autoupdate = auto_update

        self.fig.subplots_adjust(top=0.98,right=0.98,bottom=0.02,hspace=0.2)

        plt.show()

    def update_autoupdate(self,label):
        self.autoupdate = not self.autoupdate

    def update_if_auto(self,val):
        if self.autoupdate:
            self.update(val)

    def update(self,event):
        self.pars = [slider.val for slider in self.parsliders]
        model = self.modelfn(self.pars, self.data_for_model)[0]
        self.line.set_ydata((model*self.sedf).to(self.f_unit))
        self.fig.canvas.draw_idle()


