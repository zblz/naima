#!/usr/bin/env python
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats
from scipy.interpolate import interp1d

from .plot import *

__all__ = ["generate_energy_edges","build_data_dict","generate_diagnostic_plots"]

## Convenience tools

def generate_energy_edges(ene):
    """Generate an array of energy edges from given energy array to be used as
    abcissa error bar limits when no energy uncertainty or energy band is
    provided

    Parameters
    ----------
    ene : array
        Array of energies

    Returns
    -------
    edge_array : array with shape (2,len(ene))
        Array of energy edge pairs corresponding to each given energy of the
        input array.
    """
    midene=np.sqrt((ene[1:]*ene[:-1]))
    elo,ehi=np.zeros_like(ene),np.zeros_like(ene)
    elo[1:]=ene[1:]-midene
    ehi[:-1]=midene-ene[:-1]
    elo[0]=ehi[0]
    ehi[-1]=elo[-1]
    return np.array(zip(elo,ehi))

def build_data_dict(ene,dene,flux,dflux,ul=None,cl=0.99):
    """
    read data into data dict
    """
    if ul==None:
        ul=np.array((False,)*len(ene))

    # data is a dict with the fields:
    # ene dene flux dflux ul cl
    data={}
    for val in ['ene', 'dene', 'flux', 'dflux', 'ul', 'cl']:
        data[val]=eval(val)

    return data

def generate_diagnostic_plots(outname,sampler,modelidxs=None):
    """
    Generate diagnostic plots:

    - A corner plot of sample density in the two dimensional parameter space of
      all parameter pairs of the run: ``outname_corner.png``
    - A plot for each of the chain parameters showing walker progression, final
      sample distribution and several statistical measures of this distribution:
      ``outname_chain_parN.png``
    - A plot for each of the models returned as blobs by the model function. The
      maximum likelihood model is shown, as well as the 1 and 3 sigma confidence
      level contours. The first model will be compared with observational data
      and residuals shown. ``outname_fit_modelN.png``

    Parameters
    ----------
    outname: str
        Name to be used to save diagnostic plot files.

    sampler: emcee.EnsembleSampler instance
        Sampler instance from which chains, blobs and data are read.

    modelidxs: iterable (optional)
        Model numbers to be plotted. Default: All returned in sampler.blobs

    """

    print 'Generating diagnostic plots'

    try:
        ## Corner plot
        f = corner(sampler.flatchain,labels=sampler.labels)
        f.savefig('{0}_corner.png'.format(outname))
    except NameError:
        print 'triangle.py not installed, corner plot will not be available'

    ## Chains

    for par,label in zip(range(sampler.chain.shape[-1]),sampler.labels):
        f = plot_chain(sampler,par)
        f.savefig('{0}_chain_{1}.png'.format(outname,label))

    ## Fit

    if modelidxs==None:
        nmodels=len(sampler.blobs[-1][0])
        modelidxs=range(nmodels)

    for modelidx in modelidxs:
        if modelidx==0:
            labels=('Energy','Flux')
        else:
            labels=( None, None)
        f = plot_fit(sampler,xlabel=labels[0],ylabel=labels[1],modelidx=modelidx,last_step=False)
        f.savefig('{0}_fit_model{1}.png'.format(outname,modelidx))
