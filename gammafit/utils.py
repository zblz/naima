# Licensed under a 3-clause BSD style license - see LICENSE.rst
import numpy as np

from .plot import plot_fit, plot_chain

__all__ = ["generate_energy_edges","build_data_dict","generate_diagnostic_plots"]

## Convenience tools

def generate_energy_edges(ene):
    """Generate energy bin edges from given energy array.
    
    Generate an array of energy edges from given energy array to be used as
    abcissa error bar limits when no energy uncertainty or energy band is
    provided. 

    Parameters
    ----------
    ene : array
        Array of energies

    Returns
    -------
    edge_array : array with shape (len(ene),2)
        Array of energy edge pairs corresponding to each given energy of the
        input array.
    """
    midene=np.sqrt((ene[1:]*ene[:-1]))
    elo,ehi=np.zeros_like(ene),np.zeros_like(ene)
    elo[1:]=ene[1:]-midene
    ehi[:-1]=midene-ene[:-1]
    elo[0]=ehi[0]
    ehi[-1]=elo[-1]
    return np.array(list(zip(elo,ehi)))

def build_data_dict(ene,dene,flux,dflux,ul=None,cl=0.99):
    """
    Read data into data dict.

    Parameters
    ----------

    ene : array (Nene)
        Spectrum energies

    dene : array (Nene,2) or None
        Difference from energy points to lower (column 0) and upper (column 1)
        energy edges. Currently only used on plots. If ``None`` is given, they
        will be generated with function ``generate_energy_edges``.

    flux : array (Nene)
        Spectrum flux values.

    dflux : array (Nene,2) or (Nene)
        Spectrum flux uncertainties. If shape is (Nene, 2), columns 0 and 1
        correspond to lower and upper uncertainties, respectively.

    ul : array of bool (optional)
        Boolean array indicating which of the flux values given in ``flux``
        correspond to upper limits.

    cl : float (optional)
        Confidence level of the flux upper limits given by ``ul``.

    Returns
    -------
    data : dict
        Data stored in a `dict`.
    """
    if ul==None:
        ul=np.array((False,)*len(ene))

    if dene==None:
        dene=generate_energy_edges(ene)

    # data is a dict with the fields:
    # ene dene flux dflux ul cl
    data={}
    for val in ['ene', 'dene', 'flux', 'dflux', 'ul', 'cl']:
        data[val]=eval(val)

    return data

def generate_diagnostic_plots(outname,sampler,modelidxs=None,pdf=False,converttosed=None,**kwargs):
    """
    Generate diagnostic plots.

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
    outname : str
        Name to be used to save diagnostic plot files.

    sampler : `emcee.EnsembleSampler` instance
        Sampler instance from which chains, blobs and data are read.

    modelidxs : iterable (optional)
        Model numbers to be plotted. Default: All returned in sampler.blobs

    pdf : bool (optional)
        Whether to save plots to multipage pdf.
    """

    if pdf:
        from matplotlib import pyplot as plt
        plt.rc('pdf',fonttype = 42)
        print('Generating diagnostic plots in file {}_plots.pdf'.format(outname))
        from matplotlib.backends.backend_pdf import PdfPages
        outpdf=PdfPages('{}_plots.pdf'.format(outname))

    ## Chains

    for par,label in zip(range(sampler.chain.shape[-1]),sampler.labels):
        f = plot_chain(sampler,par,**kwargs)
        if pdf:
            f.savefig(outpdf,format='pdf')
        else:
            f.savefig('{0}_chain_{1}.png'.format(outname,label))

    ## Corner plot

    try:
        from .plot import corner
        from .plot import find_ML

        ML,MLp,MLvar,model_ML = find_ML(sampler,0)
        f = corner(sampler.flatchain,labels=sampler.labels,truths=MLp,quantiles=[0.16,0.5,0.84],verbose=False,**kwargs)
        if pdf:
            f.savefig(outpdf,format='pdf')
        else:
            f.savefig('{0}_corner.png'.format(outname))
    except NameError:
        print('triangle.py not installed, corner plot not available')

    ## Fit

    if modelidxs==None:
        nmodels=len(sampler.blobs[-1][0])
        modelidxs=list(range(nmodels))

    if converttosed==None:
        converttosed=[False for idx in modelidxs]

    for modelidx,tosed in zip(modelidxs,converttosed):
        modelx=sampler.blobs[-1][0][modelidx][0]
        xunit='eV' if np.max(modelx)>1e8 else 'TeV'
        if modelidx==0:
            if tosed:
                labels=('Energy [{0}]'.format(xunit),r'$E^2$d$N$/d$E$ [erg/cm$^2$/s]')
            else:
                labels=('Energy [{0}]'.format(xunit),r'd$N$/d$E$ [1/cm$^2$/s/{0}]'.format(xunit))
        elif modelidx==1:
            labels=('Particle Energy [TeV]',r'Particle energy distribution [erg$\times 4\pi d^2$]')
        else:
            labels=( None, None)
        try:
            f = plot_fit(sampler, xlabel=labels[0], ylabel=labels[1],
                    modelidx=modelidx, converttosed=tosed,**kwargs)
            if pdf:
                f.savefig(outpdf,format='pdf')
            else:
                f.savefig('{0}_fit_model{1}.png'.format(outname,modelidx))
        except Exception as e:
            # Maybe one of the returned models does not conform to the needed format
            print(e)

    if pdf:
        outpdf.close()
