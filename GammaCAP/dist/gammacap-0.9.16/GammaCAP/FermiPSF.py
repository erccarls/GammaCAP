"""@package FermiPSF.  Tools for extracting Fermi-LAT point spread functions (averaged PSF's for front, back, and front+back converting events already included in package). 

"""
#import matplotlib.pyplot as plt  #@UnresolvedImport
import numpy as np
import pyfits
import os

def GetR68(eMin, eMax, convType = 'both',fraction=.68):
    '''
    Finds the containment radius table for the energies specified based on Fermi data.
    @param eMin Minimum photon energy in MeV
    @param eMax Maximum photon energy in MeV
    @param convType Fermi-LAT event conversion type.  Either 'front', 'back', or 'both'
    @param fraction Containment fraction
    @returns: Returns the radius for 'fraction' containment, averaged over the energy range provided weighted by E^-2.5 
    '''     
    #===========================================================================
    # Load PSF from FITS averaged over all-sky fermi data
    #===========================================================================
    if convType == 'both':
        hdulist = pyfits.open(os.path.join(os.path.dirname(__file__), '.', 'psf_both.fits'), mode='update')
    if convType == 'front':
        hdulist = pyfits.open(os.path.join(os.path.dirname(__file__), '.', 'psf_front.fits'), mode='update')
    elif convType == 'back': 
        hdulist = pyfits.open(os.path.join(os.path.dirname(__file__), '.', 'psf_back.fits'), mode='update')
    #hdulist.info()
    # Load the list of PSF's.  The first dimension corresponds to each energy.
    E   = np.array([data[0] for data in hdulist[1].data])
    Ebin= np.append(np.logspace(np.log10(50),np.log10(6e5),250), (6e5**1.00284327291,))
    de = np.array([0.5*(Ebin[i+1]+Ebin[i]) for i in range(len(E))])
    
    # One psf for each energy
    PSF_LIST = np.array([data[2] for data in hdulist[1].data])
    THETA = np.array([i[0] for i in hdulist[2].data])
    
    idx1 = range(0,len(THETA)-1)
    idx2 = range(1,len(THETA))
    R68 = []
    for PSF in PSF_LIST:
        # (multiply each entry by r*dr to obtain fraction in that band.  Normalize to one.
        PSF  = PSF[idx1]*(THETA[idx2]+THETA[idx1])*(THETA[idx2]-THETA[idx1])
        PSF /= np.sum(PSF)
        # Append the 68% containment radius for this energy.
        R68.append(THETA[np.argmin(np.abs(np.cumsum(PSF)-fraction))])
    R68 = np.array(R68)
    # Now for each R68 between the energy range, weight by the energy^-2.5
    idx = range(np.argmin(np.abs(E-eMin)),np.argmin(np.abs(E-eMax)))
    # handle case where the indices are the same.
    if len(idx)==0:idx = np.argmin(np.abs(E-eMin))    
    weights = de*np.power(E,-2.5)
    # don't bother averaging if only one psf anyway.
    if type(idx)==np.int64: return R68[idx]
    meanR68 = np.average(R68[idx],weights=weights[idx])
    return meanR68
    
def GetPSF(eMin, eMax, convType = 'both'):
    '''
    Finds the containment radius table for the energies specified based on Fermi data.
    @param eMin Minimum photon energy in MeV
    @param eMax Maximum photon energy in MeV
    @param convType Fermi-LAT event conversion type.  Either 'front', 'back', or 'both'
    @returns: (r,psf) where 'r' is the psf radius in degrees and 'psf[i]' is the probability 
    of a photon to be at corresponding radius r[i] averaged over the energy range provided 
    weighted by E^-2.5.  This has already been weighted by the area of the annulus and can 
    simply be sampled as r.
    '''     
    #===========================================================================
    # Load PSF from FITS averaged over all-sky fermi data
    #===========================================================================
    if convType == 'both':
        hdulist = pyfits.open(os.path.join(os.path.dirname(__file__), '.', 'psf_both.fits'), mode='update')
    if convType == 'front':
        hdulist = pyfits.open(os.path.join(os.path.dirname(__file__), '.', 'psf_front.fits'), mode='update')
    elif convType == 'back': 
        hdulist = pyfits.open(os.path.join(os.path.dirname(__file__), '.', 'psf_back.fits'), mode='update')
    #hdulist.info()
    # Load the list of PSF's.  The first dimension corresponds to each energy.
    E   = np.array([data[0] for data in hdulist[1].data])
    Ebin= np.append(np.logspace(np.log10(50),np.log10(6e5),250), (6e5**1.00284327291,))
    de = np.array([0.5*(Ebin[i+1]+Ebin[i]) for i in range(len(E))])
    
    # One psf for each energy
    PSF_LIST = np.array([data[2] for data in hdulist[1].data])
    THETA = np.array([i[0] for i in hdulist[2].data])
    
    idx1 = range(0,len(THETA)-1)
    idx2 = range(1,len(THETA))
    meanpsf = []
    for PSF in PSF_LIST:
        # (multiply each entry by r*dr to obtain fraction in that band.  Normalize to one.
        PSF  = PSF[idx1]*(THETA[idx2]+THETA[idx1])*(THETA[idx2]-THETA[idx1])
        PSF /= np.sum(PSF)
        # Append the 68% containment radius for this energy.
        meanpsf.append([PSF,])
    meanpsf = np.array(meanpsf)
    # Now for each R68 between the energy range, weight by the energy^-2.5
    idx = range(np.argmin(np.abs(E-eMin)),np.argmin(np.abs(E-eMax)))
    # handle case where the indices are the same.
    if len(idx)==0:idx = np.argmin(np.abs(E-eMin))    
    weights = de*np.power(E,-2.5)
    # don't bother averaging if only one psf anyway.
    if type(idx)==np.int64: return 0.5*(THETA[idx2]+THETA[idx1]), meanpsf[idx][0]
    meanpsf = np.average(meanpsf[idx],weights=weights[idx],axis=0)
    return 0.5*(THETA[idx2]+THETA[idx1]), meanpsf[0]




