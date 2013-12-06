"""@package FermiPSF.  Tools for extracting Fermi-LAT point spread functions (averaged PSF's for front, back, and front+back converting events already included in package). 

"""

import matplotlib.pyplot as plt  #@UnresolvedImport
import numpy as np
import pyfits

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
        hdulist = pyfits.open('../psf_both.fits', mode='update')
    if convType == 'front':
        hdulist = pyfits.open('../psf_front.fits', mode='update')
    elif convType == 'back': 
        hdulist = pyfits.open('../psf_back.fits', mode='update')
    #hdulist.info()
    # Load the list of PSF's.  The first dimension corresponds to each energy.
    E   = np.array([data[0] for data in hdulist[1].data])
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
    plt.loglog(E,R68)
    plt.show()
    # Now for each R68 between the energy range, weight by the energy^-2.5
    idx = range(np.argmin(np.abs(E-eMin)),np.argmin(np.abs(E-eMax)))
    weights = E**-2.5
    meanR68 = np.average(R68[idx],weights=weights[idx])
    return meanR68


print GetR68(50000,5e5,convType='both')

#for convType in ['front','back','both']:
#    E,R68 = GetR68(50,1000,convType=convType)
    #plt.loglog(E,R68)
plt.show()


