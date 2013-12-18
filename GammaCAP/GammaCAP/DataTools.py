"""@package GammaCAP
GammaCAP.DataTools provides functions to import Fermi-LAT and gtobssim data. 
"""

import pyfits
import numpy as np
 
def LoadData(fname, eMin=50,eMax=1e6,tMin=0,tMax=1e10):
    """
    Load a Fermi-LAT event file or gtobssim simulation.  This should be previously filtered through gtselect and gtmktime with standard options and contain P7Source class events.
    @param eMin Minimum event energy in MeV.
    @param eMax Maximum event energy in MeV.
    @param tMin Minimum event time in seconds.
    @param tMax Maximum event time in seconds.
    @returns ndarray of shape (n,4) with columns (B,L,T,E)
    """
    hdulist = pyfits.open(fname, mode='update')
    #hdulist.info()
    E = hdulist[1].data['Energy']
    L = hdulist[1].data['L']
    B = hdulist[1].data['B']
    T = hdulist[1].data['Time']
    ecut = np.logical_and(E>eMin,E<eMax)
    tcut = np.logical_and(T>tMin,T<tMax)
    idx = np.where(np.logical_and(ecut,tcut)==True)[0]
    return (B[idx],L[idx],T[idx],E[idx])

