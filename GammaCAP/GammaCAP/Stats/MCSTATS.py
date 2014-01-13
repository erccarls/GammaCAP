"""@package MCSTATS
Main package for computing cluster statistics and performing DBSCAN operation.

@author: Eric Carlson
"""
import matplotlib
if matplotlib.get_backend() == 'WXAgg': matplotlib.use('QT4Agg')
import matplotlib.pyplot as plt  
import cPickle as pickle
import numpy as np
import scipy.cluster as cluster
from matplotlib.backends.backend_pdf import PdfPages
import DBSCAN
from .. import BGTools
import multiprocessing as mp
from multiprocessing import pool
from functools import partial
import ClusterResult
import scipy.linalg as la
import time,os
import pyfits
from .. import FermiPSF
import sys


class Scan:
    """
    Class containing DBSCAN setup and cluster properties.  This is the only object a general user will need to interact with.
    """
    def __init__(self, eps=-1. ,nMinMethod='BGInt', a = -1.,D=2, nMinSigma=5. ,nCorePoints = 3, 
                 nMin = -1 , sigMethod ='BGInt', bgDensity=-1, totalTime=-1,inner=1.25,outer=2.0, 
                 fileout = '',numProcs = 1, plot=False,indexing=True,metric='spherical',eMax=-1,eMin=-1,
                 output = True, diffModel = '',convType='both',containment=0.68):
        """
        Initialize the Scan object which contains settings for DBSCAN and various cluster property calculations.  Default settings handle most cases, but custom scans may also be done.
        All settings are immediately put into Scan member variables and detailed descriptions can be found there.
        """
        ##@var D
        # Number of dimensions for DBSCAN to use. DDEFAULT=2\n
        # Values:\n
        # int in {2,3} (DEFAULT=2)
        ##@var a
        #The DBSCAN3d temporal search half-height.  Unused for 2d scans.\n
        #Values:  \n
        #    -1 (DEFAULT): Uses average expected background above 20% latitude l to estimate where the fragmentation limit lies.  Often, this is larger than one may want to search for and this parameter should be set explicitly.
        #    float>0: Specified in seconds.
        ##@var eps
        #The DBSCAN3d spatial search radius.\n
        #Values:  \n
        #    -1 (DEFAULT): Averages the Fermi 68% containment radius over the input energy ranges (by default) or ranges specified by eMin and eMax.\n  
        #    float>0: Specified in degrees.     
        ##@var nMinMethod 
        #Specifies which method to use when evaluating the background count for DBSCAN.\n
        #Values:\n
        #    'BGInt' (DEFAULT) : This is the best approach for FermiData, but for very large numbers of points (>1e6) can be computationally inefficient\n
        #    'BGCenter'        : Same as above, but just samples the center value of the BG template instead of integrating.  Much more efficient.\n
        #    'isotropic'       : One must specify bgDensity as well which is multiplied by the DBSCAN search volume to obtain the background rate.\n
        #    'nMin'            : Specify the value of nMin yourself, instead of using nMinSigma (not very useful for varying background)
        ##@var nMinSigma
        #Z-score over mean background density to calculate the nMin parameter of DBSCAN.  Unused if nMinMethod='nMin'.\n
        #Values:\n
        #float>0 (DEFAULT 5).
        ##@var convType
        # Fermi-LAT conversion type.  Can be 'front','back', or 'both' (default)
        ##@var nMin
        # DBSCAN parameter for the minimum number of events in an eps-neighborhood before becoming a core point (Unused by default). If nMinMethod='nMin', this specifies the value to use.\n
        # Values:\n
        # int>0: For uniform nMin.\n
        # numpy.ndarray shape(n,1): Specify an array of positive integers where n is the number of input samples.  
        ##@var metric
        # Specify the metric of the input data.
        # Values:\n
        # 'spherical' (DEFAULT):  This should be used for any real data in galactic coordinates\n
        # 'euclidean' : If simulating data it can be easier to use euclidean coordinates.
        ##@var diffModel
        # Absolute path to the galactic diffuse model. (typically '$FERMI_DIR/refdata/fermi/galdiffuse/gll_iem_v05.fits') where $FERMI_DIR is the Fermi science tools installation path.
        ##@var isoModel
        # Unused
        ##@var clusterResults
        # Stores the ClusterResults object from each scan. 
        ##@var nCorePoints
        #Clusters must have at least nCorePoints members to be valid. default=3 (i.e. this is low threshold for the number of events).\n
        #Values:\n
        #int>2 (DEFAULT=3)
        ##@var sigMethod
        #Specifies the method for evaluating the expected background count (thus significance) of a cluster.\n  
        #Values:\n
        #'BGInt' (DEFAULT): Integrate the background template over the cluster ellipse to determine local background density. **note**:currently integrates a square circumscribed by the semimajor axis to find density and multiplies by ellipse area.\n
        #'annulus': Computes count density in an annulus of radii ('inner','outer')*semimajor-axis centered on the cluster centroid.        \n
        #'isotropic': Uses bgDensity*ellipse-area to determine the background count.\n
        ##@var bgDensity
        # Specifies the mean background density (Unused unless (sigMethod or nMinMethod)='isotropic').\n
        # Values:\n
        # float>0 (DEFAULT -1): If -1 computes area spanned by input and divides by total number of photons.  Else specify in photons/deg^2. **Automatic calculation not yet implemented
        ##@var totalTime
        # The total exposure or simulation time being used in seconds.\n
        # Values: \n
        # float>1 (DEFAULT -1): If -1, use max-min time of input. Best option as long as data not extremely sparse.  Else specified in seconds
        ##@var inner
        # Inner fractional radius of the annulus. Unused unless sigMethod='annulus'.\n
        # The inner radius used to evaluate the background count expected for a cluster is given by inner*Semimajor-axis in degrees.
        # Values: \n
        # float>0 (DEFAULT 1.25)
        ##@var outer
        # Outer fractional radius of the annulus. Unused unless sigMethod='annulus'.\n
        # The outer radius used to evaluate the background count expected for a cluster is given by inner*Semimajor-axis in degrees.
        # Values: \n
        # float>0 (DEFAULT 2)
        ##@var fileout
        # If specified, ClusterResult is written (via cpickle) to file at this path. 
        ##@var numProcs
        # For multithreaded portions, specifies the number of processors to use (currently not implemented).
        ##@var plot
        # If True, scatter plot the DBSCAN results with clusters color-coded and noise in black (DEFAULT=False).
        ##@var indexing
        # If True, use grid-based indexing to speedup cluster computations (default=True)\n
        # Can slightly improve speed if processing many simulations with <2k points each.  Otherwise leave True 
        ##@var eMin
        # Specifies the minimum energy of the input in MeV (autodetected by default, but if specified will clip data)
        ##@var eMax
        # Specifies the maximum energy of the input in MeV (autodetected by default, but if specified will clip data)
        ##@var output
        #If False, supresses progress output.
        ##@var BG 
        #The BGTools instance being used for the current scan.  Unused if nMinMethod & sigMethod = 'isotropic'
        ##@var containment
        # Containment fraction to use in automating choice of epsilon.  Defaults to 0.68, the 68% containment fraction.  Must be float in (0,1)
        self.D = D
        self.a           = float(a)
        self.eps         = float(eps)
        self.nMinMethod  = nMinMethod
        self.nMinSigma   = float(nMinSigma)
        self.nMin        = int(nMin)
        self.metric      = metric
        self.diffModel   = diffModel
        self.isoModel    = ''
        self.clusterResults = []
        self.nCorePoints = int(nCorePoints)  
        self.sigMethod   = sigMethod
        self.bgDensity   = bgDensity
        self.totalTime   = totalTime
        self.inner       = float(inner)
        self.outer       = float(outer)
        self.fileout     = str(fileout)
        self.numProcs    = int(numProcs) 
        self.plot        = plot
        self.indexing    = bool(indexing)
        self.eMin = float(eMin)
        self.eMax = float(eMax)
        self.output = output
        self.BG = []
        self.convType = convType
        self.containment = containment
        
    def ComputeClusters(self, mcSims):
        '''
        Main DBSCAN cluster method.  Input a list of simulation outputs and output a list of clustering properties for each simulation.
        @param mcSims numpy.ndarray of shape(4,n) containing (latitude,longitude,time,energy) for 'spherical' (galactic) coordinates or (x,y,t,E) for 'euclidean' coordinates.
        @returns Returns a ClusterResult object with DBSCAN results.
        '''     
        np.seterr(divide='raise')   
        #====================================================================
        # Check if the diffuse model path has been specified.  
        # If not, try to locate in fermitools directory
        #====================================================================
        if self.diffModel == '':
            try: fermi_dir = os.environ['FERMI_DIR']
            except: raise KeyError('It appears that Fermitools is not setup or $FERMI_DIR environmental variable is not setup.  This is ok, but you must specify a path to the galactic diffuse model in the Scan.diffModel setting and isotropic model in Scan.isoModel') 
            path = os.environ['FERMI_DIR'] + '/refdata/fermi/galdiffuse/gll_iem_v05.fits'
            if os.path.exists(path)==True: self.diffModel = path
            else: raise ValueError('Fermitools appears to be setup, but cannot find diffuse model at path ' + path + '.  Please download to this directory or specify the path in Scan.diffModel' )
        #if self.isoModel == '':
        #    path = os.environ['FERMI_DIR'] + '/refdata/fermi/galdiffuse/iso_source_v05.txt'
        #    if os.path.exists(path)==True: self.isoModel = path
        #    else: raise ValueError('Fermitools appears to be setup, but cannot find diffuse model at path ' + path + '.  Please download or specify an alternate path in Scan.isoModel.' ) 
        # Check if lat and longitude are mixed up.
        if (self.metric=='spherical' and (np.max(mcSims[0]>90) or np.min(mcSims[1])<0)): raise ValueError("Invalid spherical coordinates.  Double check that input is (lat,long,t,E)")
            
        # Check that input shape is correct
        if len(mcSims)!=4: 
            raise ValueError("Invalid Input.  Requires input array of shape (4,n) corresponding to (B,L,T,E) or (X,Y,T,E)")
        
        # If specified energies, clip out events outside the energy range specified
        Low,High = np.ones(len(mcSims[0])),np.ones(len(mcSims[0]))
        if self.eMin!=-1: Low  = mcSims[3]>self.eMin
        if self.eMax!=-1: High = mcSims[3]<self.eMax
        idx = np.where(np.logical_and(Low,High)==True)[0]
        if len(idx) == 0: raise ValueError('No events within energies specified by eMin and eMax.')
        mcSims = np.transpose(np.transpose(mcSims)[idx])
        
        # If integration time not specified, find min/max in input data
        if self.totalTime==-1:
            self.totalTime = np.max(mcSims[2])-np.min(mcSims[2])
        
        # By default find min/max input energies
        if (self.eMin ==-1 and self.eMax==-1):
            self.eMin, self.eMax = np.min(mcSims[3]),np.max(mcSims[3]) 
        # Check that energies are within range or throw exception.
        if (self.eMin<50 or self.eMax<0 or self.eMin>self.eMax or self.eMax>6e5): 
            raise ValueError('Invalid or unspecified energies eMin/eMax.  Must be between 50 and 6e5 MeV.  If you did not set these, check that energies below this are not included in the input.')
        
        # Find epsilon by default.
        if self.eps==-1:
            self.eps = FermiPSF.GetR68(self.eMin,self.eMax,convType=self.convType, fraction=self.containment)
        
        # Compute the background density if not specified.
        #TODO: make lat/long cuts and specify shorter or greater distance by sign of long1-long2
        #TODO: Add BDT tools
#         if self.bgDensity ==-1:
#             minX,maxX,minY,maxY = min(mcSims[0]),max(mcSims[0]),min(mcSims[1]),max(mcSims[1])
#              if in spherical coords, compute solid angle of pseudo-rectangle
#             if self.metric=='spherical':
#                 (np.sin(np.deg2rad(minX))-np.sin(np.deg2rad(maxX)))*(np.deg2rad
#              if in euclidean, just use normal rectangular area
#             elif self.metric == 'euclidean':
                
        
        #====================================================================
        # Choose temporal search half-height 'a' based on fragmentation limits if not specified.
        #====================================================================
        if self.D==3:
            if self.a==-1 and self.bgDensity!=-1:
                self.a = 3./2.*self.totalTime/(np.pi*self.eps**2*self.bgDensity) # set according to fragmentation limit.
            elif self.a==-1:
                # Initialize the background model
                self.BG = BGTools.BGTools(self.eMin,self.eMax,self.totalTime,self.diffModel, self.isoModel,self.convType)
                # weight latitudes by solid angle
                weights = np.abs(np.cos(np.linspace(-np.pi/4.,np.pi/4,1441)))
                # mask out regions of low latitude (abs(b)<5 deg.)
                start,stop = int(85/180.*1441.), int(95/180.*1441.)
                weights[start:stop] = 0. 
                # Compute Average
                dens = np.mean(np.average(self.BG.BGMap, weights = weights,axis=0))
                self.a = 3./2.*self.totalTime/(np.pi*self.eps**2*dens)# set 'a' according to fragmentation limit.
                if self.output == True: print 'Expected background count is', (np.pi*self.eps**2*dens)/self.totalTime*3.15e7/12, 'events per month.'
            
            elif self.a==-1: raise ValueError('Must specify temporal search radius "a", provide bgDensity, or use nMin method "BGInt" or "BGCenter"')
            if self.output == True: print 'Temporal search half height set to', self.a/3.15e7*12, 'months.'

        #====================================================================
        # Determine the values for nMin depending on the desired method
        #====================================================================
        start=time.time()
        # If isotropic method, we don't care about energy ranges.        
        if self.nMinMethod == 'isotropic':
            if (self.bgDensity <=0): raise ValueError('bgDensity must be > 0 for nMinMethod="isotropic"') # Check Density
            self.nMin=self.bgDensity*np.pi*self.eps**2.  # area of search times bg density
            if self.D==3: self.nMin*=2*float(self.a)/self.totalTime # If 3-d search, rescale by temporal width. 
            # set nMin according to poisson z-score specified by nMinSigma
            self.nMin = self.nMin+self.nMinSigma*np.sqrt(self.nMin)
        if self.nMinMethod=='nMin' and self.nMin<3: raise ValueError('Error: nMin must be >=3 for nMin mode')
        # if not evaluating nMin isotropically or using a custom nMin, need to check energies of input
        else:
            if self.BG==[]:
                # Initialize the background model
                self.BG = BGTools.BGTools(self.eMin,self.eMax,self.totalTime,self.diffModel, self.isoModel,self.convType)
        if self.nMinMethod == 'BGInt':
            # Integrate the background
            self.nMin = self.BG.GetIntegratedBG(l=mcSims[1],b=mcSims[0],A=self.eps,B=self.eps) # Integrate the background template to find nMins
            if self.D==3: self.nMin*=2*float(self.a)/self.totalTime # If 3-d search, rescale by temporal width. 
            # set nMin according to poisson z-score specified by nMinSigma  
            self.nMin = self.nMin+self.nMinSigma*np.sqrt(self.nMin)
        elif(self.nMinMethod == 'BGCenter'):
            # Retreive event density and multiply by eps-neighborhood area
            self.nMin = BG.GetBG(l=mcSims[1],b=mcSims[0])*np.pi*self.eps**2
            if self.D==3: self.nMin*=2*float(self.a)/self.totalTime # If 3-d search, rescale by temporal width. 
            self.nMin = self.nMin+self.nMinSigma*np.sqrt(self.nMin)         
        if self.output==True: print 'Completed Initial Background Integration in', (time.time()-start), 's'
        if self.output==True: print 'Mean nMin' , np.mean(self.nMin)
        sys.stdout.flush()
        
        if (np.mean(self.nMin)<3): print "WARNING: Most points have expected eps-neighborhoods of <3 events.  DBSCAN is less reliable on such sparse data. You may want to increase the size of energy bins."
        
        #====================================================================
        # Compute Clusters Using DBSCAN     
        #====================================================================
        #if self.output==True:print "Beginning DBSCAN..."
        start=time.time()    

        dbscanResults = self.__DBSCAN_THREAD(mcSims)
        if self.output==True:print  'Completed DBSCAN in', (time.time()-start), 's'
        sys.stdout.flush()
        #if self.output==True:print  "Computing Cluster Properties..."
        start=time.time()    
        ClusterResults = self.__Cluster_Properties_Thread([dbscanResults,mcSims])
        if self.output==True: print 'Completed Cluster Properties in', (time.time()-start), 's'

        if (self.fileout != ''): 
            pickle.dump(ClusterResults, open(self.fileout,'wb')) # Write to file if requested
            if self.output==True: print 'Results written to', self.fileout
        self.clusterResults = ClusterResults
        if self.output ==True : print 'Found' , len(np.unique(dbscanResults))-1, 'clusters.'
        return ClusterResults

        #====================================================================
        # Multithreaded versions, purely for reference.
        #====================================================================
        # 
        # Check number to analyze
#         if ((numAnalyze == 0) or (numAnalyze > len(mcSims))):
#             numAnalyze =len(mcSims)
#         # Define methods for mapping
#         
#         
#         DBSCAN_PARTIAL = partial(__DBSCAN_THREAD,  eps=self.eps, min_samples=self.min_samples,timeScale=self.timeScale,nCorePoints = self.nCorePoints,plot=self.plot,indexing=self.indexing,metric=self.metric)
#         if numProcs>1:
#             p = pool.Pool(numProcs) # Allocate thread pool
#             dbscanResults = p.map(DBSCAN_PARTIAL, mcSims[:numAnalyze]) # Call mutithreaded map.
#             p.close()  # Kill pool after jobs complete.  required to free memory.
#             p.join()   # wait for jobs to finish.
#         else:
#         #    # Serial Version.  Only use for debugging
#             dbscanResults = map(DBSCAN_PARTIAL, mcSims[:numAnalyze])
#         #dbscanResults = map(DBSCAN_PARTIAL, mcSims[:numAnalyze])
#         
#         ################################################################
#         # Compute cluster properties for each cluster in each simulation
#         ################################################################
#         PROPS_PARTIAL = partial( __Cluster_Properties_Thread,BGDensity=self.bgDensity,TotalTime=self.totalTime, inner=self.inner,outer=self.outer,sigMethod=self.sigMethod,metric=self.metric)
#         if numProcs>1:
#             p = pool.Pool(numProcs) # Allocate thread pool 
#             ClusterResults = p.map(PROPS_PARTIAL,zip(dbscanResults,mcSims))
#             #ClusterResults = parmap(PROPS_PARTIAL,zip(dbscanResults,mcSims))
#             p.close()  # Kill pool after jobs complete.  required to free memory.
#             p.join()   # wait for jobs to finish.
#         else:
#             ClusterResults = map(PROPS_PARTIAL,zip(dbscanResults,mcSims))
        

    
    ####################################################################################################
    #
    # Internal Methods 
    #
    ####################################################################################################
    def __DBSCAN_THREAD(self,sim):
            """
            Internal Method which calls DBSCAN. seperate for multithreading (not implemented), although must be top-level function for that.
            @param sim numpy.ndarray of shape (3+,n) where the first three elements are (lat,long,T)
            @return Labels corresponding to the input points. 
            """
            X = np.transpose(sim[0:3])
            return DBSCAN.RunDBScan3D(X, self.eps, nMin=self.nMin, a=self.a, nCorePoints=self.nCorePoints, plot=self.plot,indexing=self.indexing,metric=self.metric,D=self.D, output=self.output)      
    
    def __Cluster_Properties_Thread(self,input):
        """Internal Method which computes manages computation of various cluster properties.
        @param input A tuple (labels,sim) where labels are the return of __DBSCAN_THREAD() and sim is the original input (b,l,T,E).
        @return
        """
        labels,sim = input
        idx=np.where(labels!=-1)[0] # ignore the noise points
        clusters = np.unique(labels[idx]).astype(int) # find all the unique cluster labels
        CRLabels = np.array(labels).astype(int)
    
        # Some beurocracy because of way numpy array typecast handles things
        numClusters = len(np.unique(clusters))
        if numClusters==0: # no clusters 
            CR = ClusterResult.ClusterResult(Labels=[], Coords=[], 
                                         CentX=[]    , CentY=[]    , CentT=[], 
                                         Sig95X=[]  , Sig95Y=[]  , Sig95T=[], 
                                         Size95X=[], Size95Y=[], Size95T=[], 
                                         MedR=[]      , MedT=[],
                                         Members=[], Sigs=[], 
                                         SigsMethod=[], NumClusters=0,PA=[])  # initialize new cluster results object
            return CR
        elif numClusters==1: clusters = [clusters,]# if one, we need to make into a list so we can iterate over it.
        CRNumClusters = numClusters # Number of clusters found in the simulation     
        
        CRCoords = [self.__Get_Cluster_Coords(sim, labels, cluster) for cluster in clusters] # contains coordinate triplets for each cluster in clusters.

        # Compute sizes and centroids
        if self.metric=='euclidean':
            CRSize95X, CRSize95Y, CRSize95T, CRPA, CRMedR, CRMedT, CRCentX,CRCentY,CRCentT,CRSig95X,CRSig95Y,CRSig95T,CRSig95R, CRe, CRDens33, CRDens66, CRDens100 = np.transpose([self.__Cluster_Size(CRCoords[cluster]) for cluster in range(len(clusters))])
        elif self.metric=='spherical':
            CRSize95X, CRSize95Y, CRSize95T, CRPA, CRMedR, CRMedT, CRCentX,CRCentY,CRCentT,CRSig95X,CRSig95Y,CRSig95T, CRSig95R, CRe, CRDens33, CRDens66, CRDens100 = np.transpose([self.__Cluster_Size_Spherical(CRCoords[cluster]) for cluster in range(len(clusters))])
        else: raise ValueError('Invalid metric: ' + str(self.metric))
        
            
        CRMembers = np.array([len(CRCoords[cluster]) for cluster in range(len(clusters))]) # count the number of points in each cluster.
        # Input into cluster results instance
        CR = ClusterResult.ClusterResult(Labels=np.array(CRLabels), Coords=CRCoords, 
                                         CentX=CRCentX    , CentY=CRCentY    , CentT=CRCentT, 
                                         Sig95X=CRSig95X  , Sig95Y=CRSig95Y  , Sig95T=CRSig95T, 
                                         Size95X=CRSize95X, Size95Y=CRSize95Y, Size95T=CRSize95T, 
                                         MedR=CRMedR      , MedT=CRMedT,
                                         Members=CRMembers, Sigs=[], Sig95R=CRSig95R, e=CRe, Dens33=CRDens33, Dens66=CRDens66, Dens100=CRDens100,
                                         SigsMethod=self.sigMethod, NumClusters=CRNumClusters,PA=CRPA)  # initialize new ClusterResults instance
        # If two dimensional, set the size95T to half  the total exposure time so that significance is computed correctly.
        if self.D==2: CR.Size95T = self.totalTime/2.*np.ones(len(CR.Size95T))
        #----------------------------------------------------
        # Compute significances
        #----------------------------------------------------
        if self.sigMethod == 'isotropic':
            CR.Sigs  = np.array([self.__Compute_Cluster_Significance_3d_Isotropic(CR.Coords[cluster],CR.Size95X[cluster],CR.Size95Y[cluster], CR.Size95T[cluster]) for cluster in range(len(clusters))])
        elif self.sigMethod =='annulus':
            CR.Sigs   = np.array([self.__Compute_Cluster_Significance_3d_Annulus(CR.Coords[cluster], np.transpose(sim),CR.Size95X[cluster],CR.Size95Y[cluster], CR.Size95T[cluster], CR.CentX[cluster],CR.CentY[cluster]) for cluster in range(len(clusters))])
        elif self.sigMethod == 'BGInt':
            CR.SigsMethod ='BGInt'
            CR.Sigs = np.array(self.BG.SigsBG(CR))
        else: raise ValueError('Invalid significance evaluation method: ' + str(self.sigMethod))

        return CR
        
    
    
    def __Get_Cluster_Coords(self,sim,labels, cluster_index):
        """Returns a set of coordinate triplets for cluster 'cluster_index' given input vectors [X (1xn),Y(1xn),T(1xn)] in sim, and a set of corresponding labels"""
        idx = np.where(labels==cluster_index)[0] # find indices of points which are in the given cluster
        return np.transpose(np.array(sim))[:][idx] # select out those points and return the transpose (which provides (x,y,t) triplets for each point 
    
    
    def __Cluster_Size(self,cluster_coords):
        """Returns basic cluster properties, given a set of cluster coordinate triplets""" 
        X,Y,T,E = np.transpose(cluster_coords)
        CentX0,CentY0,CentT0 = np.mean(X),np.mean(Y), np.mean(T)
        X, Y = X-CentX0, Y-CentY0 
        r_all = np.sqrt((x-CentX0)**2 + (y-CentY0)**2)
        CentX0, CentY0 = np.average(x,weights=1/r_all),np.average(y,weights=1/r_all)
        r_all = np.sqrt((x-CentX0)**2 + (y-CentY0)**2 )
        SIG95R = 2.*np.std(r_all/np.sqrt(len(X)))
        
        # Singular Value Decomposition
        U,S,V = la.svd((X,Y))
        # Rotate to align x-coord to principle component
        x = U[0][0]*X + U[0][1]*Y
        y = U[1][0]*X + U[1][1]*Y
        
        # Compute weighted average and stdev in rotated frame
        weights = np.divide(1,np.sqrt(np.square(x)+np.square(y))) # weight by 1/r 
        CentX, CentY = np.average(x,weights=weights), np.average(y,weights=weights)
        SigX,SigY = np.sqrt(np.average(np.square(x-CentX), weights=weights)), np.sqrt(np.average(np.square(y-CentY), weights=weights))
        # Find Position Angle
        xref,yref = np.dot(U,[0,1])
        theta = -np.rad2deg(np.arctan2(xref,yref))
        POSANG = theta+90.
        
        CentX ,CentY = np.dot(la.inv(U),(CentX,CentY)) #Translate the updated centroid into the original frame
        CENTX,CENTY =  CentX+CentX0,CentY+CentY0          # Add this update to the old centroids
        SIG95X,SIG95Y = 2*SigX/np.sqrt(np.shape(x)[0]),2*SigY/np.sqrt(np.shape(x)[0]) 
        SIZE95X, SIZE95Y = 2*S/np.sqrt(len(X)) 
        
        r = np.sqrt(np.square(X-CentX)+np.square(Y-CentY))  # Build list of radii from cluster centroid
        SIG95T = np.std(T)/np.sqrt(np.shape(r)[0])
        dt = np.abs(T-CentT0)
        countIndexT = int(np.ceil(0.95*np.shape(dt)[0]-1))
        SIZE95T = np.sort(dt)[countIndexT]   # choose the radius at this index
        
        e = np.sqrt(1-SIZE95Y**2/SIZE95X**2)
        val,bins = np.histogram(r_all, bins=np.linspace(0,SIZE95X,4))
        Dens33, Dens66, Dens100 = val/float(len(r_all))
    
        return SIZE95X, SIZE95Y,SIZE95T, POSANG, np.median(r), np.median(dt), CENTX,CENTY,CentT0,SIG95X,SIG95Y,SIG95T, SIG95R, e, Dens33, Dens66, Dens100
    
    def __Cluster_Size_Spherical(self,cluster_coords):
        """Returns basic cluster properties, given a set of cluster coordinate triplets"""
        X,Y,T,E = np.transpose(cluster_coords)
        # Map to cartesian
        X, Y = np.deg2rad(X), np.deg2rad(Y)
        x = np.cos(X) * np.cos(Y)
        y = np.cos(X) * np.sin(Y)
        z = np.sin(X)
    
        # Compute Cartesian Centroids
        CentX0,CentY0,CentZ0,CentT0 = np.mean(x),np.mean(y), np.mean(z), np.mean(T)
        r = np.sqrt(CentX0**2 + CentY0**2 + CentZ0**2)
        # Compute distance from zeroth order centroid for all points
        r_all = np.sqrt((x-CentX0)**2 + (y-CentY0)**2 + (z-CentZ0)**2)
        
        # Sort list and determine radius containing 6/7 of points.
        sorted_r = np.sort(r_all)
        ref_rad  = sorted_r[np.ceil(6./7*len(sorted_r))]
        
        idx = np.where(r_all<=ref_rad)[0]
        
        CentX0, CentY0, CentZ0 = np.average(x[idx],weights=1./r_all[idx]),np.average(y[idx],weights=1./r_all[idx]),np.average(z[idx],weights=1./r_all[idx])
        r_all = np.sqrt((x-CentX0)**2 + (y-CentY0)**2 + (z-CentZ0)**2)
        SIG95R = 2.*np.std(r_all[idx]/np.sqrt(6./7.*len(x)))
        
        # Rotate away Z components so we are in the x,z plane and properly oriented with galactic coordinates
        def rotation_matrix(axis,theta):
            axis = axis/np.sqrt(np.dot(axis,axis))
            a = np.cos(theta/2.)
            b,c,d = -axis*np.sin(theta/2.)
            return np.array([[a*a+b*b-c*c-d*d, 2*(b*c-a*d), 2*(b*d+a*c)],
                             [2*(b*c+a*d), a*a+c*c-b*b-d*d, 2*(c*d-a*b)],
                             [2*(b*d-a*c), 2*(c*d+a*b), a*a+d*d-b*b-c*c]])
        # Pick vector pointing from origin to centroid of cluster
        n = np.array([CentX0, CentY0,CentZ0])/r
        # normalize to 1
        n = n/np.sqrt(np.dot(n,n))
        # z unit vector pointing at north celestial pole
        nz = np.array([0.,0.,1.])
        # Check that the normal vector isn't equal to +-90 or we will get a singularity later
        # if we are ok, then choose a rotation axis in the xy plane which will rotate away the z component.
        if (n!=nz).all() and (n!=-nz).all() : axis = np.cross(nz,n)  
        else: axis = np.array([0,1,0]) #if at one of the poles, choose any axis.
        # Find the rotation angle in radians
        theta = np.pi/2.-np.arccos(np.dot(nz,n))
        
        R1 = rotation_matrix(axis,-theta) # construct the rotation matrix
        # Find the second rotation angle which will center the cluster at (X,Y,Z)=(0,1,0)  
        dot = np.dot(R1,n)
        theta2 = (np.pi/2-np.arccos(dot[0]))
        if dot[1]>0: theta2*=-1  # need to handle a sign issue
        # construct second rotation matrix
        R2 = rotation_matrix(nz,theta2)
        R  = np.dot(R2,R1) # construct full rotation matrix 
        def rotate(n):
            n = n/np.sqrt(np.dot(n,n))
            return np.dot(R,n)
    
        # rotate all the vectors (Y component should be zero for all)
        X,Z,Y = np.rad2deg(np.transpose([rotate(np.transpose((x,y,z))[i]) for i in range(len(x))]))
        
        # DEBUG
        #from matplotlib import pyplot as plt
        #plt.figure(0)
        #plt.scatter(X,Y)
        #plt.axis('equal')
        #plt.show()  
        
        # Convert Centroids back to lat/long in radians
        CentY0 = (np.rad2deg(np.arctan2(CentY0, CentX0)) + 360.)%360
        CentX0 = np.rad2deg(np.arcsin(CentZ0/r))
         
        # Singular Value Decomposition
        U,S,V = la.svd((X,Y))
        
        # Rotate to align x-coord to principle component
        x = U[0][0]*X + U[0][1]*Y
        y = U[1][0]*X + U[1][1]*Y
        
        # Compute weighted average and stdev in rotated frame
        weights = np.divide(1,np.sqrt(np.square(x)+np.square(y))) # weight by 1/r 
        CentX, CentY = np.average(x,weights=weights), np.average(y,weights=weights)
        CentX,CentY =     CentX0, CentY0
        SigX,SigY = np.sqrt(np.average(np.square(x-CentX), weights=weights)), np.sqrt(np.average(np.square(y-CentY), weights=weights))
        
        # Find Position Angle
        xref,yref = np.dot(U,[0,1])
        theta = -np.rad2deg(np.arctan2(xref,yref))
        POSANG = theta+90.
        
       
        #CentX ,CentY = np.dot(la.inv(U),(CentX,CentY)) #Translate the updated centroid into the original frame
        #CENTX,CENTY =  CentX+CentX0,CentY+CentY0          # Add this update to the old centroids
        SIG95X,SIG95Y = 2*SigX/np.sqrt(np.shape(x)[0]),2*SigY/np.sqrt(np.shape(x)[0]) 
        SIZE95X, SIZE95Y = 2*S/np.sqrt(len(X)) 
        #r = np.sqrt(np.square(X-CENTX)+np.square(y-CENTY))  # Build list of radii from cluster centroid
        r = np.sqrt(np.square(X-CentX0)+np.square(y-CentY0))  # Build list of radii from cluster centroid
        
        SIG95T = np.std(T)/np.sqrt(np.shape(r)[0])
        dt = np.abs(T-CentT0)
        countIndexT = int(np.ceil(0.95*np.shape(dt)[0]-1))
        SIZE95T = np.sort(dt)[countIndexT]   # choose the radius at this index
    
        e = np.sqrt(1-SIZE95Y**2/SIZE95X**2)
        val,bins = np.histogram(r_all, bins=np.linspace(0,SIZE95X,4))
        Dens33, Dens66, Dens100 = val/float(len(r_all))
        
    
        return SIZE95X, SIZE95Y,SIZE95T, POSANG, np.median(r), np.median(dt), CentX0,CentY0,CentT0,SIG95X,SIG95Y,SIG95T, SIG95R, e, Dens33, Dens66, Dens100
    
    
    def __Compute_Cluster_Significance_3d_Isotropic(self, X, Size95X,Size95Y, Size95T):
        """
        Computes the cluster significance using assuming an isotropic background. 
        @param X Coordinate triplets lat/long/time of length 'cluster members'
        @param Size95T from ClusterResults object            
        @returns significance
        """
        # Default to zero significance
        if (len(X)==1):return 0
        # Otherwise.......
        x,y,t,E = np.transpose(X) # Reformat input
        centX,centY,centT = np.mean(x), np.mean(y),np.mean(t) # Compute Centroid
        r = np.sqrt(  np.square(x-centX) + np.square(y-centY) ) # Build list of radii from cluster centroid
        countIndex = int(np.ceil(0.95*np.shape(r)[0]-1)) # Sort the list and choose the radius where the cumulative count is >95% 
        clusterRadius = np.sort(r)[countIndex]   # choose the radius at this index 
        N_bg = np.pi * Size95X*Size95Y * self.bgDensity # Use isotropic density to compute the background expectation
        dT = 2*Size95T/float(self.totalTime) # Rescale according to the total time.
        ######################################################
        # Evaluate significance as defined by Li & Ma (1983).  N_cl corresponds to N_on, N_bg corresponds to N_off
        if dT > .01: 
            N_bg,N_cl = N_bg*dT, countIndex
            if N_cl/(N_cl+N_bg)<1e-20 or N_bg/(N_cl+N_bg)<1e-20:
                return 0
            S2 = 2.0*(N_cl*np.log(2.0*N_cl/(N_cl+N_bg)) + N_bg*np.log(2.0*N_bg/(N_cl+N_bg)))
            if S2>0.0:
                return np.sqrt(S2)   
            else:
                return 0
        else: return 0
        
        
        
    def __Compute_Cluster_Significance_3d_Annulus(self,X_cluster,X_all,Size95X,Size95Y,Size95T,CentX0,CentY0):
        """
        Takes input list of coordinate triplets for the cluster and for the entire simulation and computes the cluster size.
        Next, the background level is computed by drawing an annulus centered on the the cluster with inner and outer radii 
        specified as a fraction of the initial radius.  Then the significance is calculated.  The cluster is cylindrical
        with the axis aligned temproally.  Similarly, the background annulus is taken over a cylindrical shell and is 
        computed over the range of times in X_all (thus if the background is time varying, this will average that).  
        
        @param X_cluster A tuple containing a coordinate triplet (x,y,z) for each point in a cluster.  
        @param X_all     A tuple containing coordinate triplets for background events, typically just all events.
        @param Size95T from ClusterResults object            
        @returns significance
            
        return:
            Cluster significance from Li & Ma (1983)
        """
        #TODO: Update annulus method to ellipse calculation instead of spherical.     
        # Default to zero significance
        if (len(X_cluster)==1):return 0
        # Otherwise.......
        x,y,t,e = np.transpose(X_cluster) # Reformat input
        x_all,y_all,t_all,e_all = np.transpose(X_all) # Reformat input
        
        
        if self.metric=='euclidean':
            ################################################################
            # Estimate the background count
            AnnulusVolume = np.pi* ((self.outer*Size95X)**2 -(self.inner*Size95X)**2)*(self.totalTime)
            r_all = np.sqrt(np.square(x_all-CentX0)+np.square(y_all-CentY0)) # compute all points radius from the centroid. 
            r_cut = np.logical_and(r_all>Size95X*self.inner,r_all<Size95X*self.outer)# Count the number of points within the annulus and find BG density
            idx =  np.where(r_cut==True)[0] # pick points in the annulus
            BGDensity = np.shape(idx)[0]/AnnulusVolume # Density = counts / annulus volume 
        if self.metric=='spherical':
            dPhi = np.deg2rad(x_all-CentX0) # lat 
            dLam = np.deg2rad(y_all-CentY0) # lon
            
            x_all = np.deg2rad(x_all)
            # Distances using Vincenty's formula for arc length on a great circle.
            cos = np.cos
            sin = np.sin
            d = np.arctan2(np.sqrt( np.square(cos(x_all)*sin(dLam) ) + np.square(cos(CentX0)*sin(x_all)-sin(CentX0)*cos(x_all)*cos(dLam)) ) , sin(CentX0)*sin(x_all)+cos(CentX0)*cos(x_all)*cos(dLam) )
            gt = np.rad2deg(d)>self.inner*Size95X        
            lt = np.rad2deg(d)<self.outer*Size95X
            cnt = np.count_nonzero(np.logical_and(gt,lt))
            x_all = np.rad2deg(x_all)
            AnnulusVolume = np.pi* ((self.outer*Size95X)**2 -(self.inner*Size95X)**2)*(self.totalTime)            
            BGDensity = float(cnt) / AnnulusVolume            
        
        ######################################################
        # Evaluate significance as defined by Li & Ma (1983).  N_cl corresponds to N_on, N_bg corresponds to N_off
        N_bg = np.pi * Size95X*Size95Y * 2*Size95T* BGDensity # BG count = cluster volume*bgdensity
        N_cl = 0.95*len(x) # Number of counts in cluster.
        # Ensure log args are greater than 0.
        if N_cl/(N_cl+N_bg) <= 0 or N_bg/(N_cl+N_bg) <= 0: return 0 
        S2 = 2.0*(N_cl*np.log(2.0*N_cl/(N_cl+N_bg)) + N_bg*np.log(2.0*N_bg/(N_cl+N_bg)))
        if S2>0.:
            return np.sqrt(S2)   
        else:
            return 0.

def PlotGal(b,l,fname='',figsize=(6,4),s=0.01,**kwargs):
    """
    Helper function which requires the matplotlib toolkit 'basemap' to be installed.  Will plot galactic coordinates appropriately.
    @param l Galactic latitude vector.
    @param b Galactic latitude vector.
    @param fname Save plot to the path fname.  Note that if this is a large scatter plot, one should use a rasterized format like .png instead of .pdf.
    @param fname Save plot to the path fname.  Note that if this is a large scatter plot, one should use a rasterized format like .png instead of .pdf.
    @param figsize matplotlib figure size in inches.  Default = (6,4)
    @param ms marker size for scatter plot.  Default is 0.01 pt.
    @returns (plt,m) Pyplot instance and basemap instance
    """
    try:
        from matplotlib import pyplot as plt
        from mpl_toolkits.basemap import Basemap
    except:
        raise ImportError('It appears that the basemap package is not installed.  Please see instructions at http://matplotlib.org/basemap/users/installing.html')
    
    plt.figure(0,figsize=(10,10))
    m = Basemap(projection='hammer',lon_0=0)
    m.drawmapboundary(fill_color='#FFffff')
    x, y = m(l,b)   
    plt.scatter(x,y,s=s,**kwargs)
    if fname!='': plt.savefig(fname)
    return plt,m

def PlotClusters(CR,**kwargs):
    """
    Plots clusters from a cluster results object.
    @param CR A ClusterResults object
    @param **kwargs Keyword arguments to pass to pyplot.scatter()
    """
    master = [[],[],[],[]]
    for coords in CR.Coords:
        master = np.append(master,np.transpose(coords),axis=1)
    b,l,t,e = master
    PlotGal(b,l,**kwargs)

def PlotSpectrum(E,bins=30,**kwargs):
    """
    Plot the energy spectrum between the lowest and highest values using log-spaced bins.
    @param E Vector of energies
    @param bins Number of log-spaced bins.
    @returns Pyplot instance
    """
    # log space bins
    bins=np.logspace(np.log10(min(E)),np.log10(max(E)),bins)
    
    counts, bins = np.histogram(E,bins=bins)
    # need to compute dN/dE so divide by bin width.
    values = [counts[i]/(bins[i+1]-bins[i]) for i in range(len(counts))]
    # find center of bins
    bins = [0.5*(bins[i]+bins[i+1]) for i in range(len(counts))]
    plt.step(bins, values,**kwargs)
    plt.yscale('log')
    plt.xscale('log')
    plt.xlabel('Energy [MeV]')
    plt.ylabel(r'dN/dE [MeV$^{-1}$]')
    return plt

def Mean_Significance(ClusterResults):
        """Given a set of ClusterResults, calculates the mean of the "mean significance weighted by number of cluster members" """
        return np.mean([np.ma.average(CR.Sigs, weights=CR.Members) for CR in ClusterResults])
    
def Mean_Radius(ClusterResults):
    """Given a set of ClusterResults, calculates the mean of the "mean Radius weighted by Members" """
    return np.mean([np.ma.average(CR.Size95X, weights=CR.Members) for CR in ClusterResults])    

def Mean_SizeT(ClusterResults):
    """Given a set of ClusterResults, calculates the mean of the "mean temporal size weighted by Members" """
    return np.mean([np.ma.average(CR.Size95T, weights=CR.Members) for CR in ClusterResults])        

def Mean_SizeSigR(ClusterResults):
    """Given a set of ClusterResults, calculates the mean of the "stdev of radius from centroid weighted by Members" """
    return np.mean([np.ma.average(CR.MedR, weights=CR.Members) for CR in ClusterResults]) 

def Mean_SizeSigT(ClusterResults):
    """Given a set of ClusterResults, calculates the mean of the "stdev temporal dist from centroid weighted by Members" """
    return np.mean([np.ma.average(CR.MedT, weights=CR.Members) for CR in ClusterResults])
        
def Mean_Members(ClusterResults):
    """Given a set of ClusterResults, calculates the mean of the "mean number of cluster members weighted by Members" """
    return np.mean([np.ma.average(CR.Members, weights=CR.Sigs) for CR in ClusterResults])   
        
def Mean_Clusters(ClusterResults,sig_cut=0.):
    """Given a set of ClusterResults, calculates the mean number of detected clusters with significance greater than sig_cut"""
    return np.mean([np.count_nonzero(CR.Sigs >= sig_cut)  for CR in ClusterResults])


