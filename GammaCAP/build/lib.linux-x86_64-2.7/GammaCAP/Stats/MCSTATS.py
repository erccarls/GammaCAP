"""@package MCSTATS
Main package for computing cluster statistics and performing DBSCAN operation.

@author: Eric Carlson
"""

import matplotlib.pyplot as plt  #@UnresolvedImport
import cPickle as pickle
import numpy as np
import scipy.cluster as cluster
from matplotlib.backends.backend_pdf import PdfPages
import DBSCAN
from GammaCAP.BGTools import BGTools
import multiprocessing as mp
from multiprocessing import pool
from functools import partial
import ClusterResult
import scipy.linalg as la
import time
import pyfits

class Scan:
    """
    Class for setting up scan, computing DBSCAN clusters, and properties given event data.
    """
    def __init__(self, eps=-1. ,nMinMethod='BGInt', a = -1., nMinSigma=5. ,nCorePoints = 3, 
                 nMin = -1 , sigMethod ='BGInt', bgDensity=-1, TotalTime=-1,inner=1.25,outer=2.0, 
                 fileout = '',numProcs = 1, plot=False,indexing=True,metric='spherical',eMax=-1,eMin=-1,
                 output = True, 
                 diffModel = '/home/carlson/fermitools/refdata/fermi/galdiffuse/gll_iem_v05.fits',
                 isoModel  = '/home/carlson/fermitools/refdata/fermi/galdiffuse/isotrop_4years_P7_v9_repro_clean_v1.txt'):
        """
        Initialize DBSCAN.  Most properties are specific to certain settings and all should be left default unless used.
        """
        ##@var a
        #The DBSCAN3d temporal search half-height.  Unused for 2d scans.\n
        #Values:  \n
        #    -1 (DEFAULT): Uses the mean counts for the entire input simulation to compute the background density and then sets a as low as possible within the fragmentation limits.\n 
        #    float>0: Specified in seconds.
        self.a           = float(a)
        ##@var eps
        #The DBSCAN3d spatial search radius.\n
        #Values:  \n
        #    -1 (DEFAULT): Averages the Fermi 68% containment radius over the input energy ranges (by default) or ranges specified by eMin and eMax.\n  
        #    float>0: Specified in degrees.     
        self.eps         = float(eps)
        ##@var nMinMethod 
        #Specifies which method to use when evaluating the background count for DBSCAN.\n
        #Values:\n
        #    'BGInt' (DEFAULT) : This is the best approach for FermiData, but for very large numbers of points (>1e6) can be computationally inefficient\n
        #    'BGCenter'        : Same as above, but just samples the center value of the BG template instead of integrating.  Much more efficient.\n
        #    'isotropic'       : One must specify bgDensity as well which is multiplied by the DBSCAN search volume to obtain the background rate.\n
        #    'nMin'            : Specify the value of nMin yourself, instead of using nMinSigma (not very useful for varying background)
        self.nMinMethod  = nMinMethod
        ##@var nMinSigma
        #Z-score over mean background density to calculate the nMin parameter of DBSCAN.  Unused if nMinMethod='nMin'.\n
        #Values:\n
        #float>0 (DEFAULT 5).
        self.nMinSigma   = float(nMinSigma)
        ##@var nMin
        # DBSCAN parameter for the minimum number of events in an eps-neighborhood before becoming a core point (Unused by default). If nMinMethod='nMin', this specifies the value to use.\n
        # Values:\n
        # int>0: For uniform nMin.\n
        # numpy.ndarray shape(n,1): Specify an array of positive integers where n is the number of input samples.  
        self.nMin        = int(nMin)
        ##@var metric
        # Specify the metric of the input data.
        # Values:\n
        # 'spherical' (DEFAULT):  This should be used for any real data in galactic coordinates\n
        # 'euclidean' : If simulating data it can be easier to use euclidean coordinates.
        self.metric      = metric
        ##@var diffModel
        # Absolute path to the galactic diffuse model. (typically '$FERMI_DIR/refdata/fermi/galdiffuse/gll_iem_v05.fits') where $FERMI_DIR is the Fermi science tools installation path.
        self.diffModel   = diffModel
        ##@var isoModel
        # Absolute path to the diffuse isotropic model. (typically '$FERMI_DIR/refdata/fermi/galdiffuse/isotrop_4years_P7_v9_repro_clean_v1.txt') where $FERMI_DIR is the Fermi science tools installation path.
        self.isoModel    = isoModel
        self.clusterResults = []
        ##@var nCorePoints
        #Clusters must have at least nCorePoints members to be valid. default=3 (i.e. this is low threshold for the number of events).\n
        #Values:\n
        #int>2 (DEFAULT=3)
        self.nCorePoints = int(nCorePoints)    
        ##@var sigMethod
        #Specifies the method for evaluating the expected background count (thus significance) of a cluster.\n  
        #Values:\n
        #'BGInt' (DEFAULT): Integrate the background template over the cluster ellipse to determine local background density. **note**:currently integrates a square circumscribed by the semimajor axis to find density and multiplies by ellipse area.\n
        #'annulus': Computes count density in an annulus of radii ('inner','outer')*semimajor-axis centered on the cluster centroid.        \n
        #'isotropic': Uses bgDensity*ellipse-area to determine the background count.\n
        self.sigMethod   = sigMethod
        ##@var bgDensity
        # Specifies the mean background density (Unused unless (sigMethod or nMinMethod)='isotropic').\n
        # Values:\n
        # float>0 (DEFAULT -1): If -1 computes area spanned by input and divides by total number of photons.  Else specify in photons/deg^2.
        self.bgDensity   = bgDensity
        ##@var totalTime
        # The total exposure or simulation time being used in seconds.\n
        # Values: \n
        # float>1 (DEFAULT -1): If -1, use max-min time of input. Best option as long as data not extremely sparse.  Else specified in seconds
        self.totalTime   = TotalTime
        ##@var inner
        # Inner fractional radius of the annulus. Unused unless sigMethod='annulus'.\n
        # The inner radius used to evaluate the background count expected for a cluster is given by inner*Semimajor-axis in degrees.
        # Values: \n
        # float>0 (DEFAULT 1.25)
        self.inner       = float(inner)
        ##@var outer
        # Outer fractional radius of the annulus. Unused unless sigMethod='annulus'.\n
        # The outer radius used to evaluate the background count expected for a cluster is given by inner*Semimajor-axis in degrees.
        # Values: \n
        # float>0 (DEFAULT 2)
        self.outer       = float(outer)
        ##@var fileout
        # If specified, ClusterResult is written (via cpickle) to file at this path. 
        self.fileout     = ''
        ##@var numProcs
        # For multithreaded portions, specifies the number of processors to use (currently not implemented).
        self.numProcs    = int(numProcs) 
        ##@var plot
        # If True, scatter plot the DBSCAN results with clusters color-coded and noise in black (DEFAULT=False).
        self.plot        = plot
        ##@var indexing
        # If True, use grid-based indexing to speedup cluster computations (default=True)\n
        # Can slightly improve speed if processing many simulations with <2k points each.  Otherwise leave True 
        self.indexing    = bool(indexing)
        ##@var eMin
        # Specifies the minimum energy of the input in MeV (autodetected by default, but can be specified)
        self.eMin = float(eMin)
        ##@var eMax
        # Specifies the maximum energy of the input in MeV (autodetected by default, but can be specified)
        self.eMax = float(eMax)
        #@var output
        #If False, supresses progress output.
        self.output = output
        
        
    def Compute_Clusters(self, mcSims):
        '''
        Main DBSCAN cluster method.  Input a list of simulation outputs and output a list of clustering properties for each simulation.
        Inputs:
            mcSims: this is the output from the runMC methods (i.e. the list of simulation outputs) (python object not filename)
            eps: DBSCAN epsilon parameter
            min_samples: min num points in epislon neighborhood for the DBSCAN algorithm.
        Optional Inputs:
            nCorePoints=3: After DBSCAN is run, there must be at least this many points for a cluster to not be thrown out.
            numAnalyze=0 : number of simulations to analyze out of list.  default is 0 which analyzes all simulations passed
            fileout=''   : if not empty string, store all the clustering info in a pickle file.
            indexing     : if None, automatically choose fastest method. True, always uses grid index, if False always computes full distance matrix and requires much more memory.
            metric       : 'euclidean' or 'spherical' default is euclidean
        Returns:
            dbScanResults: a tuple (labels) for each simulation
                clusterReturn: For each cluster, a list of points in that cluster
                labels: A list of points with cluster identification number.  -1 corresponds to noise.
                        NOTE: ALL BORDER POINTS ARE CONSIDERED NOISE.  See DBSCAN.py for info if you need
                        Border points.
        '''
        
        ###############################################################
        # Compute Clusters Using DBSCAN     
        ###############################################################
        if len(mcSims)!=3:
            print "Invalid Simulation Input.  Requires input array of (3,n) corresponding to (B,L,T) or (X,Y,T)"
        
        if self.output==True:"Error: Beginning Initial Background Integration..."
        start=time.time()
        
        if self.nMinMethod == 'isotropic':  
            if self.bgDensity <=0: 
                print 'Error: Must have BGDensity>0 for nMinMethod isotropic'
                return  
            self.nMin=self.bgDensity*np.pi*self.eps**2.
            self.nMin = self.nMin+self.nMinSigma*np.sqrt(self.nMin)
            
        elif self.nMinMethod == 'BGInt':
            # Integrate the background template to find nMins
            if (self.eMin<50 or self.eMax<0 or self.eMin>self.eMax): print 'Invalid or unspecified energies eMin/eMax.  Must be between 50 and 3e6 MeV.'
            BG = BGTools.BGTools(self.eMin,self.eMax,self.totalTime,self.diffModel, self.isoModel)
            self.nMin = BG.GetIntegratedBG(l=mcSims[1],b=mcSims[0],A=self.eps,B=self.eps)
            self.nMin = self.nMin+self.nMinSigma*np.sqrt(self.nMin)
            
        elif(self.nMinMethod == 'BGCenter'):
            BG = BGTools.BGTools(self.Emin,self.Emax,self.diffModel, self.isoModel)
            self.nMin = BG.GetBG(l=mcSims[1],b=mcSims[0],A=self.eps,B=self.eps)*np.pi*self.eps**2
            self.nMin = self.nMin+self.nMinSigma*np.sqrt(self.nMin)
        
        if self.nMinMethod=='nMin' and self.nMin<3: print 'Error: nMin must be >=3 for nMin mode' 
        

        
        if self.output==True: 'Completed Initial Background Integration in', (time.time()-start), 's'
        
        if self.output==True:"Beginning DBSCAN..."
        start=time.time()    
        dbscanResults = self.__DBSCAN_THREAD(mcSims, eps=self.eps, min_samples=self.nMin,timeScale=self.a,nCorePoints = self.nCorePoints,plot=self.plot,indexing=self.indexing,metric=self.metric)
        if self.output==True: 'Completed DBSCAN and Cluster Statistics in', (time.time()-start), 's'
        
        if self.output==True:"Computing Cluster Properties..."
        start=time.time()    
        ClusterResults = self.__Cluster_Properties_Thread([dbscanResults,mcSims], BGDensity=self.bgDensity,TotalTime=self.totalTime, inner=self.inner,outer=self.outer,sigMethod=self.sigMethod,metric=self.metric)
        if self.output==True: 'Completed Cluster Properties in', (time.time()-start), 's'
        
        if (self.fileout != ''): 
            pickle.dump(ClusterResults, open(self.fileout,'wb')) # Write to file if requested
            if self.output==True: 'Results written to', self.fileout
        self.clusterResults = ClusterResults
        return ClusterResults
        
        
        # Multithreaded versions
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
            X = np.transpose([sim[0],sim[1],sim[2]])
            return DBSCAN.RunDBScan3D(X, self.eps, nMin=self.nMin, a=self.a, N_CorePoints=self.nCorePoints, plot=self.plot,indexing=self.indexing,metric=self.metric)      
    
    def __Cluster_Properties_Thread(self,input):
        labels,sim = input
        idx=np.where(labels!=-1)[0] # ignore the noise points
        clusters = np.unique(labels[idx]).astype(int)
    #    clusters   = np.array(np.int_(np.unique(labels)[1:])) # want to ignore the -1 for noise so ignore first element
        CRLabels = np.array(labels)
    
        # Some beurocracy because of way numpy array typecast handles things
        arrlen = len(np.unique(labels))
        if 0 not in labels: # no clusters 
            CR = ClusterResult.ClusterResult(Labels=[], Coords=[], 
                                         CentX=[]    , CentY=[]    , CentT=[], 
                                         Sig95X=[]  , Sig95Y=[]  , Sig95T=[], 
                                         Size95X=[], Size95Y=[], Size95T=[], 
                                         MedR=[]      , MedT=[],
                                         Members=[], Sigs=[], 
                                         SigsMethod=[], NumClusters=[],PA=[])  # initialize new cluster results object
            CRNumClusters = 0 # Number of clusters found in the simulation
            return CR
        elif arrlen != 2: CRNumClusters = np.array(np.shape(clusters))[0] # Number of clusters found in the simulation
        elif arrlen == 2: 
            CRNumClusters = 1 # Number of clusters found in the simulation
            CRLabels, clusters = [np.array(labels),], [clusters,]
        
        CRCoords = [self.__Get_Cluster_Coords(sim, labels, cluster) for cluster in clusters] # contains coordinate triplets for each cluster in clusters.

        # Compute sizes and centroids
        if self.metric=='euclidean':
            CRSize95X, CRSize95Y, CRSize95T, CRPA, CRMedR, CRMedT, CRCentX,CRCentY,CRCentT,CRSig95X,CRSig95Y,CRSig95T = np.transpose([self.__Cluster_Size(CRCoords[cluster]) for cluster in range(len(clusters))])
        elif self.metric=='spherical':
            CRSize95X, CRSize95Y, CRSize95T, CRPA, CRMedR, CRMedT, CRCentX,CRCentY,CRCentT,CRSig95X,CRSig95Y,CRSig95T = np.transpose([self.__Cluster_Size_Spherical(CRCoords[cluster]) for cluster in range(len(clusters))])
        else: print 'Invalid metric: ' , str(self.metric)
        
        
        
        
        CRMembers = np.array([len(CRCoords[cluster]) for cluster in range(len(clusters))]) # count the number of points in each cluster.
        # Input into cluster results instance
        CR = ClusterResult.ClusterResult(Labels=np.array(CRLabels), Coords=CRCoords, 
                                         CentX=CRCentX    , CentY=CRCentY    , CentT=CRCentT, 
                                         Sig95X=CRSig95X  , Sig95Y=CRSig95Y  , Sig95T=CRSig95T, 
                                         Size95X=CRSize95X, Size95Y=CRSize95Y, Size95T=CRSize95T, 
                                         MedR=CRMedR      , MedT=CRMedT,
                                         Members=CRMembers, Sigs=[], 
                                         SigsMethod=self.sigMethod, NumClusters=CRNumClusters,PA=CRPA)  # initialize new ClusterResults instance
        #----------------------------------------------------
        # Compute significances
        #----------------------------------------------------
        if self.sigMethod == 'isotropic':
            CR.Sigs  = np.array([self.__Compute_Cluster_Significance_3d_Isotropic(CRCoords[cluster]) for cluster in range(len(clusters))])
        elif self.sigMethod =='annulus':
            CR.Sigs   = np.array([self.__Compute_Cluster_Significance_3d_Annulus(CRCoords[cluster], np.transpose(sim)) for cluster in range(len(clusters))])
        elif self.sigMethod == 'BGInt':
            CR.SigsMethod ='BGInt'
            CR.Sigs = np.array(self.BG.SigsBG(CR))
        else: print 'Invalid significance evaluation method: ' , str(self.sigMethod)
            
        return CR
        
    
    
    def __Get_Cluster_Coords(self,sim,labels, cluster_index):
        """Returns a set of coordinate triplets for cluster 'cluster_index' given input vectors [X (1xn),Y(1xn),T(1xn)] in sim, and a set of corresponding labels"""
        idx = np.where(labels==cluster_index)[0] # find indices of points which are in the given cluster
        return np.transpose(np.array(sim))[:][idx] # select out those points and return the transpose (which provides (x,y,t) triplets for each point 
    
    
    def __Cluster_Size(self,cluster_coords):
        """Returns basic cluster properties, given a set of cluster coordinate triplets""" 
        X,Y,T = np.transpose(cluster_coords)
        CentX0,CentY0,CentT0 = np.mean(X),np.mean(Y), np.mean(T)
        X, Y = X-CentX0, Y-CentY0 
        
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
    
        return SIZE95X, SIZE95Y,SIZE95T, POSANG, np.median(r), np.median(dt), CENTX,CENTY,CentT0,SIG95X,SIG95Y,SIG95T
    
    
    def __Cluster_Size_Spherical(self,cluster_coords):
        """Returns basic cluster properties, given a set of cluster coordinate triplets""" 
        X,Y,T = np.transpose(cluster_coords)
        # Map to cartesian
        X, Y = np.deg2rad(X), np.deg2rad(Y)
        x = np.cos(X) * np.cos(Y)
        y = np.cos(X) * np.sin(Y)
        z = np.sin(X)
    
        # Compute Cartesian Centroids
        CentX0,CentY0,CentZ0,CentT0 = np.mean(x),np.mean(y), np.mean(z), np.mean(T)
        r = np.sqrt(CentX0**2 + CentY0**2 + CentZ0**2)
        
        # Rotate away Z components so we are in the x,z plane and properly oriented with galactic coordinates
        def rotation_matrix(axis,theta):
            axis = axis/np.sqrt(np.dot(axis,axis))
            a = np.cos(theta/2.)
            b,c,d = -axis*np.sin(theta/2.)
            return np.array([[a*a+b*b-c*c-d*d, 2*(b*c-a*d), 2*(b*d+a*c)],
                             [2*(b*c+a*d), a*a+c*c-b*b-d*d, 2*(c*d-a*b)],
                             [2*(b*d-a*c), 2*(c*d+a*b), a*a+d*d-b*b-c*c]])
        # Pick axes
        n = np.array([CentX0, CentY0,CentZ0])/r
    
        n=np.array(n)/np.sqrt(np.dot(n,n))
        nz = np.array([0.,0.,1.])
        if (n!=nz).all():axis = np.cross(nz,n) 
        else: axis = np.array([0,1,0])
    
        theta = np.pi/2.-np.arccos(np.dot(nz,n))
    
        R1 = rotation_matrix(axis,-theta)
        print np.dot(R1,n)
        theta2 = np.pi/2-np.arccos(np.dot(R1,n)[0])
        R2 = rotation_matrix(nz,theta2)
        R  = np.dot(R2,R1)
        def rotate(n):
            n = n/np.sqrt(np.dot(n,n))
            return np.dot(R,n)
    
        # rotate all the vectors (Y component should be zero for all)
        X,Z,Y = np.rad2deg(np.transpose([rotate(np.transpose((x,y,z))[i]) for i in range(len(x))]))
        
        #from matplotlib import pyplot as plt
        #plt.figure(0)
        #plt.scatter(X,Y)
        #plt.axis('equal')
        #plt.show()  
        
        # Convert Centroids back to lat/long in radians
        CentY0 = np.rad2deg(np.arctan2(CentY0, CentX0))
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
    
        return SIZE95X, SIZE95Y,SIZE95T, POSANG, np.median(r), np.median(dt), CentX0,CentY0,CentT0,SIG95X,SIG95Y,SIG95T
    
    
    def __Compute_Cluster_Significance_3d_Isotropic(self, X):
        """
        Takes input list of coordinate triplets (in angular scale) and computes the cluster significance based on a background model.
        
        Inputs:
            -X is a tuple containing a coordinate pair for each point in a cluster.  
            -BGDensity is the 2-d number of events per square degree integrated over the background.  Time is handled seperately
            -TotalTime is the full simulation time in units of the time dimension.  Usually months
            
        returns significance
        """
        # Default to zero significance
        if (len(X)==1):return 0
        # Otherwise.......
        x,y,t = np.transpose(X) # Reformat input
        centX,centY,centT = np.mean(x), np.mean(y),np.mean(t) # Compute Centroid
        r = np.sqrt(  np.square(x-centX) + np.square(y-centY) ) # Build list of radii from cluster centroid
        countIndex = int(np.ceil(0.95*np.shape(r)[0]-1)) # Sort the list and choose the radius where the cumulative count is >95% 
        clusterRadius = np.sort(r)[countIndex]   # choose the radius at this index 
        N_bg = np.pi * clusterRadius**2. * self.BGDensity # Use isotropic density to compute the background expectation
        dT = (np.max(t)-np.min(t))/float(self.TotalTime) # Rescale according to the total time.
        ######################################################
        # Evaluate significance as defined by Li & Ma (1983).  N_cl corresponds to N_on, N_bg corresponds to N_off
        if dT > .01: # This would be a 15 day period     
            N_bg,N_cl = N_bg*dT, countIndex
            if N_cl/(N_cl+N_bg)<1e-20 or N_bg/(N_cl+N_bg)<1e-20:
                return 0
            S2 = 2.0*(N_cl*np.log(2.0*N_cl/(N_cl+N_bg)) + N_bg*np.log(2.0*N_bg/(N_cl+N_bg)))
            if S2>0.0:
                return np.sqrt(S2)   
            else:
                return 0
        else: return 0
        
        
        
    def __Compute_Cluster_Significance_3d_Annulus(self,X_cluster,X_all):
        """
        Takes input list of coordinate triplets for the cluster and for the entire simulation and computes the cluster size.
        Next, the background level is computed by drawing an annulus centered on the the cluster with inner and outer radii 
        specified as a fraction of the initial radius.  Then the significance is calculated.  The cluster is cylindrical
        with the axis aligned temproally.  Similarly, the background annulus is taken over a cylindrical shell and is 
        computed over the range of times in X_all (thus if the background is time varying, this will average that).  
        
        Inputs:
            -X_cluster: A tuple containing a coordinate triplet (x,y,z) for each point in a cluster.  
            -X_all:     A tuple containing coordinate triplets for background events, typically just all events.
            -inner:    The inner radius of the background annulus in fraction of cluster radius.
            -outer:    The outer radius of the background annulus in fraction of cluster radius.
            
        return:
            Cluster significance from Li & Ma (1985)
        """
        # Default to zero significance
        if (len(X_cluster)==1):return 0
        # Otherwise.......
        x,y,t = np.transpose(X_cluster) # Reformat input
        x_all,y_all,t_all = X_all # Reformat input
        centX,centY,centT = np.mean(x), np.mean(y),np.mean(t) # Compute Centroid
        minT,maxT = np.min(t), np.max(t) # find the window of times for the cluster
        r = np.sqrt(np.square(x-centX)+np.square(y-centY)) # Build list of radii from cluster centroid
        countIndex = int(np.ceil(0.95*np.shape(r)[0]-1)) # Sort the list and choose the radius where the cumulative count is >95% 
        clusterRadius = np.sort(r)[countIndex]             # choose the radius at this index 
        min_T_all, max_T_all = np.min(t_all),np.max(t_all)
        ################################################################
        # Estimate the background count
        AnnulusVolume = np.pi* ((self.outer*clusterRadius)**2 -(self.inner*clusterRadius)**2)*(max_T_all-min_T_all)
        r_all = np.sqrt(np.square(x_all-centX)+np.square(y_all-centY)) # compute all points radius from the centroid. 
        r_cut = np.logical_and(r_all>clusterRadius*self.inner,r_all<clusterRadius*self.outer)# Count the number of points within the annulus and find BG density
        idx =  np.where(r_cut==True)[0] # pick points in the annulus
        BGDensity = np.shape(idx)[0]/AnnulusVolume # Density = counts / annulus volume 
        ######################################################
        # Evaluate significance as defined by Li & Ma (1983).  N_cl corresponds to N_on, N_bg corresponds to N_off
        N_bg = np.pi * clusterRadius**2. * (maxT-minT)* BGDensity # BG count = cluster volume*bgdensity
        N_cl = countIndex # Number of counts in cluster.
        # Ensure log args are greater than 0.
        if N_cl/(N_cl+N_bg) <= 0 or N_bg/(N_cl+N_bg) <= 0: return 0 
        S2 = 2.0*(N_cl*np.log(2.0*N_cl/(N_cl+N_bg)) + N_bg*np.log(2.0*N_bg/(N_cl+N_bg)))
        if S2>0.:
            return np.sqrt(S2)   
        else:
            return 0.


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


