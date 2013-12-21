# -*- coding: utf-8 -*-
#===============================================================================
# DBSCAN.py: Methods for running clustering algorithm and computing some cluster
#  statistics such as significance, background counts.
# Author: Eric Carlson
#===============================================================================
import numpy as np
import math 
from sklearn.base import BaseEstimator, ClusterMixin
import multiprocessing as mp
from multiprocessing import pool
from functools import partial

def RunDBScan3D(X,eps,nMin,a,nCorePoints =3 , plot=False,indexing=True, metric='euclidean',D=3, output=True):
    """
    Runs DBSCAN3D on photon list, X, of coordinate triplets using parameters eps and n_samples defining the search radius and the minimum number of events.
    
    Inputs:
        X:         a list of coordinate pairs (n x 3) for each photon
        eps:       DBSCAN epsilon parameter
        N_min:     DBSCAN core point threshold.  Must have AT LEAST n_samples points per eps-neighborhood to be core point
        TimeScale: This number multiplies the epsilon along the time dimension.
    Optional inputs 
        N_CorePoints: Number of core points required to form a cluster.  This is different than the N_min parameter.
        plot:         For debugging, this allows a a 3-d visualization of the clusters
        
    Returns: 
        Labels: A list of cluster labels corresponding to X.  -1 Implies noise.
    """
    #===========================================================================
    # Compute DBSCAN
    #===========================================================================
    db = DBSCAN(eps, nMin=nMin, a=a,indexing=indexing,metric=metric,D=D, output=output).fit(X)
    core_samples = db.core_sample_indices_ # Select only core points.
    labels = db.labels_                    # Assign Cluster Labels
    # Get the cluster labels for each core point
   
    # pick out the cluster labels of core points
    coreLabels = labels[core_samples]
    # Count the number of members for each cluster    

    clusterCounts = np.bincount((np.array(coreLabels)+1).astype(int))[1:]
    # find which clusters have too few points
    invalid = np.where(clusterCounts < nCorePoints)[0] # clusters which aren't valid
    # set invalid clusters to noise.
    for i in invalid:
        coreLabels[np.where(coreLabels==i)[0]]=-1 
    # set all labels to -1 for noise
    labels = -np.ones(len(labels)).astype(int)
    labels[core_samples] = coreLabels

    #===========================================================================
    # # Plot result
    #===========================================================================
    if (plot == True):   
        import pylab as pl
        from itertools import cycle
        from mpl_toolkits.mplot3d import Axes3D
        pl.close('all')
        fig = pl.figure(1)
        pl.clf()
        fig = pl.gcf()
        ax = fig.gca(projection='3d')
        # Black removed and is used for noise instead.
        colors = cycle('bgrcmybgrcmybgrcmybgrcmy')
        
        for k, col in zip(set(labels), colors):
            if k == -1:
                # Black used for noise.
                col = 'k'
                markersize = 6
            class_members = [index[0] for index in np.argwhere(labels == k)]

            #cluster_core_samples = [index for index in core_samples
            #                        if labels[index] == k]
            for index in class_members:
                x = X[index]
                if index in core_samples and k != -1:
                    markersize = 6
                    ax.scatter(x[0],x[1],x[2], color=col,s=markersize)
                else:
                    markersize = 2
                    ax.scatter(x[0], x[1],x[2], c=col,s=2)
        pl.axis('equal')
        pl.xlabel(r'$l$ [$^\circ$]')
        pl.ylabel(r'$b$ [$^\circ$]')
        pl.show()

    return labels

def epsilonQuerySpherical_Thread(k,Xidx, Yidx, XX, XY, XT, Grid,GridSizeX, GridSizeY, high_grid_lat, low_grid_lat, a,eps,D,indexing):  
    
    where = np.where     # Defined for quicker calls
    square = np.square   # Defined for quicker calls
    sin = np.sin
    cos = np.cos
    arctan2 = np.arctan2
    sqrt = np.sqrt
    if indexing == True:
        i,j = Xidx[k],Yidx[k]
        il,ih = i-1, i+2 # select neighboring grid indices.
        if (XX[k]<85 and XX[k]>-85):
            jl,jh = int(j-1./np.sin(np.abs(np.deg2rad(90-XX[k])))), int(j+2./np.abs(np.sin(np.deg2rad(90-XX[k])))) # np.sin(np.deg2rad(90-84))
        # if within 10 degrees of either pole, return all points above or below
        if il<=low_grid_lat:
            jl,jh  = 0,GridSizeY # select all longitudes
            il,ih  = 0,low_grid_lat+1
        if ih>=high_grid_lat: 
            jl,jh  = 0,GridSizeY # select all longitudes
            il,ih  = high_grid_lat-1, GridSizeX
        idx = []
        # if we span the line of 0 longitude, we need to break into 2 chunks.
        if (jl<0 or jh > GridSizeY):
            
            idx = idx + [item for sublist in [item for sublist2 in Grid[il:ih,int(-2-2):] for item in sublist2] for item in sublist]
            idx = np.array(idx + [item for sublist in [item for sublist2 in Grid[il:ih,0:2+int(2.)] for item in sublist2] for item in sublist])
        else:
            idx = np.array([item for sublist in [item for sublist2 in Grid[il:ih,jl:jh] for item in sublist2] for item in sublist])
    
    if indexing==False: idx = np.array(range(0,n)).astype(int) # select all points
    
    #Compute real arc lengths for these points.
    idx = np.append(idx,k).astype(int)
    j = -1 # index of original point in reduced list
    x = np.deg2rad(XX[idx])
    y = np.deg2rad(XY[idx])
    dPhi = x-x[j] # lat 
    dLam = y-y[j] # lon
    # Distances using Vincenty's formula for arc length on a great circle.
    d = arctan2(sqrt( square(cos(x)*sin(dLam) ) + square(cos(x[j])*sin(x)-sin(x[j])*cos(x)*cos(dLam)) ) , sin(x[j])*sin(x)+cos(x[j])*cos(x)*cos(dLam) )
    
    # Find where within time constraints if D=3, else just set all true
    if D==3: tcut = np.logical_and(XT[idx] <= XT[k]+float(a),XT[idx] >= XT[k]-
    float(a))
    else:    tcut = np.ones(len(idx)) 
    # Find where the real distance is less than eps
    rcut = d<np.deg2rad(eps)
    return idx[where(np.logical_and(rcut, tcut)==True)[0]] # This now contains indices of points in the eps neighborhood 
    

def __epsQueryThread(k,Xidx,Yidx,GridSizeX,GridSizeY,Grid,XX,XY,XT,a,eps,nMin,D):
    """ Returns the epsilon neighborhood of a point for euclidean metric"""  
    i,j = Xidx[k],Yidx[k]
    il,ih = i-1, i+2
    jl,jh = j-1, j+2
    if jl<0  : jl=0
    if il<0  : il=0
    if ih>=GridSizeX: ih=-1
    if jh>=GridSizeY: jh=-1
    idx = np.array([item for sublist in [item for sublist2 in Grid[il:ih,jl:jh] for item in sublist2] for item in sublist])
    if len(idx) !=0:
        # if D==3, cut on T, else include all.
        if D==3: tcut = np.logical_and(XT[idx] <= XT[k]+float(a),XT[idx] >= XT[k]-float(a))
        else:    tcut = np.ones(len(idx)) 
        tcut = np.where(tcut==True)[0]
        if len(tcut)!=0: 
            idx = idx[tcut] #original indices meeting tcut  This is the rough eps neighborhood
            # Compute actual distances using numpy vector methods                
            return idx[np.where( np.square( XX[idx] - XX[k]) + np.square(XY[idx] - XY[k]) <= eps*eps)[0]]
        else: return np.array([])
    else: return np.array([])
        
import sys
class DBSCAN(BaseEstimator, ClusterMixin):
#class DBSCAN(BaseEstimator):
    """Perform DBSCAN clustering from vector array or distance matrix.

    DBSCAN - Density-Based Spatial Clustering of Applications with Noise.
    Finds core samples of high density and expands clusters from them.
    Good for data which contains clusters of similar density.

    Parameters
    ----------
    eps : float, optional
        The maximum distance between two samples for them to be considered
        as in the same neighborhood.
    min_samples : int, optional
        The number of samples in a neighborhood for a point to be considered
        as a core point.
    metric : string, or callable
        'euclidean' or 'spherical', where 'spherical is in degrees.

    Attributes
    ----------
    `core_sample_indices_` : array, shape = [n_core_samples]
        Indices of core samples.

    `components_` : array, shape = [n_core_samples, n_features]
        Copy of each core sample found by training.

    `labels_` : array, shape = [n_samples]
        Cluster labels for each point in the dataset given to fit().
        Noisy samples are given the label -1.

    Notes
    -----
    See examples/plot_dbscan.py for an example.

    References
    ----------
    Ester, M., H. P. Kriegel, J. Sander, and X. Xu, “A Density-Based
    Algorithm for Discovering Clusters in Large Spatial Databases with Noise”.
    In: Proceedings of the 2nd International Conference on Knowledge Discovery
    and Data Mining, Portland, OR, AAAI Press, pp. 226–231. 1996
    """

    def __init__(self, eps, nMin, a,D=3, metric='spherical',indexing = True, output = True):
        ##@var eps 
        # DBSCAN Search radius float in input units of X.
        ##@var nMin 
        #Minimum number of samples in eps neighborhood for core points. Can be single int, or array of shape (n,1) for n input samples.
        ##@var a
        #DBSCAN temporal search half-height.  The temporal analog of epsilon, execept that the dbscan searches cylindrically with flat time surfaces as the boundary. Unused for D=2
        ##@var D
        # Number of dimensions to search.  Must be 2 (for 2 spatial dimensions) or 3 for (2 spatial + 1 temporal). DEFAULT 3 
        ##@var metric
        #Metric used for distance calculations. Can be 'euclidean' or 'spherical' (spherical uses lat,long and vincenty's formula for great circle distances.
        ##@var indexing
        #Grid based spatial indexing improves speeds for input > 2000 samples.  May provide slight speedup for lower number.  Usually irrelevant to disable. 
        self.eps = eps
        self.nMin = nMin
        self.metric = metric
        self.a = a
        self.indexing = indexing
        self.D = D
        self.output= output
    def fit(self, X, **params):
        """Perform DBSCAN clustering from vector array or distance matrix.

        Parameters
        ----------
        X: array [n_samples X (or lat), n_samples Y (or long),n_samples T]
        """
        self.core_sample_indices_, self.labels_ = self.dbscan3_indexed(X,**self.get_params())
        return self

    def dbscan3_indexed(self, X, eps, nMin, a, metric,indexing,D, output ):
        """Perform DBSCAN clustering from vector array or distance matrix.
        @param X a set of coordinate triplets (lat,long,time) for each input sample.  If D=2, time is unused, and need not be given. 
        @param eps DBSCAN Search radius float in input units of X.
        @param nMin Minimum number of samples in eps neighborhood for core points. Can be single int, or array of shape (n,1) for n input samples.
        @param a DBSCAN temporal search half-height.  The temporal analog of epsilon, execept that the dbscan searches cylindrically with flat time surfaces as the boundary. Unused for D=2
        @param D Number of dimensions to search.  Must be 2 (for 2 spatial dimensions) or 3 for (2 spatial + 1 temporal). DEFAULT 3 
        @param metric Metric used for distance calculations. Can be 'euclidean' or 'spherical' (spherical uses lat,long and vincenty's formula for great circle distances.
        @param indexing Grid based spatial indexing improves speeds for input > 2000 samples.  May provide slight speedup for lower number.  Usually irrelevant to disable.
    
        @returns core_samples: array [n_core_samples] Indices of core samples.
        @returns labels : array [n_samples] Cluster labels for each point.  Noisy samples are given the label -1.
    
        References
        ----------
        Ester, M., H. P. Kriegel, J. Sander, and X. Xu, “A Density-Based
        Algorithm for Discovering Clusters in Large Spatial Databases with Noise”.
        In: Proceedings of the 2nd International Conference on Knowledge Discovery
        and Data Mining, Portland, OR, AAAI Press, pp. 226–231. 1996
        """
        
        X = np.asarray(X,dtype = np.float32)    # convert to numpy array
        n = np.shape(X)[0]   # Number of points
        if D==3: XX,XY,XT = X[:,0],X[:,1],X[:,2] # Separate spatial component
        if D==2: XX,XY,XT = X[:,0],X[:,1],np.zeros(n) # Separate spatial component and make up time component
        where = np.where     # Defined for quicker calls
        square = np.square   # Defined for quicker calls
        sin = np.sin
        cos = np.cos
        arctan2 = np.arctan2
        sqrt = np.sqrt
        if np.shape(nMin)==():nMin = (nMin*np.ones(len(XX))).astype(int)
        ############################################################################
        # In this section we create assign each point in the input array an index i,j
        # which locates the points on a large scale grid.  The epsilon query will then
        # only compute distances to points in the neighboring grid points.
        # For a Euclidean metric, we want a grid with elements of width ~epsilon.
        # This will have to be refined for spherical metrics since the poles produce
        # anomalous behavior.
        startX, stopX = np.min(XX), np.max(XX) # widths
        startY, stopY = np.min(XY), np.max(XY)
        eps2=eps*2. # Larger grids offer a speed advantage in most cases (i.e. takes longer to index with finer grid)
        # We choose rectangles of 6x6 epsilons and return all grid elements when query point is within 10 degrees of a pole.
        GridSizeX, GridSizeY = int(np.ceil((stopX-startX)/eps2)), int((np.ceil((stopY-startY)/eps2)))
        Xidx, Yidx = np.floor(np.divide((XX-startX),eps2)).astype(int), np.floor(np.divide((XY-startY),eps2)).astype(int) # Grid indices for all points.
        # Iterate through each grid element and add indices
        Grid = np.empty(shape=(GridSizeX,GridSizeY),dtype=object)
        for i in range(GridSizeX):
            Xtrue = where((Xidx==i))[0] # indices where x condition is met
            for j in range(GridSizeY):
                #find indicies 
                Ytrue = np.where((Yidx[Xtrue]==j))[0]
                Grid[i][j] = Xtrue[Ytrue]        
        #==============================================================================
        
        #===========================================================================
        # Refine the epislon neighborhoods (compute the euclidean distances)
        #===========================================================================
        if (metric == 'euclidean'):
            EPS_PARTIAL = partial(__epsQueryThread,  Xidx=Xidx,Yidx=Yidx,GridSizeX=GridSizeX,GridSizeY=GridSizeY,Grid=Grid,XT=XT,XX=XX,XY=XY,a=a,eps=eps,nMin=nMin)    
            #neighborhoods = map(EPS_PARTIAL,range(0,n)) # Call mutithreaded map.
            p = pool.Pool(mp.cpu_count()) # Allocate thread pool
            neighborhoods = p.map(EPS_PARTIAL,range(0,n)) # Call mutithreaded map.
            p.close()  # Kill pool after jobs complete.  required to free memory.
            p.join()   # wait for jobs to finish.
    
        elif (metric=='spherical'):
            #=========================================================================================
            # First query the grid
            # find the grid point boundaries within 12 deg of pole, return all longitudes and all latitudes above/below
            high_grid_lat = np.floor((84.-startX)/eps2)  # Grid queries for elements above this latitude should return all longitudes
            low_grid_lat  = np.ceil((-84.-startX)/eps2) # Grid queries for elements below this latitude should return all longitudes
            # ensure that these are within the grid bounds.
            if low_grid_lat<0: low_grid_lat=0
            if high_grid_lat >= GridSizeX: high_grid_lat=GridSizeX-1
            #sys.stdout.write("\r" + '[') # return to start of line, after '['
            #def get_grid_points(k):  
            def __epsilonQuerySpherical(k):  
                if (k)%(n/1000)==0:
                    if output==True: sys.stdout.write("\rDBSCAN Run " + ("%.1f" % (100.*(k+1)/float(n))) + '% Complete')
                    sys.stdout.flush()
            
                if indexing == True:
                    i,j = Xidx[k],Yidx[k]
                    il,ih = i-1, i+2 # select neighboring grid indices.
                    if (XX[k]<85 and XX[k]>-85):
                        jl,jh = int(j-1./np.sin(np.abs(np.deg2rad(90-XX[k])))), int(j+2./np.abs(np.sin(np.deg2rad(90-XX[k])))) # np.sin(np.deg2rad(90-84))
                    # if within 10 degrees of either pole, return all points above or below
                    if il<=low_grid_lat:
                        jl,jh  = 0,GridSizeY # select all longitudes
                        il,ih  = 0,low_grid_lat+1
                    if ih>=high_grid_lat: 
                        jl,jh  = 0,GridSizeY # select all longitudes
                        il,ih  = high_grid_lat-1, GridSizeX
                    idx = []
                    # if we span the line of 0 longitude, we need to break into 2 chunks.
                    if (jl<0 or jh > GridSizeY):
                        
                        idx = idx + [item for sublist in [item for sublist2 in Grid[il:ih,int(-2-2):] for item in sublist2] for item in sublist]
                        idx = np.array(idx + [item for sublist in [item for sublist2 in Grid[il:ih,0:2+int(2.)] for item in sublist2] for item in sublist])
                    else:
                        idx = np.array([item for sublist in [item for sublist2 in Grid[il:ih,jl:jh] for item in sublist2] for item in sublist])
                
                if indexing==False: idx = np.array(range(0,n)).astype(int) # select all points
                
                #Compute real arc lengths for these points.
                idx = np.append(idx,k).astype(int)
                j = -1 # index of original point in reduced list
                x = np.deg2rad(XX[idx])
                y = np.deg2rad(XY[idx])
                dPhi = x-x[j] # lat 
                dLam = y-y[j] # lon
                # Distances using Vincenty's formula for arc length on a great circle.
                d = arctan2(sqrt( square(cos(x)*sin(dLam) ) + square(cos(x[j])*sin(x)-sin(x[j])*cos(x)*cos(dLam)) ) , sin(x[j])*sin(x)+cos(x[j])*cos(x)*cos(dLam) )
                
                # Find where within time constraints if D=3, else just set all true
                if D==3: tcut = np.logical_and(XT[idx] <= XT[k]+float(a),XT[idx] >= XT[k]-
                float(a))
                else:    tcut = np.ones(len(idx)) 
                # Find where the real distance is less than eps
                rcut = d<np.deg2rad(eps)
                return idx[where(np.logical_and(rcut, tcut)==True)[0]] # This now contains indices of points in the eps neighborhood 
            #=========================================================================================
            #import DBSCAN
            #eps_partial = partial(DBSCAN.epsilonQuerySpherical_Thread,Xidx=Xidx, Yidx=Yidx, XX=XX, XY=XY, XT=XT, Grid=Grid,
            #                       GridSizeX=GridSizeX, GridSizeY=GridSizeY, high_grid_lat=high_grid_lat, low_grid_lat=low_grid_lat,
            #                       a=a,eps=eps,D=D,indexing=indexing)
            #print mp.cpu_count()
            #p = pool.Pool(mp.cpu_count()) # Allocate thread pool
            #neighborhoods = p.map(eps_partial,range(0,n)) # Call mutithreaded map.
            #p.close()  # Kill pool after jobs complete.  required to free memory.
            #p.join()   # wait for jobs to finish.                       
            # Serial Version

            neighborhoods = [ __epsilonQuerySpherical(k) for k in range(0,n)]
            if output==True: print '\n',
            
        #print "Mean counts/eps-neighborhood", np.mean([len(nhood) for nhood in neighborhoods])
        # Initially, all samples are noise.
        labels = -np.ones(n).astype(int)
        #======================================================
        # From here the algorithm is essentially the same as sklearn
        #======================================================
        core_samples = [] # A list of all core samples found.
        label_num = 0 # label_num is the label given to the new clust
    
        for index in range(0,n):
            if labels[index] != -1 or len(neighborhoods[index]) < nMin[index]:
                # This point is already classified, or not enough for a core point.
                continue
            core_samples.append(index)
    
            labels[index] = label_num
            # candidates for new core samples in the cluster.
            candidates = [index]
            while len(candidates) > 0:
                new_candidates = []
                # A candidate is a core point in the current cluster that has
                # not yet been used to expand the current cluster.
                for c in candidates:
                    noise = where(labels[neighborhoods[c]] == -1)[0]
                    noise = neighborhoods[c][noise]
                    labels[noise] = label_num
                    for neighbor in noise:
                        # check if its a core point as well
                        if len(neighborhoods[neighbor]) >= nMin[index]:
                            # is new core point
                            new_candidates.append(neighbor)
                            core_samples.append(neighbor)
                # Update candidates for next round of cluster expansion.
                candidates = new_candidates
            # Current cluster finished.
            # Next core point found will start a new cluster.
            label_num += 1
        return core_samples, labels


    

