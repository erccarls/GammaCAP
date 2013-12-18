import numpy as np

class ClusterResult():
    '''The ClusterResult class contains the results of DBSCAN and cluster information for a single simulation including size, labels, coordinates, significances, centroid information, etc..'''
    
    def __init__(self,Labels=0, Coords=0, CentX=0, CentY=0, CentT=0,Sig95X=0,Sig95Y=0,Sig95T=0,Sig95R=0,Size95X=0,Size95Y=0, Size95T=0,PA=0,MedR=0,MedT=0, Members=0, Sigs=0, SigsMethod=0, NumClusters=0,Dens33=0,Dens66=0,Dens100=0,e=0):
        """
        Initialize the ClusterResults object.
        """
        ##@var Labels
        # Cluster labels for each point in the input list.  -1 for noise >=0 for clusters. Note that the label numbers may not increment by one as some clusters may not posess enough core points to be valid. 
        #@var Coords
        # For each of n detected clusters, contains a list of coordinate triplets of shape (m,3) for m members of the cluster (3 corresponds to [lat.,long.,time]) numpy.ndarray of shape (n, m_i,3) for n valid clusters with cluster i containing m_i elements.
        ##@var CentX 
        # X centroids (latitude for spherical coordinates).  Computed by finding standard centroid and then reaveraging distances weighted by 1/r. numpy.ndarray of shape (n,1) for n valid clusters
        ##@var CentY 
        # Y centroids (longitude for spherical coordinates). Computed by finding standard centroid and then reaveraging distances weighted by 1/r. numpy.ndarray of shape (n,1) for n valid clusters
        ##@var CentT 
        # Time centroids. numpy.ndarray of shape (n,1) for n valid clusters
        ##@var Sig95R
        # 2-sigma uncertainties on the centroids position (geometric mean).  This should be used for positional comparison Computed by taking 2*stdev of radii from centroid and dividing by sqrt(cluster members). numpy.ndarray of shape (n,1) for n valid clusters
        ##@var Sig95X 
        # 2-sigma uncertainties on the centroids X position (latitude for spherical).  Computed by taking the major and minor 95% containment radii
        # from a principle component analysis, rotating back to the reference coordinates, and dividing by sqrt(cluster members).numpy.ndarray of shape (n,1) for n valid clusters
        ##@var Sig95Y
        # 2-sigma uncertainties on the centroids Y position (longitude for spherical).  Computed by taking the major and minor 95% containment radii
        # from a principle component analysis, rotating back to the reference coordinates, and dividing by sqrt(cluster members). numpy.ndarray of shape (n,1) for n valid clusters
        ##@var Sig95T 
        # 2-sigma uncertainties on the mean time found by taking the stdev of the temporal distribution and dividing by sqrt(cluster members).  numpy.ndarray of shape (n,1) for n valid clusters
        ##@var Size95X
        # Semi-major axis of bounding ellipse found by taking 2-sigma value of the first principle component. numpy.ndarray of shape (n,1) for n valid clusters
        ##@var Size95Y
        # Semi-minor axis of bounding ellipse found by taking 2-sigma value of the second principle component. numpy.ndarray of shape (n,1) for n valid clusters
        ##@var Size95T
        # 95% containment half-time.  That is, the distance from the centroid which contains 95% of the detected cluster elements. numpy.ndarray of shape (n,1) for n valid clusters
        ##@var PA
        # Position angle of semi-major axis of bounding ellipse measured counter-clockwise with respect to north celestial pole along lines of constant longitude. numpy.ndarray of shape (n,1) for n valid clusters
        ##@var MedR 
        # Median radius from centroid.  Provides complementary measurement compared with Size95X/Y as it is more robust to outliers.  
        # Combining the two measures can indicate the cluster's density profile and is useful in boosted decision tree. Note that the effective cluster radius is therefore
        # 2*MedR[i]. numpy.ndarray of shape (n,1) for n valid clusters.
        ##@var MedT
        # Median temporal distance from centroid.  Provides complementary measurement compared with Size95T as it is more robust to outliers.  
        # Combining the two measures can indicate the cluster's density profile and is useful in boosted decision tree. Note that the effective cluster length is therefore
        # 4*MedT[i]. numpy.ndarray of shape (n,1) for n valid clusters.
        ##@var Members
        # Number of elements in each cluster. numpy.ndarray of shape (n,1) for n valid clusters
        ##@var Sigs
        # Cluster significance over background counts (computed by method specified in SigsMethod).  Given the cluster count and background expectation, the significance measure proposed of Li & Ma (1983) is used.  
        # For reasonably large photon counts, this is effectively a guassian z-score, and thus a cluster with significance 3 is a 3-stdev event.  The squared significance provides a test-statistic. 
        # numpy.ndarray of shape (n,1) for n valid clusters
        ##@var SigsMethod
        # Method used to estimate the background level in the significance computation.  Either 'isotropic' or 'BGInt'.
        ##@var NumClusters
        # Number of valid clusters (note not in general equal to max(Labels). Single integer.
        ##@var e
        # Cluster eccentricity defined as sqrt(1-Size95Y^2/Size95X^2).  Should be near zero for point sources.
        ##@var Dens33
        # Percentage of cluster elements in the inner 33% of Size95X
        ##@var Dens66
        # Percentage of cluster elements between 33%-66% of Size95X
        ##@var Dens100
        # Percentage of cluster elements in the outer 33% of Size95X
        self.Labels  = Labels # Array of integer cluster labels
        self.Coords  = Coords # List of coordinate triplet arrays for each cluster
        self.CentX   = CentX # Array of X centroids
        self.CentY   = CentY # Array of Y centroids
        self.CentT   = CentT # Array of T centroids
        self.Sig95X  = Sig95X  # 95% Uncertainty on the centroid major axis position
        self.Sig95Y  = Sig95Y  # 95% Uncertainty on the centroid minor axis position
        self.Sig95T  = Sig95T  # Uncertainty on the time centroid
        self.Sig95R  = Sig95R
        self.Size95X = Size95X # Array of Cluster Radii 95% Containment
        self.Size95Y = Size95Y # Array of Temporal Cluster Lengths
        self.Size95T = Size95T # Array of Temporal Cluster Lengths
        self.PA      = PA # Array of Temporal Cluster Lengths
        self.MedR    = MedR # Median radius from centroid
        self.MedT    = MedT # Median absolute time from centroid
        self.Members = Members # Array of Cluster Member Counts
        self.Sigs    = Sigs # Array of Cluster Signficances
        self.SigsMethod = SigsMethod # Method used to compute significances
        self.NumClusters= NumClusters # Number of clusters found 
        self.e = e # eccentricity
        self.Dens33 = Dens33# Percentage of cluster elements in the inner 33% of Size95X
        self.Dens66 = Dens66 # Percentage of cluster elements between 33%-66% of Size95X
        self.Dens100 = Dens100# Percentage of cluster elements in the outer 33% of Size95X
    
    def SaveToTxt(self,fname,delimiter=','):
        """
        Write the cluster results to a csv text file.
        @param fname Filename.
        @param delimiter Delimiter between fields.
        """
        import csv
        CR = self
        Names = ['Label','CentX', 'CentY', 'CentT', 'Sig95X', 'Sig95Y','Sig95T','PA',
         'Size95X', 'Size95Y','Size95T','MedR','MedT','Members','Sigs']
        data = np.transpose([np.arange(len(CR.CentX)) , CR.CentX, CR.CentY, CR.CentT,CR.Sig95X,CR.Sig95Y,CR.Sig95T,CR.PA,
         CR.Size95X,CR.Size95Y, CR.Size95T, CR.MedR,CR.MedT,CR.Members, CR.Sigs])
        w = csv.writer(open(fname,'wb'),delimiter=delimiter)
        w.writerow(Names)
        for row in data.astype(np.float32):
            w.writerow(row)

