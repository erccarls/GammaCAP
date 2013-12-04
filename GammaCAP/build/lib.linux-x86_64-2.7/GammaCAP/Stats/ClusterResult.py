import numpy as np

class ClusterResult():
    '''The ClusterResult class contains the results of DBSCAN and cluster information for a single simulation including size, labels, coordinates, significances, centroid information, etc..'''
    
    def __init__(self,Labels=0, Coords=0, CentX=0, CentY=0, CentT=0,Sig95X=0,Sig95Y=0,Sig95T=0,Size95X=0,Size95Y=0, Size95T=0,PA=0,MedR=0,MedT=0, Members=0, Sigs=0, SigsMethod=0, NumClusters=0):
        self.Labels  = Labels # Array of integer cluster labels
        self.Coords  = Coords # List of coordinate triplet arrays for each cluster
        self.CentX   = CentX # Array of X centroids
        self.CentY   = CentY # Array of Y centroids
        self.CentT   = CentT # Array of T centroids
        self.Sig95X  = Sig95X  # 95% Uncertainty on the centroid major axis position
        self.Sig95Y  = Sig95Y  # 95% Uncertainty on the centroid minor axis position
        self.Sig95T  = Sig95T  # Uncertainty on the time centroid
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
    
        
