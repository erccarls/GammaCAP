"""@package SimTools
SimTools is a package to quickly simulate Fermi-LAT data for testing new models or 
training machine learning algorithms.  It can simulate 10^6 points in much less than a
minute compared with gtobssim, which can take multiple days.  It includes the galactic 
diffuse and isotropic gamma ray background models, as well as the energy dependent 
fermi point spread function. It also includes an easy to use tool for adding generic 
point sources.  It does not include spacecraft pointing or flight data.  For detailed 
studies, one should still use gtobssim.
"""

import BGTools
import FermiPSF
import numpy as np

class SimTools:
    """
    SimTools is a package to quickly simulate Fermi-LAT data for testing new models or 
    training machine learning algorithms.  It can simulate 10^6 points in much less than a
    minute compared with gtobssim, which can take multiple days.  It includes the galactic 
    diffuse and isotropic gamma ray background models, as well as the energy dependent 
    fermi point spread function. It also includes an easy to use tool for adding generic 
    point sources. It does not include spacecraft pointing or flight data.  For detailed 
    studies, one should still use gtobssim.
    """

    def __init__(self,eMin=1000,eMax=6e5,time=4*3.15e7,diff_f='', iso_f='',convType='both'):
        """
        Initialize the simulation object.  Background tempmlates are built during initilization so any future changes to parameters should create a new SimTools instance.
        """
        ##@var eMin
        # Minimum energy in MeV.
        ##@var eMax  
        # Maximum energy in MeV.
        ##@var time 
        # Total Integration time in seconds.
        ##@var diff_f 
        # Abosulte path to diffuse BG model (typically '$FERMI_DIR/refdata/fermi/galdiffuse/gll_iem_v05.fits') where $FERMI_DIR is the Fermi science tools installation path.
        # if left as empty string "" will attempt to locate it automatically if $FERMI_DIR is a valid environmental variable pointing to the fermi science tools directory.  Otherwise can be downloaded 
        # (instructions at http://planck.ucsc.edu/gammacap).
        ##@var iso_f 
        # Abosulte path to isotropic BG model (typically '$FERMI_DIR/refdata/fermi/galdiffuse/isotrop_4years_P7_v9_repro_clean_v1.txt') where $FERMI_DIR is the Fermi science tools installation path.
        # if left as empty string "" will attempt to locate it automatically if $FERMI_DIR is a valid environmental variable pointing to the fermi science tools directory.  Otherwise can be downloaded 
        # (instructions at http://planck.ucsc.edu/gammacap)
        ##@var BGMaps
        # Contains a list of 2-d arrays with the diffuse galactic and isotropic backgrounds integrated over the energies and times specified during initialization.  Map units are photons/deg^2 and the effective area etc.. 
        # have been empirically determined to agree with gtobssim.  If the diffuse and isotropic models are used with an instrument other than Fermi-LAT, such as CTA or Veritas, the normalizations of each energy 
        # band should be matched in the BGTools.__Prep_Data() routine.
        ##@var sim
        # Contains an array of shape (4,n_samples) containing (latitude,longitude,Time,Energy) for the simulation instance.
        ##@var convType
        # Fermi-LAT conversion type.  Defaults='both' but can also be 'front' or 'back'
        self.eMin = float(eMin)
        self.eMax = float(eMax)
        self.time = float(time)
        self.diff_f = diff_f
        self.iso_f  = iso_f
        self.convType = convType
        self.BGMaps = []
        self.sim    = []
        
        if eMin<50:  raise ValueError("High Energy must be >= than 50 GeV in units MeV")
        if eMax>6e5: raise ValueError("High Energy must be <= than 600 GeV in units MeV")
        
        # Calculate a list of the energies corresponding to the diffuse galactic model
        energies = np.logspace(np.log10(50),np.log10(6e5),31)
        start  = np.argmax(energies>self.eMin)
        stop   = np.argmax(energies>self.eMax)
        
        # Build maps in each of the energy ranges.
        if start-stop<=1: 
            self.BGMaps.append(BGTools.BGTools(self.eMin,self.eMax,self.time,self.diff_f,self.iso_f,self.convType))
        elif start-stop==2:
            self.BGMaps.append(BGTools.BGTools(self.eMin,energies[start+1],self.time,self.diff_f,self.iso_f,self.convType))
            self.BGMaps.append(BGTools.BGTools(energies[start+1],self.eMax,self.time,self.diff_f,self.iso_f,self.convType))
        else:
            self.BGMaps.append(BGTools.BGTools(self.eMin,energies[start],self.time,self.diff_f,self.iso_f,self.convType))
            self.BGMaps += [BGTools.BGTools(energies[i],energies[i+1],self.time,self.diff_f,self.iso_f,self.convType) for i in range(start,stop)]
            self.BGMaps.append(BGTools.BGTools(energies[stop],self.eMax,self.time,self.diff_f,self.iso_f,self.convType))
        # load PSF
        self.theta, self.psf = FermiPSF.GetPSF(eMin,eMax,convType=self.convType)
            
        
    def SimBG(self):
            """
            Simulates a full sky diffuse background model + isotropic.
            @returns Returns a list (B,L,E,T)
            """    
            bM,lM,tM,eM = np.array([]),np.array([]),np.array([]),np.array([])
            tot_phot=0
            # For each background map we simulate b,l,t,e
            for BG in self.BGMaps:
                # Find the maximum of the map, this will provide the sampling window height for our monte carlo
                MAX = np.max(BG.BGMap) 
                # total number of photons to be simulated is given by average of the map (weighted by area - i.e. cos(b))
                # times the number of square degrees in a sphere.
                N_photons = int(129600./np.pi * np.mean(np.average(BG.BGMap, weights = np.abs(np.cos(np.linspace(-np.pi/4.,np.pi/4,1441))),axis=0)))
                N_photons = np.random.poisson(N_photons)
                #print N_photons       
                
                # energy and time distributions for this map
                eM = np.append(eM,self.SampleE(BG.Emin,BG.Emax,N_photons))
                tM = np.append(tM,np.random.ranf(N_photons)*self.time)
                
                b,l = np.zeros(N_photons),np.zeros(N_photons)
                bidx,lidx = 0,0
                tot_phot+=N_photons
                
                while N_photons!=0:
                    # Choose random numbers in the appropriate intervals.
                    B,L,Z = np.rad2deg(np.arccos(2*np.random.ranf(N_photons)-1))-90,np.random.ranf(N_photons)*360.,np.random.ranf(N_photons)*MAX
                    # acceptance/rejection monte carlo.
                    idx = np.where(Z<BG.GetBG(L,B))[0]
                    cnt = len(idx)
                    b[bidx:(bidx+cnt)] = B[idx]
                    l[lidx:(lidx+cnt)] = L[idx]
                    lidx, bidx = lidx+cnt,bidx+cnt
                    N_photons -=cnt
                    #print N_photons
                bM = np.append(bM,b)
                lM = np.append(lM,l)
            if self.sim == []: self.sim = np.array((bM,lM,tM,eM))
            else: self.sim = np.append(self.sim,(bM,lM,tM,eM),axis=1)
            return self.sim
    
    def __rotation_matrix(self,axis,theta):
            axis = axis/np.sqrt(np.dot(axis,axis))
            a = np.cos(theta/2.)
            b,c,d = -axis*np.sin(theta/2.)
            return np.array([[a*a+b*b-c*c-d*d, 2*(b*c-a*d), 2*(b*d+a*c)],
                             [2*(b*c+a*d), a*a+c*c-b*b-d*d, 2*(c*d-a*b)],
                             [2*(b*d-a*c), 2*(c*d+a*b), a*a+d*d-b*b-c*c]])
    
    def SampleE(self,eMin,eMax,n):
        """
        Returns samples from e**-2.5 between the specified energies via inverse sampling method.        
        @param eMin Min energy
        @param EMax Max energy
        @param n Number of samples.
        @returns n Energies distributed according to E^-2.5.
        """
        u = np.random.ranf(n)
        return np.power(eMin**-1.5-u*(eMin**-1.5-eMax**-1.5),-1/1.5)
    
    def SampleDist(self, lam, n, a, b,res=250,spacing='log'):
        """
        Given a function 'lam', sample from the distribution using inverse Monte Carlo sampling method.
        Should be a relatively well behaved function on the interval or else your own sampling should be used.
        @param lam a lambda function defining the distribution (e.g. lam= lambda x: np.exp(-x) )
        @param n number of samples
        @param a left endpoint of sampling
        @param b right endpoint of sampling
        @param res number of bins when sampling the distribution.
        @param spacing is the distribution to be evaluted with 'log' or 
        'lin' spacing (this does not effect how it is sampled, which is always linear here).
        """
        if spacing=='log':
            bins = np.logspace(np.log10(a),np.log10(b),res+1)
            widths = np.array([bins[i+1]-bins[i] for i in range(len(bins)-1)])
            x = np.array([(bins[i+1]+bins[i])*0.5 for i in range(len(bins)-1)])
        else:
            x = np.linspace(a,b,res)
            widths = (b-a)/float(res)
        y   = lam(x) # sample the function
        SUM = np.sum(y*widths) # integrate
        cdf = np.cumsum(y*widths)/SUM # get CDF 
        return x[np.argmin(np.abs(np.transpose(np.ones((n,len(cdf)))*cdf)-np.random.ranf(n)),axis=0)]
    
    def AddPointSource(self, n , E=-1, T='rand', l='rand',b='rand',eMin=-1,eMax=-1):
        """
        Add a point source with n photons and time and energies given by T,E.
        A point spread function will automatically be applied at the energy weighted mean energy, thus if simulating over many energies, should be done in chunks.
        @param n Number of photons
        @param E Energies in MeV If left unspecified will distributed according to power law E^-2.5 between eMin and eMax. numpy.ndarray shape(n) 
        @param T Times in seconds.  If left unspecified will distribute uniformly over SimTools.time. numpy.ndarray shape(n) 
        @param l Galactic longitude of centroid.  If left unspecified will choose random direction. float 
        @param b Galactic latitude of centroid.  If left unspecified will choose random direction. float
        @param eMin Minimum energy for sampling in MeV. float
        @param eMax Maximum energy for sampling in MeV. float
        @returns (b,l,T,E) of point source coordinates.
        """
        # Check energies.
        if eMin==-1 or eMax==-1 or E==-1:
            eMin,eMax = self.eMin,self.eMax
            # if using the preset energy range, don't need to reload psf
            theta, psf = self.theta, self.psf
        # otherwise load the updated psf
        else: theta, psf = FermiPSF.GetPSF(eMin,eMax,convType=self.convType)
        # Sample the energy spectrum if not provided
        if E==-1: E = self.SampleE(eMin,eMax,n)
        if T =='rand': T = np.random.randint(0,high=self.time,size=n)
        if l =='rand': l=np.random.ranf()*360.
        if b =='rand': b=np.rad2deg(np.arccos(2*np.random.ranf()-1))-90
        #=============================================================
        # Here we apply the fermi point spread function.
        # Inverse monte carlo sampling of psf to obtain r
        # Get the energy averaged psf (with weighting ~ E^-2.5)
        #=============================================================
         
        psf = np.cumsum(psf) # Obtain CDF
        # Invert histogram and sample
        r = theta[np.argmin(np.abs(np.transpose(np.ones((n,len(psf)))*psf)-np.random.ranf(n)),axis=0)]
        theta = 2*np.pi*np.random.ranf(n) # Random Angle 
        # Find X and Y displacements
        dY,dZ = np.deg2rad(r*np.cos(theta)),np.deg2rad(r*np.sin(theta))
        # normalize the vectors.  Now (dx,dy,dz) can be rotated to correct galactic coords.
        dX = np.sqrt(1-dZ*dZ-dY*dY)
        # First rotate about y-axis to the correct lat.
        ny = np.array([0.,1.,0.])
        nz = np.array([0.,0.,1.])
        theta2,theta1 = np.deg2rad((l,b))    
        R1 = self.__rotation_matrix(axis=ny,theta=theta1) # construct the rotation matrix
        # The second rotation will move to the correct longitude
        #R2 = self.__rotation_matrix(axis=nz,theta = theta2)
        R2 = self.__rotation_matrix(axis=nz,theta =-l)
        R  = np.dot(R2,R1) # construct full rotation matrix 
        def rotate(n):
            n = n/np.sqrt(np.dot(n,n))
            return np.dot(R,n)
    
        # rotate all the vectors (Y component should be zero for all)
        X,Y,Z = np.transpose([rotate(np.transpose((dX,dY,dZ))[i]) for i in range(len(dX))])

        # Convert Centroids back to lat/long in radians
        Y = (np.rad2deg(np.arctan2(Y, X)) + 360.)%360 # longitude
        X = np.rad2deg(np.arcsin(Z)) # latitude
        # recondition points which have lat<-90 or lat>90 by meridian flipping.
        idx = np.where(X<-90)[0]
        X[idx] = -(X[idx]%90)
        Y[idx] = (Y[idx] + 180)%360
        idx = np.where(X>90)[0]
        X[idx] = 90-(X[idx]%90)
        Y[idx] = (Y[idx] + 180)%360
        
        if self.sim == []: self.sim = np.array((X,Y,T,E))
        else: self.sim = np.append(self.sim,(X,Y,T,E),axis=1)
        return self.sim
        
