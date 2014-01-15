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

    def __init__(self,eMin=1000,eMax=6e5,time=4*3.15e7,diff_f='',convType='both'):
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
        # Unused
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
        self.iso_f  = ''
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
        # find how many psf bins are needed
        nSteps = int(np.ceil((np.log10(eMax)-np.log10(eMin))/0.25))
        self.psfbins = np.logspace(np.log10(eMin-1e-5),np.log10(eMax+1e-5),nSteps+1)
        self.psfenergies = (self.psfbins[:-1]+self.psfbins[1:])*0.5
        self.psf = [FermiPSF.GetPSF(self.psfbins[i],self.psfbins[i+1],convType=self.convType)[1] for i in range(len(self.psfbins)-1)]
        self.theta = FermiPSF.GetPSF(eMin,eMax,convType=self.convType)[0]
            
        
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
    
    def dNdE(self, eMin, eMax, flux, dnde,time=-1):
        """
        Converts a total flux and differential spectrum dN/dE (in (MeV)^-1 and arbitrary normalization) into a list of photon energies sampled between eMin and eMax. N_photons/s for Fermi-LAT P7Source with standard gtmktime cut 
        filter="DATA_QUAL>0 && LAT_CONFIG==1 && ABS(ROCK_ANGLE)<52", taking into account the energy dependent effective area.
        @param eMin Starting energy in MeV
        @param eMax Final energy in MeV
        @param flux Total integrated flux over energy range eMin-eMax in units (cm^2 s)^-1
        @param dnde Differential spectrum in MeV^-1 with arbitrary normalization (e.g. dnde = lambda E: np.power(E/E_0,-Gamma)*np.exp(-(E-E_0)/float(E_C)) )
        @param time Total exposure time in seconds.  If unspecified defaults to SimTools.totalTime.
        @returns Energies. An array of energies corresponding to the sampled distribution.
        """
        if eMin>=eMax: raise ValueError("eMin must be less than eMax.")
        if time==-1: time=self.time
        # Interpolate effective area.  Empirically fudged to agree with  gtobssim simulations.
        EAE =[1000.0, 1246.8043074919699, 1554.5209811805289, 1938.1834554225268, 2416.5354809304745, 3012.9468468312944, 3756.5551068736063, 4683.6890885809644, 5839.6437305958789, 7280.8929575254178, 9077.8487018306387, 11318.300864202816, 14111.706270978171, 17594.536164717007, 21936.943478492387, 27351.075622192155, 34101.438900287758, 42517.820912553041, 53011.402258943228, 66094.844682639901, 82407.337053328229, 102745.82280703213, 128103.93445261421, 159720.53728218854, 199140.25387836422, 248288.9263305887, 309567.7028515347, 385970.34537568642, 481229.48917856964]
        EA = [0.26566958171424865, 0.26608387305457637, 0.26822652118120804, 0.27883475085592402, 0.27412325222410217, 0.24807407731536435, 0.22237229018807317, 0.19595484703425622, 0.16945407469838525, 0.14885401739218102, 0.12622109396344877, 0.10740940927219765, 0.078579024875150283, 0.061169643393579727, 0.04899840542269137, 0.034365021837047274, 0.026774609448862948, 0.022577932439850099, 0.019402914216450195, 0.015139421794612319, 0.014013736466734964, 0.01261170848409187, 0.011208106431659736, 0.0096815936257026152, 0.0084159048191962214, 0.007989708225086373, 0.0059608741119928955, 0.0050705820434280794, 0.0014264672647316718]
        if self.convType=='front':
            EA = [0.14246926714016778, 0.14323587277419941, 0.1449347133648716, 0.15071045564532884, 0.14850256714400339, 0.13487477064697417, 0.12092611023440181, 0.10742156472460739, 0.093636791525725932, 0.081982063594383758, 0.0701762823512838, 0.060883969061369766, 0.044306259607601456, 0.033848301610655239, 0.027213910160945973, 0.018741019722253387, 0.014676421175564567, 0.012201402010701452, 0.010177519738709114, 0.0082101361332825185, 0.0072346777424326311, 0.0067488788014017564, 0.0058985190662511414, 0.0051023044437976577, 0.0045550000310082637, 0.00427900691552743, 0.0031205829990683808, 0.0027736517160632228, 0.00090092669351474013]
        if self.convType=='back':
            EA = [0.12320031457408084, 0.12284800028037693, 0.12329180781633645, 0.12812429521059515, 0.12562068508009877, 0.11319930666839019, 0.10144617995367136, 0.088533282309648839, 0.07581728317265933, 0.066871953797797243, 0.056044811612164964, 0.046525440210827891, 0.034272765267548827, 0.027321341782924492, 0.021784495261745401, 0.015624002114793886, 0.012098188273298383, 0.010376530429148649, 0.0092253944777410841, 0.0069292856613297994, 0.0067790587243023318, 0.0058628296826901145, 0.0053095873654085942, 0.0045792891819049593, 0.0038609047881879568, 0.0037107013095589434, 0.0028402911129245148, 0.0022969303273648561, 0.00052554057121693173]
        
        # energy bin edges for integration
        n_samples = 250
        e = np.logspace(np.log10(eMin),np.log10(eMax),n_samples+1)
        ave = 0.5*(e[:-1]+e[1:]) # average energy in bin
        width = e[1:]-e[:-1] #bin widths
        
        dnde2 = lambda x : dnde(x)*flux/np.sum(dnde(ave)*width)# normalize the integrated dnde to the given flux.
        # convolve with effective area
        effArea = lambda x: np.interp(x,EAE,np.array(EA))*1000.*time 
        # This is the reshaped dn/de taking into account effective area etc..
        dist = lambda x: effArea(x)*dnde2(x)
        
        # reintegrate new distribution
        N_phot = np.random.poisson(np.sum(dist(ave)*width))
        # This gives the number of photons.  Now we need to sample from dist
        E = self.SampleDist(dist,n=N_phot, a=eMin,b=eMax)
        return E
    
    def AddPointSource(self,n=-1, E=-1, T='rand', l='rand',b='rand'):
        """
        Add a point source with n photons and time and energies given by T,E.
        A point spread function will automatically be applied at the energy weighted mean energy, thus if simulating over many energies, should be done in chunks.
        @param n Number of photons.  If unspecified, taken to be len(E). (and E must be specified)
        @param E Energies in MeV If left unspecified will distributed according to power law E^-2.5 between eMin and eMax with n photons. numpy.ndarray shape(n) 
        @param T Times in seconds.  If left unspecified will distribute uniformly over SimTools.time. numpy.ndarray shape(n) 
        @param l Galactic longitude of centroid.  If left unspecified will choose random direction. float 
        @param b Galactic latitude of centroid.  If left unspecified will choose random direction. float
        @returns (b,l,T,E) of point source coordinates.
        """
        # Check energies.
        if (type(E)!=int) and (min(E)<self.eMin or max(E)>self.eMax):
            # otherwise load the updated psf between the given energies
            eMin,eMax = min(E)-1e-5,max(E)+1e-5
            nSteps = int(np.ceil((np.log10(eMax)-np.log10(eMin))/0.25))
            psfbins = np.logspace(np.log10(eMin),np.log10(eMax),nSteps+1)
            psf = [FermiPSF.GetPSF(psfbins[i],psfbins[i+1],convType=self.convType)[1] for i in range(len(psfbins)-1)]
            if len(psfbins)==0:
                psf = (FermiPSF.GetPSF(eMin,eMax,convType=self.convType)[1],)
            theta = FermiPSF.GetPSF(eMin,eMax,convType=self.convType)[0]
        else:
            eMin,eMax = self.eMin,self.eMax
            # if using the preset energy range, don't need to reload psf
            theta, psf,psfbins = self.theta, self.psf,self.psfbins
        
        # Sample the energy spectrum if not provided
        if type(E)==int:
            if E ==-1: E = self.SampleE(eMin,eMax,n)
        if n==-1: 
            try: n=len(E)
            except: raise ValueError('Need to specify n or a vector E.')

        if str(T) =='rand': T = np.random.randint(0,high=self.time,size=n)
        if str(l) =='rand': l=np.random.ranf()*360.
        if str(b) =='rand': b=np.rad2deg(np.arccos(2*np.random.ranf()-1))-90
        #=============================================================
        # Here we apply the fermi point spread function.
        # Inverse monte carlo sampling of psf to obtain r
        # Get the energy averaged psf (with weighting ~ E^-2.5)
        #=============================================================
        dY,dZ = np.zeros(n),np.zeros(n)
        # bin the energies
        e = np.digitize(E,psfbins)
        # make a list of bins containing samples
        ue = np.unique(e)-1
        # for each bin
        for i in range(len(ue)):
            # find which points are in this energy bin
            idx = np.where(e-1==ue[i])[0]   
            psfcum = np.cumsum(psf[ue[i]]) # Obtain CDF
            # Invert histogram and sample
            r = theta[np.argmin(np.abs(np.transpose(np.ones((n,len(psfcum)))*psfcum)-np.random.ranf(n)),axis=0)]
            phi = 2*np.pi*np.random.ranf(n) # Random Angle 
            # Find X and Y displacements
            dY[idx],dZ[idx] = np.deg2rad(r*np.cos(phi)),np.deg2rad(r*np.sin(phi))
        # normalize the vectors.  Now (dx,dy,dz) can be rotated to correct galactic coords.
        dX = np.sqrt(1-dZ*dZ-dY*dY)
        # First rotate about y-axis to the correct lat.
        ny = np.array([0.,1.,0.])
        nz = np.array([0.,0.,1.])
        theta2,theta1 = np.deg2rad((l,b))    
        R1 = self.__rotation_matrix(axis=ny,theta=theta1) # construct the rotation matrix
        # The second rotation will move to the correct longitude
        #R2 = self.__rotation_matrix(axis=nz,theta = theta2)
        R2 = self.__rotation_matrix(axis=nz,theta =-theta2)
        R  = np.dot(R2,R1) # construct full rotation matrix 
        def rotate(n):
            #n = n/np.sqrt(np.dot(n,n))
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
        
