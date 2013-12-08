"""@package BGTools
Tools for estimating cluster significances and background counts based on the Fermi Collaboration's diffuse galactic and isotropic background models.
"""

import pyfits
import numpy as np
import os

class BGTools:
    """
    Tools for estimating cluster significances and background counts based on the Fermi Collaboration's diffuse galactic and isotropic background models.
    """
    def __init__(self,Emin,Emax,Time,diff_f='', iso_f=''):
        """ Initializes the background map.
        @param Emin  Minimum energy in MeV.
        @param Emax  Maximum energy in MeV.
        @param Time   Total Integration time in seconds.
        @param diff_f Abosulte path to diffuse BG model (typically '$FERMI_DIR/refdata/fermi/galdiffuse/gll_iem_v05.fits') where $FERMI_DIR is the Fermi science tools installation path.
        @param iso_f Abosulte path to diffuse BG model (typically '$FERMI_DIR/refdata/fermi/galdiffuse/isotrop_4years_P7_v9_repro_clean_v1.txt') where $FERMI_DIR is the Fermi science tools installation path.
        """
        #TODO: Recalibrate BGTemplate normalization with gtselect filtering
        ##@var Emin  
        # Minimum energy in MeV.
        ##@var Emax  
        # Maximum energy in MeV.
        ##@var Time   
        # Total Integration time in seconds.
        ##@var diff_f 
        # Abosulte path to diffuse BG model (typically '$FERMI_DIR/refdata/fermi/galdiffuse/gll_iem_v05.fits') where $FERMI_DIR is the Fermi science tools installation path.
        # if left as empty string "" will attempt to locate it automatically if $FERMI_DIR is a valid environmental variable pointing to the fermi science tools directory.  Otherwise can be downloaded 
        # (instructions at http://planck.ucsc.edu/gammacap).
        ##@var iso_f 
        # Abosulte path to isotropic BG model (typically '$FERMI_DIR/refdata/fermi/galdiffuse/isotrop_4years_P7_v9_repro_clean_v1.txt') where $FERMI_DIR is the Fermi science tools installation path.
        # if left as empty string "" will attempt to locate it automatically if $FERMI_DIR is a valid environmental variable pointing to the fermi science tools directory.  Otherwise can be downloaded 
        # (instructions at http://planck.ucsc.edu/gammacap)
        ##@var BGMap
        # Contains a 2-d array with the diffuse galactic and isotropic backgrounds integrated over the energies and times specified during initialization.  Map units are photons/deg^2 and the effective area etc.. 
        # have been empirically determined to agree with gtobssim.  If the diffuse and isotropic models are used with an instrument other than Fermi-LAT, such as CTA or Veritas, the normalizations of each energy 
        # band should be matched in the BGTools.__Prep_Data() routine.   
        self.Emin   = Emin
        self.Emax   = Emax
        self.Time   = Time
        self.diff_f = diff_f
        self.iso_f  = iso_f
        self.BGMap  = self.__Prep_Data(Emin,Emax,Time,diff_f,iso_f)
    
    
    def __Prep_Data(self,E_min,E_max,Time,diff_f, iso_f):
        """
        Returns a numpy array with a sky map of the number of photons per square degree.
        @param Emin  Minimum energy in MeV.
        @param Emax  Maximum energy in MeV.
        @param Time  Total Integration time in seconds.
        @param diff_f Abosulte path to diffuse BG model (DEFAULT '$FERMI_DIR/refdata/fermi/galdiffuse/gll_iem_v05.fits') where $FERMI_DIR is the Fermi science tools installation path.
        @param iso_f Abosulte path to diffuse BG model (DEFAULT '$FERMI_DIR/refdata/fermi/galdiffuse/isotrop_4years_P7_v9_repro_clean_v1.txt') where $FERMI_DIR is the Fermi science tools installation path.        
        """ 
        if E_min<50:  raise ValueError("High Energy must be >= than 50 GeV in units MeV")
        if E_max>6e5: raise ValueError("High Energy must be <= than 600 GeV in units MeV")
        
        ###############################################################
        # Check if the diffuse model path has been specified.  If not, try to locate in fermitools setup.
        ###############################################################
        if diff_f == '':
            try:
                fermi_dir = os.environ['FERMI_DIR']
            except: raise KeyError('It appears that Fermitools is not setup or $FERMI_DIR environmental variable is not setup.  This is ok, but you must specify a path to the galactic diffuse model in the Scan.diffModel setting and isotropic model in Scan.isoModel') 
            path = fermi_dir + '/refdata/fermi/galdiffuse/gll_iem_v05.fits'
            if os.path.exists(path)==True: diff_f = path
            else: raise ValueError('Fermitools appears to be setup, but cannot find diffuse model at path ' + path + '.  Please download to this directory or specify the path in Scan.diffModel' )
        if iso_f == '':
            path = fermi_dir + '/refdata/fermi/galdiffuse/isotrop_4years_P7_v9_repro_clean_v1.txt'
            if os.path.exists(path)==True: iso_f = path
            else: raise ValueError('Fermitools appears to be setup, but cannot find diffuse model at path ' + path + '.  Please download or specify an alternate path in Scan.isoModel.' ) 
        
        
        
        # Calculate a list of the energies corresponding to the diffuse galactic model
        energies = np.logspace(np.log10(50),np.log10(6e5),31)
        # Calc indicies we care about 
        emin,emax = np.argmax(energies>E_min)-1, np.argmax(energies>E_max)
        if emax==0:emax=30
        # Now compute the average energies in each bin
        energies = np.array([np.mean(energies[i:i+2]) for i in range(len(energies)-1)])
        
        # Interpolate effective area.  Empirically fudged to agree with  gtobssim simulations.
        EAE =[1000.0, 1178.2423505388722, 1388.2550366033665, 1635.700877474977, 1927.2520466546125, 2270.769981531183, 2675.5173605724094, 3152.407863828395, 3714.3004513343931, 4376.346094387829, 5156.3963090231291, 6075.4845074533696, 7158.3931467243601, 8434.3219672778632, 9937.6753399271056, 11708.989951407897, 13796.027842782876, 16255.064273580221, 19152.405137863607, 22566.174848109185, 26588.422895707346, 31327.605889759787, 36911.512000305913, 43490.706661184231, 51242.592443070287, 60376.192567828584, 71137.787047705933, 83817.553423222605, 98757.391161795298, 116360.14069556053, 137100.44568217112, 161537.55138048826, 190330.38423884034, 224255.31930453796, 264227.11453822412, 311323.57650962099, 366814.62256486423, 432196.52310285484, 509232.24727543467]
        EA = [0.52036753187409324, 0.49307283511929845, 0.551804598116726, 0.52198010999911815, 0.59351418722030724, 0.5453142420485031, 0.59966209057237363, 0.55361869554688314, 0.61384906001730077, 0.54991232230725362, 0.60472745950396678, 0.56473971717852844, 0.62158960165951527, 0.56872126617220475, 0.62347326376674439, 0.59604165934578934, 0.62027434477916565, 0.60792228472335774, 0.61520808459002652, 0.61216533579415622, 0.61028260222486375, 0.63875497160013972, 0.603130938535548, 0.65519470340508013, 0.64922827311595976, 0.71297330489036181, 0.67331585966560892, 0.78395664077353622, 0.70620028253076084, 0.85478922335349761, 0.78918696281774015, 0.82933312639374024, 0.72262428478784013, 0.85987205364782182, 0.84660204715406784, 0.86991569204303021, 0.79324745991535839, 0.73671954673231432, 0.13621062441760959]
        #EA = np.ones(len(EAE))
        
        effArea = np.interp(energies,EAE,np.array(EA))
        
        #Determine weights to convert flux to photon counts
        energies = np.logspace(np.log10(50),np.log10(6e5),31)
        if emin==emax:emax+=1
        weights = [(energies[i+1]-energies[i]) for i in range(emin,emax)] 
        # Endpoints need to be reweighted if they don't align with the template endpoints.
        weights[0]*=(energies[emin+1]**-1.5-E_min**-1.5)/(energies[emin+1]**-1.5-energies[emin]**-1.5)
        weights[-1]*=(E_max**-1.5-energies[emax-1]**-1.5)/(energies[emax]**-1.5-energies[emax-1]**-1.5)
        weights = np.multiply(np.array(weights)*Time,effArea[emin:emax])

        # Load the diffuse background model
        try:
            hdulist = pyfits.open(diff_f, mode='update')
        except :
            raise ValueError('Invalid path for galactic diffuse model.')
            
        scidata = hdulist[0].data[emin:emax]
        scidata = [scidata[i]*weights[i] for i in range(len(scidata))]
        # Load the isotropic model
        try:
            energies,N = np.transpose(np.genfromtxt(iso_f, delimiter=None,autostrip=True))
        except:
            raise ValueError('Invalid path for isotropic diffuse model.')
            
        isotrop = np.multiply(N[emin:emax],weights)
        # Sum the photon counts
        total=np.zeros(shape=np.shape(scidata[0]))
        for i in range(len(scidata)):
            total = np.add(total,scidata[i]+isotrop[i])
        return total
 
    def GetBG(self,l,b):
        """
        Given a latitude and longitude vector, return the number of photons/sq-deg evaluated at the center of each point.
        @param l longitude vector np.array:shape (n,1)
        @param b latitude vector  np.array:shape (n,1)
        @return evt Number of photons/deg^2 at the input coordinates. np.array:shape (n,1)
        """
        # if integer, need to convert to array
        if np.shape(l)==():    
            l = np.array((l,))
            b = np.array((b,))
        # Find size of BG map
        len_b,len_l = np.shape(self.BGMap)
        # Map the input coords onto the background model
        l_idx = np.divide((np.array(l)+180.)%360,360./float(len_l)).astype(int)
        b_idx = np.divide(np.array(b)+90.,180./float(len_b)).astype(int)
        # Bounds checking on lat.  longitude is handled by modulo operator above
        b_idx[np.where(b_idx==len_b)[0]] = len_b-1
        return self.BGMap[b_idx,l_idx]
    
    ##

    def SubsampleBG(self,l,b,eps):
        """
        Given a lat and long vector, return a vector with the number of photons/sq-deg at that point computed by subsampling within the epsilon radius points. 
        @param l longitude vector np.array:shape (n,1)
        @param b latitude vector  np.array:shape (n,1)
        @param eps The DBSCAN search radius Epsilon
        @return evt Number of photons/deg^2 at the input coordinates. np.array:shape (n,1)
        """
        if np.shape(l)==():    
            l = np.array((l,))
            b = np.array((b,))
        l=l.astype(float)
        b=b.astype(float)
        def get(l,b):
            up = np.where(b>90)[0]
            down = np.where(b<-90)[0]
            l[np.append(up,down)] += 180. # flip meridian
            b[up]=-b[up]%90.        # invert bup
            b[down]=90.-b[down]%90.
            
            l = l%360.
            return self.GetBG(l, b)
        sh = eps/2. # shift
        rate = [get(l,b), get(l+sh,b-sh),
                get(l-sh,b), get(l+sh,b),get(l,b-sh)]
            
        return np.mean(rate)
        
    def GetIntegratedBG(self, l, b, A, B):
        """
        Given a lat and long vector, return a vector with the number of photons expected at that point computed by integrating over an ellipse.  **Note: Currently averages rate over square circumscribed by circle with radius=semi-major axis and then multiplies by ellipse area.
        @param l longitude vector np.array:shape (n,1).
        @param b latitude vector  np.array:shape (n,1).
        @param A Semimajor Axis in deg   np.array:shape (n,1).
        @param B Semiminor Axis in deg   np.array:shape (n,1).
        @return evt Total number of photons at the input coordinates. np.array:shape (n,1).
        """
        # if integer, need to convert to array
        if np.shape(l)==():    
            l = np.array((l,))
            b = np.array((b,))
        len_b,len_l = np.shape(self.BGMap)
        # Map the input coords onto the background model
        l_idx = np.divide((np.array(l)+180.)%360,360./float(len_l)).astype(int)
        b_idx = np.divide(np.array(b)+90.,180./float(len_b)).astype(int)
        a2 = np.sqrt(2)*A/2. # square enclosed by the circle.
        scales = np.abs(1./np.cos(np.deg2rad(b))) # amount we must expand longitude as a function of lat
        ipd    = len_l/360.   # how many index increments per degree 
        l_start, l_stop =  np.array(l_idx - ipd*a2*scales).astype(int), np.array(l_idx+ipd*a2*scales+1).astype(int)
        b_start, b_stop =  np.array(b_idx - ipd*a2).astype(int), np.array(b_idx+ipd*a2+1).astype(int)
        
        #TODO: Actually integrate over ellipse
        allidx = np.where((l_stop-l_start)>len_l)[0] # in case scale blows up just set to full length
        l_start[allidx], l_stop[allidx] = 0,len_l
        rate = np.zeros(len(l_start))
        
        for i in range(len(l_start)):
            l_slice = (l_start[i]<0 or l_stop[i]>len_l)
            b_slice = (b_start[i]<0 or b_stop[i]>len_b)
            
            # if all within bounds, give the mean rate of that square
            if (l_slice==False and b_slice==False):
                rate[i] = np.mean(self.BGMap[b_start[i]:b_stop[i],l_start[i]:l_stop[i]])
                
            # Otherwise need to use some indexing tricks to span boundaries.  This is why integrations through poles are slow
            # could speedup with cython if needed.  Still only ~1ms per circle 
            else:
                # For longitude roll the longitude indices around to the beginning using mod(len_l) (happens at end)
                
                l_idx = np.arange(l_start[i],l_stop[i])%len_l
                # For lat need to shift the longitudes where b_idx >= len_b or b_idx<0 by 180deg and then % 360 deg
                b_idx = np.arange(b_start[i],b_stop[i])
                #print b_idx[i], len_b
                up    = np.where(b_idx>=len_b)[0] # where > 90 deg we must flip over the meridian
                down  = np.where(b_idx<0)[0]      # where < -90 deg we must flip over the meridian
                normal= np.where(np.logical_and(b_idx>=0,b_idx<len_b))[0] # where lat is within range do nothing
                
                b_idx[down]   = -b_idx[down]      # invert the latitudes 
                b_idx[up]     = -b_idx[up]%len_b  # invert the latitudes 
                # Average each of the three squares mena rates with weights equal to area. 
                # (note widths all the same so just weight by height)                
                rate[i] = np.average( np.nan_to_num([np.mean(self.BGMap[b_idx[normal]][:,l_idx]),
                                       np.mean(self.BGMap[b_idx[up]][:,(l_idx+len_l/2)%len_l]),
                                       np.mean(self.BGMap[b_idx[down]][:,(l_idx+len_l/2)%len_l])]),
                                     weights=[len(normal),len(up),len(down)])
                
        # Finally, multiply by the ellipse area.
        return rate*np.pi*A*B
    
    def SigsBG(self, CR):
        """
        Returns Significance vector based on integration of the background template for each cluster.
        @param CR ClusterResult output from DBSCAN.RunDBScan3D
        @return Sigs Significances of each cluster: ndarray shape(n,1)
        """
        cx,cy = CR.CentX, CR.CentY # Get centroids
        N_bg  = self.GetIntegratedBG(l=cy,b=cx, A=CR.Size95X, B=CR.Size95Y)# Evaluate the background density at that location
        N_bg  = N_bg*2.*CR.Size95T/self.Time # Find ratio of cluster time length to total exposure time
        N_cl  = (0.95*CR.Members) # 95% containment radius so only count 95% of members
        ######################################################
        # Evaluate significance as defined by Li & Ma (1983).  N_cl corresponds to N_on, N_bg corresponds to N_off
        S2 = np.zeros(len(N_cl))
        idx = np.where(np.logical_and(N_cl/(N_cl+N_bg)>0, N_bg/(N_cl+N_bg)>0))[0]
        N_cl, N_bg = N_cl[idx], N_bg[idx]
        S2[idx] = 2.0*(N_cl*np.log(2.0*N_cl/(N_cl+N_bg)) + N_bg*np.log(2.0*N_bg/(N_cl+N_bg)))
        return np.sqrt(S2)   



