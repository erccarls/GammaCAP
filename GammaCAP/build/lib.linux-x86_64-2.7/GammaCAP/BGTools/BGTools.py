"""@package BGTools
Tools for estimating cluster significances and background counts based on the Fermi Collaboration's diffuse galactic and isotropic background models.
"""

import pyfits
import numpy as np

class BGTools:
    """
    Tools for estimating cluster significances and background counts based on the Fermi Collaboration's diffuse galactic and isotropic background models.
    """
    def __init__(self,Emin,Emax,Time,diff_f, iso_f):
        """ Initializes the background map.
        @param Emin  Minimum energy in MeV.
        @param Emax  Maximum energy in MeV.
        @param Time   Total Integration time in seconds.
        @param diff_f Abosulte path to diffuse BG model (typically '$FERMI_DIR/refdata/fermi/galdiffuse/gll_iem_v05.fits') where $FERMI_DIR is the Fermi science tools installation path.
        @param iso_f Abosulte path to diffuse BG model (typically '$FERMI_DIR/refdata/fermi/galdiffuse/isotrop_4years_P7_v9_repro_clean_v1.txt') where $FERMI_DIR is the Fermi science tools installation path.
        """
        ## @var self.Emin testing
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
        @param Time   Total Integration time in seconds.
        @param diff_f Abosulte path to diffuse BG model (typically '$FERMI_DIR/refdata/fermi/galdiffuse/gll_iem_v05.fits') where $FERMI_DIR is the Fermi science tools installation path.
        @param iso_f Abosulte path to diffuse BG model (typically '$FERMI_DIR/refdata/fermi/galdiffuse/isotrop_4years_P7_v9_repro_clean_v1.txt') where $FERMI_DIR is the Fermi science tools installation path.        
        """ 
        if E_min<50 or E_max>6e5: print "Seems like your energies are out of range. Should be 50 MeV to 600 GeV in units MeV"
        # Calculate a list of the energies corresponding to the diffuse galactic model
        energies = np.logspace(np.log10(50),np.log10(6e5),31)
        # Calc indicies we care about 
        emin,emax = np.argmax(energies>E_min)-1, np.argmax(energies>E_max)
        if emax==0:emax=30
        # Now compute the average energies in each bin
        energies = np.array([np.mean(energies[i:i+2]) for i in range(len(energies)-1)])
        
        # Interpolate effective area.  Empiracally fudged to agree with  gtobssim simulations.
        EAE =[1000.0, 1579.2081441197518, 2493.8983624541534, 3938.3846045945115, 6219.5290422515081,
              9821.9309161129022, 15510.873293707069, 24494.897427831787, 38682.541507409958, 
              61087.784583752211, 96470.326920894411, 152346.72593937156, 240587.19033343517, 379937.25034545001]
        EA = [0.56474960595679269, 0.61900037984106093, 0.6477423909408575, 0.6494559139904782,
              0.65105871317624175, 0.65634657600811919, 0.64493619600906349, 0.65972804516431405,
              0.70880567499919545, 0.77383792531196494, 0.88505388177027788, 0.84500602665697677, 
              0.93898153929664407, 0.59507693678706786]
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
        hdulist = pyfits.open(diff_f, mode='update')
        scidata = hdulist[0].data[emin:emax]
        scidata = [scidata[i]*weights[i] for i in range(len(scidata))]
        # Load the isotropic model
        energies,N = np.transpose(np.genfromtxt(iso_f, delimiter=None,autostrip=True))
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
        N_bg  =  self.GetIntegratedBG(l=cy,b=cx, A=CR.Size95X, B=CR.Size95Y)# Evaluate the background density at that location
        N_bg  = N_bg*2.*CR.Size95T/12.*3.15e7/self.Time # Find ratio of cluster time length to total exposure time
        N_cl  = (0.95*CR.Members) # 95% containment radius so only count 95% of members
        ######################################################
        # Evaluate significance as defined by Li & Ma (1983).  N_cl corresponds to N_on, N_bg corresponds to N_off
        S2 = np.zeros(len(N_cl))
        idx = np.where(np.logical_and(N_cl/(N_cl+N_bg)>0, N_bg/(N_cl+N_bg)>0))[0]
        N_cl, N_bg = N_cl[idx], N_bg[idx]
        S2[idx] = 2.0*(N_cl*np.log(2.0*N_cl/(N_cl+N_bg)) + N_bg*np.log(2.0*N_bg/(N_cl+N_bg)))
        return np.sqrt(S2)   



