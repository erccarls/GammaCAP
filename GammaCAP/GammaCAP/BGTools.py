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
    def __init__(self,Emin,Emax,Time,diff_f='', iso_f='',convType='both'):
        """ Initializes the background map.
        @param Emin  Minimum energy in MeV.
        @param Emax  Maximum energy in MeV.
        @param Time   Total Integration time in seconds.
        @param diff_f Abosulte path to diffuse BG model (typically '$FERMI_DIR/refdata/fermi/galdiffuse/gll_iem_v05.fits') where $FERMI_DIR is the Fermi science tools installation path.
        @param iso_f Abosulte path to diffuse BG model (typically '$FERMI_DIR/refdata/fermi/galdiffuse/iso_source_v05.txt') where $FERMI_DIR is the Fermi science tools installation path.
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
        # Abosulte path to isotropic BG model (typically '$FERMI_DIR/refdata/fermi/galdiffuse/iso_source_v05.txt') where $FERMI_DIR is the Fermi science tools installation path.
        # if left as empty string "" will attempt to locate it automatically if $FERMI_DIR is a valid environmental variable pointing to the fermi science tools directory.  Otherwise can be downloaded 
        # (instructions at http://planck.ucsc.edu/gammacap)
        ##@var BGMap
        # Contains a 2-d array with the diffuse galactic and isotropic backgrounds integrated over the energies and times specified during initialization.  Map units are photons/deg^2 and the effective area etc.. 
        # have been empirically determined to agree with gtobssim for the Pass 7 'Source' event class only.  If the diffuse and isotropic models are used with an instrument other than Fermi-LAT, such as CTA or Veritas, the normalizations of each energy 
        # band should be matched in the BGTools.__Prep_Data() routine. 
        ##@var convType
        # Fermi-LAT conversion type.  Can be 'front','back', or 'both' (default). 'front'/'back' simply cut down effective area to 58/42 percent which is a reasonable approximation over 1-300GeV.  No detailed energy dependence is taken into account.
        self.Emin     = Emin
        self.Emax     = Emax
        self.Time     = Time
        self.diff_f   = diff_f
        self.iso_f    = iso_f
        self.convType = convType
        self.BGMap    = self.__Prep_Data()

    
    
    def __Prep_Data(self):
        """
        Returns a numpy array with a sky map of the number of photons per square degree.
        @param Emin  Minimum energy in MeV.
        @param Emax  Maximum energy in MeV.
        @param Time  Total Integration time in seconds.
        @param diff_f Abosulte path to diffuse BG model (DEFAULT '$FERMI_DIR/refdata/fermi/galdiffuse/gll_iem_v05.fits') where $FERMI_DIR is the Fermi science tools installation path.
        @param iso_f Abosulte path to diffuse BG model (DEFAULT '$FERMI_DIR/refdata/fermi/galdiffuse/iso_source_v05.txt') where $FERMI_DIR is the Fermi science tools installation path.        
        """ 
        if float(self.Emin)<50:  raise ValueError("High Energy must be >= than 50 GeV in units MeV")
        if int(self.Emax)>int(6e5): raise ValueError("High Energy must be <= than 600 GeV in units MeV")
        
        ###############################################################
        # Check if the diffuse model path has been specified.  If not, try to locate in fermitools setup.
        ###############################################################
        if self.diff_f == '':
            try:
                fermi_dir = os.environ['FERMI_DIR']
            except: raise KeyError('It appears that Fermitools is not setup or $FERMI_DIR environmental variable is not setup.  This is ok, but you must specify a path to the galactic diffuse model in the Scan.diffModel setting and isotropic model in Scan.isoModel') 
            path = fermi_dir + '/refdata/fermi/galdiffuse/gll_iem_v05.fits'
            if os.path.exists(path)==True: self.diff_f = path
            else: raise ValueError('Fermitools appears to be setup, but cannot find diffuse model at path ' + path + '.  Please download to this directory or specify the path in Scan.diffModel' )
        if self.iso_f == '':
            path = os.path.join(os.path.dirname(__file__), 'iso_source_v05.txt')
            #path = fermi_dir + '/refdata/fermi/galdiffuse/isotrop_4years_P7_v9_repro_source_v1.txt'
            if os.path.exists(path)==True: self.iso_f = path
            else: raise ValueError('Fermitools appears to be setup, but cannot find diffuse model at path ' + path + '.  Please download or specify an alternate path in Scan.isoModel.' ) 
        
        
        
        # Calculate a list of the energies corresponding to the diffuse galactic model
        energies = np.logspace(np.log10(50),np.log10(6e5),31)
        # Calc indicies we care about 
        emin,emax = np.argmax(energies>self.Emin)-1, np.argmax(energies>self.Emax)
        if emax==0:emax=30
        # Now compute the average energies in each bin
        energies = np.array([np.mean(energies[i:i+2]) for i in range(len(energies)-1)])
        
        # Interpolate effective area.  Empirically fudged to agree with  gtobssim simulations.
        EAE =[1000.0, 1246.8043074919699, 1554.5209811805289, 1938.1834554225268, 2416.5354809304745, 3012.9468468312944, 3756.5551068736063, 4683.6890885809644, 5839.6437305958789, 7280.8929575254178, 9077.8487018306387, 11318.300864202816, 14111.706270978171, 17594.536164717007, 21936.943478492387, 27351.075622192155, 34101.438900287758, 42517.820912553041, 53011.402258943228, 66094.844682639901, 82407.337053328229, 102745.82280703213, 128103.93445261421, 159720.53728218854, 199140.25387836422, 248288.9263305887, 309567.7028515347, 385970.34537568642, 481229.48917856964]
        EA = [0.47830102293783294, 0.50143612236708124, 0.51736569365083429, 0.54004038539226262, 0.53855253320473639, 0.53067429512286857, 0.53905650630349522, 0.52082209207963281, 0.49576142465826778, 0.48582457415814617, 0.46534497049039225, 0.45394337499055176, 0.42252751654605225, 0.41288303748671856, 0.41399327554334742, 0.40868786754166897, 0.41369710265025139, 0.41180600560890945, 0.41510418174510672, 0.3789459717284655, 0.37772486337413386, 0.3647448012049429, 0.36539142705092698, 0.34270615826499029, 0.32129504973010414, 0.33778423949575176, 0.30254270311293796, 0.28210576898258521, 0.085355107981324757]
        if self.convType=='front':
            EA = [0.25649604207846322, 0.26992857478824889, 0.27955568369558764, 0.29189235667186109, 0.29175355637991224, 0.28852097170939933, 0.29313907073896833, 0.28551232552342598, 0.27394743531455168, 0.26757021296517375, 0.25872204886245243, 0.2573133451325334, 0.23823932492403188, 0.22846936498964057, 0.22993351948255242, 0.22287858340852426, 0.22676681537417168, 0.22254520595459984, 0.21773693148370998, 0.20550309366219757, 0.19500278660891243, 0.195185169390695, 0.19229548829199031, 0.18060984811324221, 0.17389680526629347, 0.18090536675923108, 0.15838442451366969, 0.15431426679389279, 0.053908489251363011]
        if self.convType=='back':
            EA = [0.22180498085936973, 0.23150754757883232, 0.23781000995524665, 0.24814802872040151, 0.24679897682482418, 0.24215332341346926, 0.24591743556452686, 0.23530976655620686, 0.22181398934371613, 0.21825436119297242, 0.20662292162793974, 0.19663002985801836, 0.18428819162202043, 0.18441367249707802, 0.18405975606079503, 0.18580928413314471, 0.18693028727607969, 0.18926079965430964, 0.19736725026139676, 0.17344287806626793, 0.1827220767652214, 0.16955963181424791, 0.17309593875893667, 0.16209631015174811, 0.14739824446381064, 0.15687887273652071, 0.14415827859926825, 0.12779150218869245, 0.03144661872996176]
        #EA = np.ones(len(EAE))
        
        effArea = np.interp(energies,EAE,np.array(EA))
        
        #Determine weights to convert flux to photon counts
        energies = np.logspace(np.log10(50),np.log10(6e5),31)
        if emin==emax:emax+=1
        weights = [(energies[i+1]-energies[i]) for i in range(emin,emax)] 
        # Endpoints need to be reweighted if they don't align with the template endpoints.
        weights[0]*=(energies[emin+1]**-1.5-self.Emin**-1.5)/(energies[emin+1]**-1.5-energies[emin]**-1.5)
        weights[-1]*=(self.Emax**-1.5-energies[emax-1]**-1.5)/(energies[emax]**-1.5-energies[emax-1]**-1.5)
        weights = np.multiply(np.array(weights)*self.Time,effArea[emin:emax])

        # Load the diffuse background model
        try:
            hdulist = pyfits.open(self.diff_f, mode='update')
        except :
            raise ValueError('Invalid path for galactic diffuse model.')
            
        scidata = hdulist[0].data[emin:emax]
        scidata = [scidata[i]*weights[i] for i in range(len(scidata))]
        # Load the isotropic model
        try:
            energies_iso,N, junk = np.transpose(np.genfromtxt(self.iso_f, delimiter=None,autostrip=True))
        except:
            raise ValueError('Isotropic diffuse model does not have 3 columns.')
        N = np.interp(energies, energies_iso, N) 
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
                # Average each of the three squares mean rates with weights equal to area. 
                # (note widths all the same so just weight by height)                
                
                # Works but gives runtime errors.
                #rate[i] = np.average( np.nan_to_num([np.mean(self.BGMap[b_idx[normal]][:,l_idx]),
                #                       np.mean(self.BGMap[b_idx[up]][:,(l_idx+len_l/2)%len_l]),
                #                       np.mean(self.BGMap[b_idx[down]][:,(l_idx+len_l/2)%len_l])]),
                #                     weights=[len(normal),len(up),len(down)])
                
                
                avg = np.mean(self.BGMap[b_idx[normal]][:,l_idx])*len(normal)
                if len(up)!=0  : avg+=np.mean(self.BGMap[b_idx[up]]  [:,(l_idx+len_l/2)%len_l])  *len(up)
                if len(down)!=0: avg+=np.mean(self.BGMap[b_idx[down]][:,(l_idx+len_l/2)%len_l])*len(down)
                avg/=float(len(normal)+len(up)+len(down))
                rate[i]=avg
                
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



