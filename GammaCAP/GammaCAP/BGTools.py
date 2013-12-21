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
            path = os.path.join(os.path.dirname(__file__), 'isotrop_4years_P7_v9_repro_source_v1.txt')
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
        EA = [0.51955367294747401, 0.53725310619600974, 0.56345609912606598, 0.59517718051541169, 0.60033182312125866, 0.59519701184785057, 0.61214804337069662, 0.60353362437011404, 0.59994041227554407, 0.61993746469187494, 0.61413705054359657, 0.62097389810831327, 0.61257477655091908, 0.60819360887649077, 0.61752099476440225, 0.61023974408392867, 0.64322968803451086, 0.66633671088141866, 0.7100934576526241, 0.72296484727761268, 0.77309826395497705, 0.79985117267717099, 0.86077869571883225, 0.83409234808019261, 0.80101651548441866, 0.88960586977831546, 0.82726449338741614, 0.83745725896659651, 0.28095470948365214]
        if self.convType=='front':
            EA = [0.27705384135395728, 0.28746741654944713, 0.30266349536113346, 0.3198112573790593, 0.32337358308150693, 0.32178752634074059, 0.33112541170349363, 0.3291900126295017, 0.32994917960478631, 0.34034233714956919, 0.34001599946144395, 0.35082953113032084, 0.34421329403821299, 0.33567575485071593, 0.3419835899058184, 0.33193906295843179, 0.35191189454062011, 0.35903789833375266, 0.37227060541766566, 0.39122898637283027, 0.39827420349539666, 0.42703918541238789, 0.45259616924773977, 0.43695291265776198, 0.43353986662816474, 0.47248776486151195, 0.43038370839781459, 0.45936474120530607, 0.17438568174847374]
        if self.convType=='back':
            EA = [0.24249983159351671, 0.24978568964656264, 0.26079260376493257, 0.27536592313635244, 0.27695824003975178, 0.27340948550711003, 0.28102263166720298, 0.27434361174061239, 0.2699912326707577, 0.27959512754230575, 0.27412105108215257, 0.27014436697799243, 0.26836148251270608, 0.27251785402577478, 0.27553740485858391, 0.27830068112549688, 0.29131779349389086, 0.30729881254766606, 0.33782285223495839, 0.33173586090478252, 0.37482406045958033, 0.37281198726478304, 0.40818252647109243, 0.39713943542243058, 0.36747664885625386, 0.41711810491680351, 0.3968807849896015, 0.37809251776129038, 0.1065690277351784]
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
            energies,N = np.transpose(np.genfromtxt(self.iso_f, delimiter=None,autostrip=True))
        except:
            raise ValueError('Isotropic diffuse model does not have 2 columns.')
            
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



