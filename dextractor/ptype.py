#import yt
#import os
#import time
import numpy as np
#from tqdm import tqdm
#from pytreegrav import Potential, PotentialTarget
#from unyt import unyt_array, unyt_quantity
#from unyt.physical_constants import G
from copy import copy

from .config import config
from .base import BaseSimulationObject

from class_methods import center_of_mass, refine_center, half_mass_radius






class ptype(BaseSimulationObject):
    """ptype class that contains the particle data, for each particle type present/relevant to your analysis, in the simulation. 
    Fields are stored as attributes and have units thanks to unyt. The class might have as much fields as wished by the user,
    but a few are mandatory:
        
    Mandatory fields
    ----------------
    - coordinates : stored as self.coords
    - velocities : stored as self.vels
    - masses : stored as self.masses
    - IDs : stored as self.ids
    
    IMPLEMENTATION OF MANDATORY FIELDS FOR EACH PARTICLE TYPE. MIGHT SEPARATE INTO THREE PTYPES.
    IMPLEMENTATION OF DYNAMICAL FIELD LOADING
    """

    ##########################################################
    ###                                                    ###
    ##                    INNIT FUNCTION                    ##
    ###                                                    ###
    ##########################################################
    
    def __init__(self,
                 data,
                 pt
                 ):
        """Initializes the ptype class.
        """
        super().__init__()
        self.ptype, self._base_ptype = pt, copy(self.ptypes[pt])
        self._fields = copy(self.fields[pt])
        self._fields_loaded = {}
        self._data = data
        
        self.cm = None
        self.vcm = None
        self.rh = None
        self.rh_3D = None

        if self.ptype == "stars":
            self.sigma_los = None
            self.ML = None     
        
        if self.ptype == "darkmatter":
            self.rvir = None
            self.rs = None
            self.c = None
            print("ok")
        
        missing_fields = [f for f in ['coords', 'vels', 'masses', 'IDs'] if f not in config.fields[pt]]
        if missing_fields:
            raise ValueError(f"Missing mandatory fields {missing_fields} for particle type {pt}")
        
        
        
        
        
        
        
        
        del self.loader
        del self.parser
        del self.ptypes
        del self.fields
        
        
        
        
    ##########################################################
    ###                                                    ###
    ##                       UTILITIES                      ##
    ###                                                    ###
    ##########################################################
        
    def __getattr__(self, field_name):
        """Dynamical loader for accessing fields.
        
        Parameters
        ----------
        field_name : str
            Name of the field to be accessed

        Returns
        -------
        field : unyt_array
        """
        funits = {
            'coords': self.units['length'],
            'hsml': self.units['length'],
            'masses': self.units['mass'],
            'masses_ini': self.units['mass'],
            'vels': self.units['velocity'],
            'formation_times': self.units['time']
        }
        
        assert field_name in self._fields.keys(), AttributeError(f"Field {field_name} not found for particle type {self.ptype}. Available fields are: {list(self._fields.keys())}")

        if field_name in self._fields_loaded:
            return self._fields_loaded[field_name].in_units(funits[field_name]) if field_name in funits else self._fields_loaded[field_name]

        
        field = (self._base_ptype, self._fields[field_name])
        self._fields_loaded[field_name] = self._data[field].in_units(funits[field_name]) if field_name in funits else self._data[field]
        return self._fields_loaded[field_name]
        
    

    def get_fields(self):
        """Returns all loadable fields
        """
        return self._fields.keys()
    
    def info(self, get_str = False):
        """Returns a pretty information summary.
        
        Parameters
        ----------
        get_str : bool
            Return str instead of print. Default: False

        Returns
        -------
        info : str, optionally

        """
        output = []
        
        output.append(f"\n{self.ptype}")
        output.append(f"{'':-<21}")
        try:
            output.append(f"{'len_pos':<20}: {len(self.coords)}")
            output.append(f"{'pos[0]':<20}: [{self.coords[0,0].value:.2f}, {self.coords[0,1].value:.2f}, {self.coords[0,2].value:.2f}] {self.units['length']}")
            output.append(f"{'pos[-1]':<20}: [{self.coords[-1,0].value:.2f}, {self.coords[-1,1].value:.2f}, {self.coords[-1,2].value:.2f}] {self.units['length']}")
            
            output.append(f"{'len_vel':<20}: {len(self.vels)}")
            output.append(f"{'vel[0]':<20}: [{self.vels[0,0].value:.2f}, {self.vels[0,1].value:.2f}, {self.vels[0,2].value:.2f}] {self.units['velocity']}")
            output.append(f"{'vel[-1]':<20}: [{self.vels[-1,0].value:.2f}, {self.vels[-1,1].value:.2f}, {self.vels[-1,2].value:.2f}] {self.units['velocity']}")
    
            
            output.append(f"{'len_mass':<20}: {len(self.masses)}")
            output.append(f"{'mass[0]':<20}: {self.masses[0]}")
            output.append(f"{'mass[-1]':<20}: {self.masses[-1]}")
            
            output.append(f"{'len_ids':<20}: {len(self.IDs)}")
            output.append(f"{'ID[0]':<20}: {self.IDs[0].value}")
            output.append(f"{'ID[-1]':<20}: {self.IDs[-1].value}")
            
        except:
            output.append(f"{'len_pos':<20}: {len(self.coords)}")
            output.append(f"{'len_vel':<20}: {len(self.coords)}")
            output.append(f"{'len_mass':<20}: {len(self.coords)}")
            output.append(f"{'len_ids':<20}: {len(self.coords)}")

        
        output.append(f"{'CoM':<20}: {self.cm}")
        output.append(f"{'V_CoM':<20}: {self.vcm}")
        
        output.append(f"{'rh':<20}: {self.rh}")
        output.append(f"{'rh_3D':<20}: {self.rh_3D}")
            
        if self.ptype == "stars":

            output.append(f"{'sigma*':<20}: {self.sigma_los}")
            output.append(f"{'ML':<20}: {self.ML}")
        
        if self.ptype == "darkmatter":
            output.append(f"{'rvir':<20}: {self.rvir}")
            output.append(f"{'rs':<20}: {self.rs}")
            output.append(f"{'c':<20}: {self.c}")


        
        if get_str:
            return "\n".join(output)
        else:
            print("\n".join(output))
            return None

    ##########################################################
    ###                                                    ###
    ##                   PRIVATE FUNCTIONS                  ##
    ###                                                    ###
    ##########################################################

    def _default_center_of_mass(self):
        """Computes coarse CoM using all the particles as 

                CoM = sum(mass * pos) / sum(mass)        
        """
        self.cm = np.median(self.coords, axis=0)
        return None


    ##########################################################
    ###                                                    ###
    ##                 USER ACCESS FUNCTIONS                ##
    ###                                                    ###
    ##########################################################
    
    def set_line_of_sight(self, los):
        """Sets the line of sight to the provided value. The default value for the line of sight is the x-axis: [1,0,0].
        By setting a new line of sight the coordinate basis in which vectorial quantities are expressed changes, aligning the
        old x-axis with the new provided line of sight. This way, a projected view of a set of particles when viewed 
        through los can be obtained by retrieving the new y,z-axes.
        
        The new coordinate system is obtained applying Gram-Schmidt to a preliminary non-orthogonal basis formed by los + identitiy.
        
        Parameters
        ----------
        los : array
            Unitless, non-normalized line of sight vector
            
        Returns
        -------
        None
        
        Changes basis and los instances in the class instance. All vectorial quantities get expressed in the new
        coordinate basis.
        """
        self._set_los(los)
        return None
    
    def refined_center_of_mass(self, 
                                scaling=0.5,
                                method="simple",
                                delta=1E-1,
                                alpha=0.95,
                                m=2,
                                nmin=40,
                                mfrac=0.5
                                ):
        
        """Refined CM position estimation. 

        The center of mass of a particle distribution is not well estimated by the full particle ensemble, since the outermost particles
        must not be distributed symmetrically around the TRUE center of mass. For that reason, a closer-to-truth value for th CM can be
        obtained by disregarding these outermost particles.
        
        Here we implement four methods in which a more refined CM estimation can be obtained. All of them avoid using gravitational 
        porentials, as these are not available to observers. Only positions and masses (analogous to luminosities in some sense)
        are used:
            
            1. SIMPLE: Discard all particles outside of rshell = scaling * rmax. 
            2. ITERATIVE: An iterative version of the SIMPLE method, where rshell decreases in steps, with rshell_i+1 = alpha*rshell_i.
                       Desired absolute accuracy can be set, but not guaranteed. This method suffers when there are few particles.
            3. HALF-MASS: A variant of the SIMPLE method. Here the cut is done in mass instead of radius. The user can specify a
                       X-mass-fraction and the X-mass radius and its center are computed iterativelly unitl convergence. Convergence is
                       guaranteed, but the X-mass-center might or might not be close to the true center. The user can play around with
                       mfrac to achieve the best result.
            4. ITERATIVE HALF-MASS: Implements the same iterative process as in 2., but at each step the radius is computed using the
                       same procedure as in method 3. The mass fraction is reduced by alpha at each step. As in method 2., convergence
                       is not guaranteed.
                        

        OPTIONAL Parameters
        -------------------
        
        scaling : float
            rshell/rmax. Must be between 0 and 1. Default: 0.5
        method : str, optional
            Method with which to refine the CoM: simple, interative, hm or iterative-hm. Default: simple
        delta : float
            Absolute tolerance for stopping re-refinement. Default 1E-1.
        alpha : float
            rhsell_i+1/rshell_i. Default: 0.95
        m : int
            Consecutive iterations under delta to converge. Default: 2
        nmin : int
            Minimum  number of particles for reliable CoM estimation. Default: 40
        mfrac : float
            Mass fraction. Default: 0.5
            

        Returns
        -------
        cm : array
            Refined Center of mass and various quantities.
        """
        self.centering_results = refine_center(self.coords, self.masses,
                                               scaling,
                                               method,
                                               delta,
                                               alpha,
                                               m,
                                               nmin,
                                               mfrac
                                               )
        self.cm = self.centering_results['center']
        return self.cm
    
    def half_mass_radius(self, mfrac=0.5):
        """By default, it computes 3D half mass radius of a given particle ensemble. If the center of the particles 
        is not provided, it is estimated by first finding the median of positions and then computing a refined CoM using
        only the particles inside r < 0.5*rmax.
        
        There is also an argument to compute other ALPHA_mass_radius of an arbitrary mass fraction. The desired ALPHA_mass_radius  
        is computed via rootfinding using scipy's implementation of brentq method.
        
        OPTIONAL Parameters
        -------------------
        mfrac : float, optional
            Mass fraction of desired radius. Default: 0.5 (half, mass radius).
        
        Returns
        -------
        MFRAC_mass_radius : float
            Desired mfrac mass fraction radius estimation. Provided in same units as pos, if any.
        """
        self.rh_3D = half_mass_radius(self.coords, self.masses, self.cm, mfrac)
        return self.rh_3D
    
    














































































class gasSPH(BaseSimulationObject):
    ##########################################################
    ###                                                    ###
    ##                    INNIT FUNCTION                    ##
    ###                                                    ###
    ##########################################################
    
    def __init__(self,
                 data,
                 pt
                 ):
        """Initializes the ptype class.
        """









class gasMESH(BaseSimulationObject):
    ##########################################################
    ###                                                    ###
    ##                    INNIT FUNCTION                    ##
    ###                                                    ###
    ##########################################################
    
    def __init__(self,
                 data,
                 pt
                 ):
        """Initializes the ptype class.
        """






