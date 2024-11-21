import yt
import os
import time
import numpy as np
from tqdm import tqdm
from pytreegrav import Potential, PotentialTarget
from unyt import unyt_array, unyt_quantity
from unyt.physical_constants import G
from copy import copy

from .config import config
from .base import BaseSimulationObject

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
        
        
        if [f for f in ['coords', 'vels', 'masses', 'IDs'] if f not in config.fields[pt]]:
            raise ValueError(f"Missing mandatory fields {missing_fields} for particle type {pt}")
        
        
        
        
        
        
        
        
        del self.loader
        del self.parser
        del self.ptypes
        del self.fields
        

        
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
        assert field_name in self._fields.keys(), AttributeError(f"Field {field_name} not found for particle type {self.ptype}. Available fields are: {list(self._fields.keys())}")

        if field_name in self._fields_loaded:
            return self._fields_loaded[field_name]
        
        field = (self._base_ptype, self._fields[field_name])
        print(field)
        self._fields_loaded[field_name] = self._data[field]
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
            output.append(f"{'pos[0]':<20}: [{self.coords[0,0]:.2f}, {self.coords[0,1]:.2f}, {self.coords[0,2]:.2f}] {self.units['length']}")
            output.append(f"{'pos[-1]':<20}: [{self.coords[-1,0]:.2f}, {self.coords[-1,1]:.2f}, {self.coords[-1,2]:.2f}] {self.units['length']}")
            
            output.append(f"{'len_vel':<20}: {len(self.vels)}")
            output.append(f"{'pos[0]':<20}: [{self.vels[0,0]:.2f}, {self.vels[0,1]:.2f}, {self.vels[0,2]:.2f}] {self.units['length']}")
            output.append(f"{'pos[-1]':<20}: [{self.vels[-1,0]:.2f}, {self.vels[-1,1]:.2f}, {self.vels[-1,2]:.2f}] {self.units['length']}")
    
            
            output.append(f"{'len_mass':<20}: {len(self.masses)}")
            output.append(f"{'mass[0]':<20}: {self.masses[0]}")
            output.append(f"{'mass[-1]':<20}: {self.masses[-1]}")
            
            output.append(f"{'len_ids':<20}: {len(self.IDs)}")
            output.append(f"{'ID[0]':<20}: {self.IDs[0]}")
            output.append(f"{'ID[-1]':<20}: {self.IDs[-1]}")
            
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






