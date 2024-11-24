#import yt
#import os
#import time
import numpy as np
#from tqdm import tqdm
#from pytreegrav import Potential, PotentialTarget
from unyt import unyt_array, unyt_quantity
#from unyt.physical_constants import G
from copy import copy

from .config import config
from .base import BaseSimulationObject, BaseComponent

from .class_methods import center_of_mass, refine_center, half_mass_radius, compute_stars_in_halo, bound_particlesBH, bound_particlesAPROX






class StellarComponent(BaseSimulationObject, BaseComponent):
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
                 **kwargs
                 ):
        """Initializes the ptype class.
        """
        super().__init__()
        self.ptype, self._base_ptype = "stars", copy(self.ptypes["stars"])
        self._dynamic_fields = copy(self.fields["stars"])
        self._fields_loaded = {}
        self._data = data
        self._kwargs = kwargs
        self.use_bound_if_computed = True
        self.bound_method = None
        
        self.cm = None
        self.vcm = None
        self.bmask = None
        
        self.rh = None
        self.rh_3D = None
        self.sigma_los = None
        
        self.set_shared_attrs(self.ptype, self._kwargs)

        missing_fields = [f for f in ['coords', 'vels', 'masses', 'IDs'] if f not in config.fields["stars"]]
        if missing_fields:
            raise ValueError(f"Missing mandatory fields {missing_fields} for particle type stars")
        
        self._default_center_of_mass()
        
        del self.loader
        del self.parser
        del self.ptypes
        del self.fields
        
        
    ##########################################################
    ###                                                    ###
    ##                      PROPERTIES                      ##
    ###                                                    ###
    ##########################################################   
    
    @property
    def ML(self):
        if self.ptype == "stars":
            value = self.get_shared_attr(self.ptype, "ML")
            return value if value is None else value.in_units("Msun/Lsun")
        raise AttributeError("Attribute 'ML' is hidden for dark matter.")

    @ML.setter
    def ML(self, value):
        if self.ptype == "stars":
            self.update_shared_attr(self.ptype, "ML", value)
        else:
            raise AttributeError("Cannot set 'ML' for dark matter.")

    # "Hidden" properties for the partner instance
    @property
    def _rvir(self):
        if self.ptype == "stars":
            value = self.get_shared_attr("darkmatter", "rvir")
            return value if value is None else value.in_units(self.units['length'])
        raise AttributeError("Attribute '_rvir' is not accessible for dark matter.")

    @property
    def _rs(self):
        if self.ptype == "stars":
            value = self.get_shared_attr("darkmatter", "rs")
            return value if value is None else value.in_units(self.units['length'])
        raise AttributeError("Attribute '_rs' is not accessible for dark matter.")

    @property
    def _c(self):
        if self.ptype == "stars":
            value = self.get_shared_attr("darkmatter", "c")
            return value if value is None else value.in_units(self.units['dimensionless'])
        raise AttributeError("Attribute '_c' is not accessible for dark matter.")

    @property
    def _rockstar_center(self):
        if self.ptype == "stars":
            value = self.get_shared_attr("darkmatter", "rockstar_center")
            return value if value is None else value.in_units(self.units['length'])
        raise AttributeError("Attribute '_rockstar_center' is not accessible for dark matter.")

    @property
    def _rockstar_vel(self):
        if self.ptype == "stars":
            value = self.get_shared_attr("darkmatter", "rockstar_vel")
            return value if value is None else value.in_units(self.units['velocity'])
        raise AttributeError("Attribute '_rockstar_vel' is not accessible for dark matter.")

    @property
    def _vmax(self):
        if self.ptype == "stars":
            value = self.get_shared_attr("darkmatter", "vmax")
            return value if value is None else value.in_units(self.units['velocity'])
        raise AttributeError("Attribute 'vmax' is hidden for stars.")

    @property
    def _vrms(self):
        if self.ptype == "stars":
            value = self.get_shared_attr("darkmatter", "vrms")
            return value if value is None else value.in_units(self.units['velocity'])
        raise AttributeError("Attribute 'vrms' is hidden for stars.")


            
    ##########################################################
    ###                                                    ###
    ##                      UTILITIES                       ##
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
            'formation_times': self.units['time'],
            'metallicity': self.units['dimensionless']
        }
        if field_name  in self._dynamic_fields.keys():
            if field_name in self._fields_loaded:
                return self._fields_loaded[field_name].in_units(funits[field_name]) if field_name in funits else self._fields_loaded[field_name]
            else:
                field = (self._base_ptype, self._dynamic_fields[field_name])
                self._fields_loaded[field_name] = self._data[field].in_units(funits[field_name]) if field_name in funits else self._data[field]
                return self._fields_loaded[field_name]
                
        elif field_name  in ['b'+ f for f in list(self._dynamic_fields.keys())]:
            #if field_name in self._fields_loaded:
            #    return self._fields_loaded[field_name].in_units(funits[field_name[1:]]) if field_name[1:] in funits else self._fields_loaded[field_name]
            #else:
            field = (self._base_ptype, self._dynamic_fields[field_name[1:]])
            self._fields_loaded[field_name] = self._data[field][self.bmask].in_units(funits[field_name[1:]]) if field_name[1:] in funits else self._data[field][self.bmask]
            return self._fields_loaded[field_name]

        try:
            print(f"with __getattribute__")
            return self.__getattribute__(field_name)
        except AttributeError:
            raise AttributeError(f"Field '{field_name}' not found for particle type {self.ptype}. "+ f"Available fields are: {list(self._dynamic_fields.keys())}")

        AttributeError(f"Field {field_name} not found for particle type {self.ptype}. Available fields are: {list(self._dynamic_fields.keys())}")
        return None
        
    def get_particle_fields(self):
        """Returns all loadable particle fields
        """
        if self.bmask is None:
            return self._dynamic_fields.keys()
        else:
            return np.append(list(self._dynamic_fields.keys()), ['b'+f for f in list(self._dynamic_fields.keys())])    
            
    ##########################################################
    ###                                                    ###
    ##                   SPECIFIC METHODS                   ##
    ###                                                    ###
    ##########################################################

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
        
        output.append(f"{'rh, rh_3D':<20}: {self.rh}, {self.rh_3D}")
            
        output.append(f"{'sigma*':<20}: {self.sigma_los}")
        output.append(f"{'ML':<20}: {self.ML}")
        

        if get_str:
            return "\n".join(output)
        else:
            print("\n".join(output))
            return None
    
    def compute_stars_in_halo(self, verbose=False):
        """Computes the stars that form a galaxy inside a given halo using the recipe of Jenna Samuel et al. (2020). 
        For this one needs a catalogue of halos (e.g. Rockstar). The steps are the following:
    
            1. All the stars inside the min(0.8*Rvir, 30) kpccm of the host halo are considered as 
               candidates to form the given galaxy.
            2. The stars with relative speed bigger than 2*V_{circ, max} (usually a quantity computed in the
               catalogues) are removed.
            3. An iterative process is started where:
                  (i) All the particles outside of 1.5*R90 (radius where m(R)/M_T = 0.9) are removed. We take into 
                      account both the stars cm and the halo center.
                 (ii) We remove all the particles that have velocities 2*sigma above the mean.
                (iii) A convergence criterion of deltaM*/M*<0.01 is stablished.
    
            4. We only keep the galaxy if it has more than six stellar particles.
    
        Parameters
        ----------
        pos, masses, vels : array-like[float] with units
            Absolute position, mass and velocity of particles.
        pindices : array[int]
            Indices of particles.
        halo_params : dict[str : unyt_quantity or unyt_array]
            Parameters of the halo in which we are searching: center, center_vel, Rvir, vmax and vrms.
        max_radius : float, optional
            Maximum radius to consider for particle unbinding. Default: 30 kpc
        imax : int, optional
            Maximum number of iterations. Default 200.
        verbose : bool
            Wether to verbose or not. Default: False.
            
        Returns
        -------
        indices : array
            Array of star particle indices belonging to the halo.
        mask : boolean-array
            Boolean array for masking quantities
        delta_rel : float
            Obtained convergence for selected total mass after imax iterations. >1E-2.
        """ 
        _, mask, delta_rel = compute_stars_in_halo(self.coords,
                                                          self.masses,
                                                          self.vels,
                                                          self.IDs,
                                                          {'center': self._rockstar_center ,
                                                           'center_vel': self._rockstar_vel,
                                                           'rvir': self._rvir,
                                                           'vmax': self._vmax,
                                                           'vrms': self._vrms                                                             
                                                           },
                                                           verbose=verbose
                                                          )
        self.bmask = mask
        self.delta_rel = delta_rel.value
        self.bound_method = "starry-halo"
        for key in list(self._fields_loaded.keys()):  
            if key.startswith("b"):
                del self._fields_loaded[key]
        return None        
    













class DarkComponent(BaseSimulationObject, BaseComponent):
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
                 **kwargs
                 ):
        """Initializes the ptype class.
        """
        super().__init__()
        self.ptype, self._base_ptype = "darkmatter", copy(self.ptypes["darkmatter"])
        self._dynamic_fields = copy(self.fields["darkmatter"])
        self._fields_loaded = {}
        self._data = data
        self._kwargs = kwargs
        self.use_bound_if_computed = True
        self.bound_method = None
        self.cm = None
        self.vcm = None
        self.rh = None
        self.rh_3D = None


        self.set_shared_attrs(self.ptype, self._kwargs)
                
        missing_fields = [f for f in ['coords', 'vels', 'masses', 'IDs'] if f not in config.fields["darkmatter"]]
        if missing_fields:
            raise ValueError(f"Missing mandatory fields {missing_fields} for particle type darkmatter")

        
        self._default_center_of_mass()
        
        
        
        del self.loader
        del self.parser
        del self.ptypes
        del self.fields
        
        
    ##########################################################
    ###                                                    ###
    ##                       UTILITIES                      ##
    ###                                                    ###
    ##########################################################   
    
    @property
    def rvir(self):
        if self.ptype == "darkmatter":
            value = self.get_shared_attr(self.ptype, "rvir")
            return value if value is None else value.in_units(self.units['length'])
        raise AttributeError("Attribute 'rvir' is hidden for stars.")

    @rvir.setter
    def rvir(self, value):
        if self.ptype == "darkmatter":
            self.update_shared_attr(self.ptype, "rvir", value)
        else:
            raise AttributeError("Cannot set 'rvir' for stars.")

    @property
    def rs(self):
        if self.ptype == "darkmatter":
            value = self.get_shared_attr(self.ptype, "rs")
            return value if value is None else value.in_units(self.units['length'])
        raise AttributeError("Attribute 'rs' is hidden for stars.")

    @rs.setter
    def rs(self, value):
        if self.ptype == "darkmatter":
            self.update_shared_attr(self.ptype, "rs", value)
        else:
            raise AttributeError("Cannot set 'rs' for stars.")

    @property
    def c(self):
        if self.ptype == "darkmatter":
            value = self.get_shared_attr(self.ptype, "c")
            return value if value is None else value.in_units(self.units['dimensionless'])
        raise AttributeError("Attribute 'c' is hidden for stars.")

    @c.setter
    def c(self, value):
        if self.ptype == "darkmatter":
            self.update_shared_attr(self.ptype, "c", value)
        else:
            raise AttributeError("Cannot set 'c' for stars.")

    @property
    def rockstar_center(self):
        if self.ptype == "darkmatter":
            value = self.get_shared_attr(self.ptype, "rockstar_center")
            return value if value is None else value.in_units(self.units['length'])
        raise AttributeError("Attribute 'rockstar_center' is hidden for stars.")

    @rockstar_center.setter
    def rockstar_center(self, value):
        if self.ptype == "darkmatter":
            self.update_shared_attr(self.ptype, "rockstar_center", value)
        else:
            raise AttributeError("Cannot set 'rockstar_center' for stars.")

    @property
    def rockstar_vel(self):
        if self.ptype == "darkmatter":
            value = self.get_shared_attr(self.ptype, "rockstar_vel")
            return value if value is None else value.in_units(self.units['velocity'])
        raise AttributeError("Attribute 'rockstar_vel' is hidden for stars.")

    @rockstar_vel.setter
    def rockstar_vel(self, value):
        if self.ptype == "darkmatter":
            self.update_shared_attr(self.ptype, "rockstar_vel", value)
        else:
            raise AttributeError("Cannot set 'rockstar_vel' for stars.")

    @property
    def vmax(self):
        if self.ptype == "darkmatter":
            value = self.get_shared_attr(self.ptype, "vmax")
            return value if value is None else value.in_units(self.units['velocity'])
        raise AttributeError("Attribute 'vmax' is hidden for stars.")

    @rockstar_vel.setter
    def vmax(self, value):
        if self.ptype == "darkmatter":
            self.update_shared_attr(self.ptype, "vmax", value)
        else:
            raise AttributeError("Cannot set 'vmax' for stars.")

    @property
    def vrms(self):
        if self.ptype == "darkmatter":
            value = self.get_shared_attr(self.ptype, "vrms")
            return value if value is None else value.in_units(self.units['velocity'])
        raise AttributeError("Attribute 'vrms' is hidden for stars.")

    @rockstar_vel.setter
    def vrms(self, value):
        if self.ptype == "darkmatter":
            self.update_shared_attr(self.ptype, "vrms", value)
        else:
            raise AttributeError("Cannot set 'vrms' for stars.")
        
    @property
    def _ML(self):
        if self.ptype == "darkmatter":
            value = self.get_shared_attr("stars", "ML")
            return value if value is None else value.in_units("Msun/Lsun")
        raise AttributeError("Attribute '_ML' is not accessible for stars.")



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
            'formation_times': self.units['time'],
            'metallicity': self.units['dimensionless']
        }
        if field_name  in self._dynamic_fields.keys():
            if field_name in self._fields_loaded:
                return self._fields_loaded[field_name].in_units(funits[field_name]) if field_name in funits else self._fields_loaded[field_name]
            else:
                field = (self._base_ptype, self._dynamic_fields[field_name])
                self._fields_loaded[field_name] = self._data[field].in_units(funits[field_name]) if field_name in funits else self._data[field]
                return self._fields_loaded[field_name]
                
        elif field_name  in ['b'+ f for f in list(self._dynamic_fields.keys())]:
            if field_name in self._fields_loaded:
                return self._fields_loaded[field_name].in_units(funits[field_name[1:]]) if field_name[1:] in funits else self._fields_loaded[field_name]
            else:
                field = (self._base_ptype, self._dynamic_fields[field_name[1:]])
                self._fields_loaded[field_name] = self._data[field][self.bmask].in_units(funits[field_name[1:]]) if field_name[1:] in funits else self._data[field][self.bmask]
                return self._fields_loaded[field_name]

        try:
            print(f"with __getattribute__")
            return self.__getattribute__(field_name)
        except AttributeError:
            raise AttributeError(f"Field '{field_name}' not found for particle type {self.ptype}. "+ f"Available fields are: {list(self._dynamic_fields.keys())}")

        AttributeError(f"Field {field_name} not found for particle type {self.ptype}. Available fields are: {list(self._dynamic_fields.keys())}")
        return None
    

    def get_particle_fields(self):
        """Returns all loadable particle fields
        """
        if self.bmask is None:
            return self._dynamic_fields.keys()
        else:
            return np.append(list(self._dynamic_fields.keys()), ['b'+f for f in list(self._dynamic_fields.keys())])    
            
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
        output.append(f"{'rh, rh_3D':<20}: {self.rh}, {self.rh_3D}")
        output.append(f"{'rvir, rs, c':<20}: {self.rvir}, {self.rs}, {self.c}")

        if get_str:
            return "\n".join(output)
        else:
            print("\n".join(output))
            return None

    def compute_bound_particles(self, method, refine=True, delta=1E-5, verbose = False):
        """Computes the bound particles of a halo/ensemble of particles using the Barnes-Hut algorithm implemented in PYTREEGRAV or 
        approximating it as:

                            pot = -G* Mtot * m_i / |x_i - x_cm|
                            
        The bound particles are defined as those with E= pot + kin < 0. The center of mass position is not required, as the potential 
        calculation is done with relative distances between particles. To compute the kinetic energy of the particles the *v_cm*
        is needed: you can provide one or the function make a crude estimates of it with all particles. 
    
        You can ask the function to refine the unbinding by iterating until the *cm* and *v_cm* vectors converge to within a user specified
        delta (default is 1E-5, relative). This is recomended if you dont specify the *v_cm*, as the initial estimation using all particles
        may be biased for anisotropic particle distributions. If you provide your own *v_cm*, taken from somewhere reliable e.g. Rockstar halo
        catalogues, it will be good enough. Note that if you still want to refine, the *v_cm* you provided will be disregarded after the
        first iteration.
    
        Typical run times are of 0.5-2s dependin if the result is refined or not. This is similar to the I/O overheads of yt.load, making it reasonable for
        use with a few halos (where you dont care if the total runtime is twice as long because it will be fast enough).
    
        Unit handling is done with unyt. 
        """
        
        if method == "APROX":
            self.E, self.kin, self.pot = bound_particlesAPROX(self.coords,
                                                             self.vels,
                                                             self.masses,
                                                             cm = self.rockstar_center,
                                                             vcm = self.rockstar_vel,
                                                             refine = refine,
                                                             delta = delta,
                                                             verbose = verbose
                                                            )


        
        elif method == "BH":
            try:
                hsml = halo.hsml
            except:
                hsml = None
    
            self.E, self.kin, self.pot = bound_particlesBH(self.coords,
                                                           self.vels,
                                                           self.masses,
                                                           soft = hsml,
                                                           cm = self.cm,
                                                           vcm = self.vcm,
                                                           refine = refine,
                                                           delta = delta,
                                                           verbose = verbose                                  
                                                          )
        self.bmask = self.E < 0
        self.bound_method = f"grav-{method}".lower()
        for key in list(self._fields_loaded.keys()):  
            if key.startswith("b"):
                del self._fields_loaded[key]
                
        return None





















class GasComponentSPH(BaseSimulationObject, BaseComponent):
    ##########################################################
    ###                                                    ###
    ##                    INNIT FUNCTION                    ##
    ###                                                    ###
    ##########################################################
    
    def __init__(self,
                 data,
                 ):
        """Initializes the ptype class.
        """









class GasComponentMESH(BaseSimulationObject, BaseComponent):
    ##########################################################
    ###                                                    ###
    ##                    INNIT FUNCTION                    ##
    ###                                                    ###
    ##########################################################
    
    def __init__(self,
                 data,
                 ):
        """Initializes the ptype class.
        """






