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
from .class_methods import (
                            compute_stars_in_halo, 
                            bound_particlesBH, 
                            bound_particlesAPROX
                            )






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
    """ 
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
        
        self._bmask = np.ones_like(self.masses.value, dtype=bool)

        missing_fields = [f for f in ['coords', 'vels', 'masses', 'IDs'] if f not in config.fields["stars"]]
        if missing_fields:
            raise ValueError(f"Missing mandatory fields {missing_fields} for particle type stars")

        self.clean_shared_attrs(self.ptype)
        self.set_shared_attrs(self.ptype, kwargs)
        self._default_center_of_mass()

        
        del self.loader
        del self.parser
        del self.ptypes
        del self.fields
        



    @property
    def ML(self):
        value = self.get_shared_attr(self.ptype, "ML")
        return value if value is None else value.in_units("Msun/Lsun")
    @ML.setter
    def ML(self, value):
        self.update_shared_attr(self.ptype, "ML", value)
    @property
    def cm(self):
        value = self.get_shared_attr(self.ptype, "cm")
        return value if value is None else value.in_units(self.units['length'])
    @cm.setter
    def cm(self, value):
        self.update_shared_attr(self.ptype, "cm", value)
    @property
    def vcm(self):
        value = self.get_shared_attr(self.ptype, "vcm")
        return value if value is None else value.in_units(self.units['velocity'])
    @vcm.setter
    def vcm(self, value):
        self.update_shared_attr(self.ptype, "vcm", value)
    @property
    def rh(self):
        value = self.get_shared_attr(self.ptype, "rh")
        return value if value is None else value.in_units(self.units['length'])
    @rh.setter
    def rh(self, value):
        self.update_shared_attr(self.ptype, "rh", value)
    @property
    def rh3d(self):
        value = self.get_shared_attr(self.ptype, "rh3d")
        return value if value is None else value.in_units(self.units['length'])
    @rh3d.setter
    def rh3d(self, value):
        self.update_shared_attr(self.ptype, "rh3d", value)
    @property
    def sigma_los(self):
        value = self.get_shared_attr(self.ptype, "sigma_los")
        return value if value is None else value.in_units(self.units['velocity'])
    @sigma_los.setter
    def sigma_los(self, value):
        self.update_shared_attr(self.ptype, "sigma_los", value)

    

    def __getattr__(self, field_name):
        """Dynamical loader for accessing fields.
        """
        return self._priv__getattr__(field_name)

   



    def get_particle_fields(self):
        """Returns all loadable particle fields
        """
        if self._bmask is None:
            return self._dynamic_fields.keys()
        else:
            return np.append(list(self._dynamic_fields.keys()), ['b'+f for f in list(self._dynamic_fields.keys())])

    def info(self, 
             get_str=False
            ):
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
            output.append(f"{'len_vel':<20}: {len(self.vels)}")
            output.append(f"{'vel[0]':<20}: [{self.vels[0,0].value:.2f}, {self.vels[0,1].value:.2f}, {self.vels[0,2].value:.2f}] {self.units['velocity']}")
            output.append(f"{'len_mass':<20}: {len(self.masses)}")
            output.append(f"{'mass[0]':<20}: {self.masses[0]}")
            output.append(f"{'len_ids':<20}: {len(self.IDs)}")
            output.append(f"{'ID[0]':<20}: {self.IDs[0].value}")
            
        except:
            output.append(f"{'len_pos':<20}: {len(self.coords)}")
            output.append(f"{'len_vel':<20}: {len(self.coords)}")
            output.append(f"{'len_mass':<20}: {len(self.coords)}")
            output.append(f"{'len_ids':<20}: {len(self.coords)}")

        output.append(f"{'cm':<20}: {self.cm}")
        output.append(f"{'vcm':<20}: {self.vcm}")
        
        output.append(f"{'rh, rh3D':<20}: {self.rh}, {self.rh3d}")
            
        output.append(f"{'sigma_los':<20}: {self.sigma_los}")
        output.append(f"{'ML':<20}: {self.ML}")
        

        if get_str:
            return "\n".join(output)
        else:
            print("\n".join(output))
            return None
    
    def compute_stars_in_halo(self, 
                              verbose=False,
                              **kwargs
                             ):
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
    
        OPTIONAL Parameters
        ----------
        verbose : bool
            Wether to verbose or not. Default: False.        

        KEYWORD Parameters
        ----------
        center, center_vel : unyt_array
            Dark Matter halo center and center velocity.
        rvir : unyt_quantity
            virial radius
        vmax : unyt_quantity
            Maximum circular velocity of the halo
        vrms : unyt_quantity
            Disperion velocity of the halo
        max_radius : tuple[float, str] 
            Max radius to consider stellar particles. Default: 30 'kpc'
        imax : int
            Maximum number of iterations. Default: 200
            
        
            
        Returns
        -------
        indices : array
            Array of star particle indices belonging to the halo.
        mask : boolean-array
            Boolean array for masking quantities
        delta_rel : float
            Obtained convergence for selected total mass after imax iterations. >1E-2.
        """ 
        halo_params = {
            'center': self._shared_attrs["darkmatter"]["rockstar_center"],
            'center_vel': self._shared_attrs["darkmatter"]["rockstar_vel"],
            'rvir': self._shared_attrs["darkmatter"]["rvir"],
            'vmax': self._shared_attrs["darkmatter"]["vmax"],
            'vrms': self._shared_attrs["darkmatter"]["vrms"]                                                             
        }
        
        for key in halo_params.keys():
            halo_params[key] = halo_params[key] if key not in kwargs.keys() else kwargs[key]
        
        print(halo_params)
        
        _, mask, delta_rel = compute_stars_in_halo(
            self.coords,
            self.masses,
            self.vels,
            self.IDs,
            halo_params=halo_params,
            max_radius=(30, 'kpc') if "max_radius" not in kwargs.keys() else kwargs["max_radius"],
            imax=200 if "imax" not in kwargs.keys() else int(kwargs["imax"]),
            verbose=verbose
        )
        
        self._bmask = mask
        self.delta_rel = delta_rel
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
    """
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
        
        self._bmask = np.ones_like(self.masses.value, dtype=bool)
                
        missing_fields = [f for f in ['coords', 'vels', 'masses', 'IDs'] if f not in config.fields["darkmatter"]]
        if missing_fields:
            raise ValueError(f"Missing mandatory fields {missing_fields} for particle type darkmatter")

        self.clean_shared_attrs(self.ptype)
        self.set_shared_attrs(self.ptype, kwargs)
        self._default_center_of_mass()
        
        del self.loader
        del self.parser
        del self.ptypes
        del self.fields
          




    @property
    def cm(self):
        value = self.get_shared_attr(self.ptype, "cm")
        return value if value is None else value.in_units(self.units['length'])
    @cm.setter
    def cm(self, value):
        self.update_shared_attr(self.ptype, "cm", value)
    @property
    def vcm(self):
        value = self.get_shared_attr(self.ptype, "vcm")
        return value if value is None else value.in_units(self.units['velocity'])
    @vcm.setter
    def vcm(self, value):
        self.update_shared_attr(self.ptype, "vcm", value)
    @property
    def rh(self):
        value = self.get_shared_attr(self.ptype, "rh")
        return value if value is None else value.in_units(self.units['length'])
    @rh.setter
    def rh(self, value):
        self.update_shared_attr(self.ptype, "rh", value)
    @property
    def rh3d(self):
        value = self.get_shared_attr(self.ptype, "rh3d")
        return value if value is None else value.in_units(self.units['length'])
    @rh3d.setter
    def rh3d(self, value):
        self.update_shared_attr(self.ptype, "rh3d", value)
        
    @property
    def rvir(self):
        value = self.get_shared_attr(self.ptype, "rvir")
        return value if value is None else value.in_units(self.units['length'])
    @rvir.setter
    def rvir(self, value):
        self.update_shared_attr(self.ptype, "rvir", value)
    @property
    def rs(self):
        value = self.get_shared_attr(self.ptype, "rs")
        return value if value is None else value.in_units(self.units['length'])
    @rs.setter
    def rs(self, value):
        self.update_shared_attr(self.ptype, "rs", value)
    @property
    def c(self):
        value = self.get_shared_attr(self.ptype, "c")
        return value if value is None else value.in_units(self.units['dimensionless'])
    @c.setter
    def c(self, value):
        self.update_shared_attr(self.ptype, "c", value)
    @property
    def rockstar_center(self):
        value = self.get_shared_attr(self.ptype, "rockstar_center")
        return value if value is None else np.linalg.inv(self.basis) @ value.in_units(self.units['length'])
    @rockstar_center.setter
    def rockstar_center(self, value):
        self.update_shared_attr(self.ptype, "rockstar_center", value)
    @property
    def rockstar_vel(self):
        value = self.get_shared_attr(self.ptype, "rockstar_vel")
        return value if value is None else np.linalg.inv(self.basis) @ value.in_units(self.units['velocity'])
    @rockstar_vel.setter
    def rockstar_vel(self, value):
        self.update_shared_attr(self.ptype, "rockstar_vel", value)
    @property
    def vmax(self):
        value = self.get_shared_attr(self.ptype, "vmax")
        return value if value is None else value.in_units(self.units['velocity'])
    @rockstar_vel.setter
    def vmax(self, value):
        self.update_shared_attr(self.ptype, "vmax", value)
    @property
    def vrms(self):
        value = self.get_shared_attr(self.ptype, "vrms")
        return value if value is None else value.in_units(self.units['velocity'])
    @rockstar_vel.setter
    def vrms(self, value):
        self.update_shared_attr(self.ptype, "vrms", value)




    

    def __getattr__(self, field_name):
        """Dynamical loader for accessing fields.
        """
        return self._priv__getattr__(field_name)





    def get_particle_fields(self):
        """Returns all loadable particle fields
        """
        if self._bmask is None:
            return self._dynamic_fields.keys()
        else:
            return np.append(list(self._dynamic_fields.keys()), ['b'+f for f in list(self._dynamic_fields.keys())]) 
    
    def info(self, 
             get_str=False
            ):
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
            output.append(f"{'len_vel':<20}: {len(self.vels)}")
            output.append(f"{'vel[0]':<20}: [{self.vels[0,0].value:.2f}, {self.vels[0,1].value:.2f}, {self.vels[0,2].value:.2f}] {self.units['velocity']}")
            output.append(f"{'len_mass':<20}: {len(self.masses)}")
            output.append(f"{'mass[0]':<20}: {self.masses[0]}")
            output.append(f"{'len_ids':<20}: {len(self.IDs)}")
            output.append(f"{'ID[0]':<20}: {self.IDs[0].value}")
            
        except:
            output.append(f"{'len_pos':<20}: {len(self.coords)}")
            output.append(f"{'len_vel':<20}: {len(self.coords)}")
            output.append(f"{'len_mass':<20}: {len(self.coords)}")
            output.append(f"{'len_ids':<20}: {len(self.coords)}")

        output.append(f"{'cm':<20}: {self.cm}")
        output.append(f"{'vcm':<20}: {self.vcm}")
        
        output.append(f"{'rh, rh3D':<20}: {self.rh}, {self.rh3d}")
        output.append(f"{'rvir, rs, c':<20}: {self.rvir}, {self.rs}, {self.c}")

        if get_str:
            return "\n".join(output)
        else:
            print("\n".join(output))
            return None

    def compute_bound_particles(self, 
                                method, 
                                refine=True, 
                                delta=1E-5, 
                                verbose=False
                               ):
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
            self.E, self.kin, self.pot = bound_particlesAPROX(
                self.coords,
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
                hsml = self.hsml
            except:
                hsml = None
    
            self.E, self.kin, self.pot = bound_particlesBH(
                self.coords,
                self.vels,
                self.masses,
                soft = hsml,
                cm = self.cm,
                vcm = self.vcm,
                refine = refine,
                delta = delta,
                verbose = verbose                                  
            )
            
        self._bmask = self.E < 0
        self.bound_method = f"grav-{method}".lower()
        
        for key in list(self._fields_loaded.keys()):  
            if key.startswith("b"):
                del self._fields_loaded[key]
                
        return None





















class GasComponent(BaseSimulationObject, BaseComponent):
    """
    """
    def __init__(self,
                 data,
                 **kwargs
                 ):
        """Initializes the ptype class.
        """
        super().__init__()
        self.ptype, self._base_ptype = "gas", copy(self.ptypes["gas"])
        
        self._dynamic_fields = copy(self.fields["gas"])
        self._fields_loaded = {}
        self._data = data
        
        self._bmask = np.ones_like(self.masses.value, dtype=bool)

        missing_fields = [f for f in ['coords', 'vels', 'masses'] if f not in config.fields["gas"]]
        if missing_fields:
            raise ValueError(f"Missing mandatory fields {missing_fields} for particle type stars")

        self.clean_shared_attrs(self.ptype)
        self.set_shared_attrs(self.ptype, kwargs)
        self._default_center_of_mass()

        
        del self.loader
        del self.parser
        del self.ptypes
        del self.fields


    @property
    def cm(self):
        value = self.get_shared_attr(self.ptype, "cm")
        return value if value is None else value.in_units(self.units['length'])
    @cm.setter
    def cm(self, value):
        self.update_shared_attr(self.ptype, "cm", value)
    @property
    def vcm(self):
        value = self.get_shared_attr(self.ptype, "vcm")
        return value if value is None else value.in_units(self.units['velocity'])
    @vcm.setter
    def vcm(self, value):
        self.update_shared_attr(self.ptype, "vcm", value)
    @property
    def rh(self):
        value = self.get_shared_attr(self.ptype, "rh")
        return value if value is None else value.in_units(self.units['length'])
    @rh.setter
    def rh(self, value):
        self.update_shared_attr(self.ptype, "rh", value)
    @property
    def rh3d(self):
        value = self.get_shared_attr(self.ptype, "rh3d")
        return value if value is None else value.in_units(self.units['length'])
    @rh3d.setter
    def rh3d(self, value):
        self.update_shared_attr(self.ptype, "rh3d", value)
    @property
    def sigma_los(self):
        value = self.get_shared_attr(self.ptype, "sigma_los")
        return value if value is None else value.in_units(self.units['velocity'])
    @sigma_los.setter
    def sigma_los(self, value):
        self.update_shared_attr(self.ptype, "sigma_los", value)



    def __getattr__(self, field_name):
            """Dynamical loader for accessing fields.
            """
            return self._priv__getattr__(field_name)



    

    def get_gas_fields(self):
        """Returns all loadable particle fields
        """
        if self._bmask is None:
            return self._dynamic_fields.keys()
        else:
            return np.append(list(self._dynamic_fields.keys()), ['b'+f for f in list(self._dynamic_fields.keys())]) 

    def info(self, 
             get_str=False
            ):
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
            output.append(f"{'len_vel':<20}: {len(self.vels)}")
            output.append(f"{'vel[0]':<20}: [{self.vels[0,0].value:.2f}, {self.vels[0,1].value:.2f}, {self.vels[0,2].value:.2f}] {self.units['velocity']}")
            output.append(f"{'len_mass':<20}: {len(self.masses)}")
            output.append(f"{'mass[0]':<20}: {self.masses[0]}")
            output.append(f"{'len_ids':<20}: {len(self.IDs)}")
            
        except:
            output.append(f"{'len_pos':<20}: {len(self.coords)}")
            output.append(f"{'len_vel':<20}: {len(self.coords)}")
            output.append(f"{'len_mass':<20}: {len(self.coords)}")

        output.append(f"{'cm':<20}: {self.cm}")
        output.append(f"{'vcm':<20}: {self.vcm}")
        output.append(f"{'rh, rh3D':<20}: {self.rh}, {self.rh3d}")

        if get_str:
            return "\n".join(output)
        else:
            print("\n".join(output))
            return None






































