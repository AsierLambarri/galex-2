#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 20 10:03:19 2024

@author: asier
"""
import numpy as np
from unyt import unyt_array
from .config import config
from .class_methods import gram_schmidt, vectorized_base_change, center_of_mass_pos, center_of_mass_vel, refine_center, half_mass_radius, easy_los_velocity, encmass


class BaseSimulationObject:
    """BaseSimulationObject that contains information shared between all objects in a simulation.
    """
    def __init__(self):
        self._parent = None  
        
        self.base_units = config.base_units
        self.units = config.working_units
        self.ptypes = config.ptypes
        self.fields = config.fields
        
        self.loader = config.default_loader
        self.parser = config.default_parser
        
        self.los = [1, 0, 0]
        self.basis = np.identity(3)
        self._old_to_new_base = np.identity(3)
        
    def set_parent(self, parent):
        """Sets the parent of this object and ensures attributes propagate from parent to child.
        """
        self._parent = parent
        if self._parent:
            if self.units is None:
                self.units = self._parent.units
            if self.basis is None:
                self.basis = self._parent.basis
    
        return None 
    
    def _set_units(self, units):
        """Sets the units for this object and propagates to children if any.
        """
        self.units = units
        if self._parent:
            self._parent._set_units(units)  
            
        return None
            
    def _set_los(self, los):
        """Sets the coordinate basis for this object and propagates to children if any.
        """
        self.los = los
        self._old_to_new_base = np.linalg.inv(gram_schmidt(los)) @ self.basis
        self.basis = gram_schmidt(los)

        if self._parent:
            self._parent._set_los(los)   

        return None
    
    
    
class BaseComponent:
    """BaseParticleType class that implements common methods and attributes for particle ensembles. These methods and attributes
    are accesible for all particle types and hence this class acts as a bridge between stars, darkmatter and gas, allowing 
    them to access properties of one another. This makes sense, as particles types in cosmological simulations are coupled to
    each other.
    
    It also simplifies the code, as a plethora of common methods are displaced to here.
    """
    _shared_attrs = {
        "darkmatter": {
            "rockstar_center": None, 
            "rockstar_vel": None,
            "cm": None,
            "vcm": None,
            "rh": None,
            "rh3d": None,
            "rvir": None, 
            "rs": None, 
            "c": None, 
            "vmax": None, 
            "vrms": None
        },
        
        "stars": {
            "cm": None,
            "vcm": None,
            "rh": None,
            "rh3d": None,
            "ML": None,
            "sigma_los": None
        },

        "gas": {
            "cm": None,
            "vcm": None,
            "rh": None,
            "rh3d": None,
            "sigma_los": None
        }
        
    }

    @classmethod
    def format_value(cls, value):
        """Formats value using unyt if value != None, else returns none
        """
        if value is None:
            return None
            
        if type(value) == tuple:
            assert len(value) >= 1 and len(value) <= 2, f"Tuple must be of the formt (X,)==(X,'dimensionless') or (X,unit). Your provided {value}."
            if value[0] is None: return None
            else: return unyt_array(*value)
                
        else:
            return cls.format_value((value,))

    @classmethod
    def set_shared_attrs(cls, pt, kwargs):
        """Set class-level shared attributes for a specific particle type.
        """
        tmp = {
            "darkmatter": "dm_params",
            "stars": "stars_params",
            "gas": "gas_params"
        }

        if pt not in cls._shared_attrs:
            raise ValueError(f"Unknown particle type: {pt}")
        if tmp[pt] not in kwargs.keys():
            pass
        else:
            for key, value in kwargs[tmp[pt]].items():
                if key in cls._shared_attrs[pt]:
                    cls._shared_attrs[pt][key] = cls.format_value(value)
                else:
                    raise ValueError(f"Invalid shared attribute '{key}' for type '{pt}'")

        return None
    
    @classmethod
    def get_shared_attr(cls, pt, key):
        """Get a specific shared attribute for a particle type.
        """
        if pt not in cls._shared_attrs:
            raise ValueError(f"Unknown particle type: {pt}")
        return cls._shared_attrs[pt].get(key)

    @classmethod
    def update_shared_attr(cls, pt, key, value):
        """Update a specific shared attribute for a particle type.
        """
        if (pt in cls._shared_attrs) and (key in cls._shared_attrs[pt]):
            cls._shared_attrs[pt][key] = value
        else:
            raise ValueError(f"Cannot update: '{key}' not valid for '{pt}'")

    @classmethod
    def list_shared_attributes(cls, pt):
        """List all shared attributes for a given particle type."""
        return list(cls._shared_attrs.get(pt, {}).keys())

    @classmethod
    def clean_shared_attrs(cls, pt):
        """Reset all shared attributes for a specific particle type to None."""
        if pt not in cls._shared_attrs:
            raise ValueError(f"Unknown particle type: {pt}")
        for key in cls._shared_attrs[pt].keys():
            cls._shared_attrs[pt][key] = None

        return None


    



    

    def _priv__getattr__(self, field_name):
        """Dynamical loader for accessing fields.
        """
        funits = {
            'coords': self.units['length'],
            'hsml': self.units['length'],
            'masses': self.units['mass'],
            'masses_ini': self.units['mass'],
            'vels': self.units['velocity'],
            'ages': self.units['time'],
            'metallicities': self.units['dimensionless']
        }
        vec_fields = ['coords', 'vels', 'bcoords', 'bvels', 'cyl_coords', 'cyl_vels', 'sp_coords', 'sp_vels', 'box_coords', 'box_vels']
        if field_name  in self._dynamic_fields.keys():
            if field_name in self._fields_loaded:
                return self._fields_loaded[field_name].in_units(funits[field_name]) if field_name in funits else self._fields_loaded[field_name]
            else:
                field = (self._base_ptype, self._dynamic_fields[field_name])
                if field_name in vec_fields:
                    self._fields_loaded[field_name] = vectorized_base_change(np.linalg.inv(self.basis), self._data[field].in_units(funits[field_name]) if field_name in funits else self._data[field])
                else:
                    self._fields_loaded[field_name] = self._data[field].in_units(funits[field_name]) if field_name in funits else self._data[field]
                return self._fields_loaded[field_name]
                
        elif field_name  in ['b'+ f for f in list(self._dynamic_fields.keys())]:
            if field_name in self._fields_loaded:
                return self._fields_loaded[field_name].in_units(funits[field_name[1:]]) if field_name[1:] in funits else self._fields_loaded[field_name]
            else:
                field = (self._base_ptype, self._dynamic_fields[field_name[1:]])
                if field_name in vec_fields:
                    self._fields_loaded[field_name] = vectorized_base_change(np.linalg.inv(self.basis), self._data[field][self.bmask].in_units(funits[field_name[1:]]) if field_name[1:] in funits else self._data[field][self.bmask])
                else:
                    self._fields_loaded[field_name] = self._data[field][self.bmask].in_units(funits[field_name[1:]]) if field_name[1:] in funits else self._data[field][self.bmask]
                return self._fields_loaded[field_name]
        
        try:
            return self.__getattribute__(field_name)
        except AttributeError:
            raise AttributeError(f"Field '{field_name}' not found for particle type {self.ptype}. "+ f"Available fields are: {list(self._dynamic_fields.keys())}")
        
        AttributeError(f"Field {field_name} not found for particle type {self.ptype}. Available fields are: {list(self._dynamic_fields.keys())}")
        return None

    def _delete_vectorial_fields(self):
        """Deletes vectorial fields: a.k.a. coordinates and velocities from cache when called. Used to force the dynamical field loader to re-load vectorial
        fields after a new line of sight has been set, so that the fields are properly oriented.
        """
        f = ['coords', 'vels', 'bcoords', 'bvels']
        for key in f:
            try:      
                del self._fields_loaded[key]
            except:
                continue

        return None

    def _default_center_of_mass(self):
        """Computes coarse CoM using all the particles as 

                CoM = sum(mass * pos) / sum(mass)        
        """
        if self.masses.sum() != 0:
            self.cm = center_of_mass_pos(self.coords, self.masses)
        else:
            self.empty_component = True
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
        self._delete_vectorial_fields()
        if self.cm is not None:
            self.cm = self._old_to_new_base @ self.cm
        if self.vcm is not None:
            self.vcm = self._old_to_new_base @ self.vcm

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
        if self.use_bound_if_computed:
            self.centering_results = refine_center(self.coords[self.bmask], self.masses[self.bmask],
                                                   scaling,
                                                   method,
                                                   delta,
                                                   alpha,
                                                   m,
                                                   nmin,
                                                   mfrac)
        else:
            self.centering_results = refine_center(self.coords, self.masses,
                                           scaling,
                                           method,
                                           delta,
                                           alpha,
                                           m,
                                           nmin,
                                           mfrac)
            
        self.cm = self.centering_results['center']
        return self.cm
    
    def half_mass_radius(self, mfrac=0.5, project=False):
        """By default, it computes 3D half mass radius of a given particle ensemble. If the center of the particles 
        is not provided, it is estimated by first finding the median of positions and then computing a refined CoM using
        only the particles inside r < 0.5*rmax.
        
        There is also an argument to compute other ALPHA_mass_radius of an arbitrary mass fraction. The desired ALPHA_mass_radius  
        is computed via rootfinding using scipy's implementation of brentq method.
        
        OPTIONAL Parameters
        -------------------
        mfrac : float
            Mass fraction of desired radius. Default: 0.5 (half, mass radius).
        project: bool
            Whether to compute projected quantity or not.
        
        Returns
        -------
        MFRAC_mass_radius : float
            Desired mfrac mass fraction radius estimation. Provided in same units as pos, if any.
        """
        if (self.use_bound_if_computed) and (self.bmask is not None):
            rh = half_mass_radius(self.coords[self.bmask], self.masses[self.bmask], self.cm, mfrac, project=project)
        else:
            rh = half_mass_radius(self.coords, self.masses, self.cm, mfrac, project=project)

        if project:
            self.rh = rh
        else:
            self.rh3d = rh
            
        return rh


    def los_velocity(self, rcyl=(1, 'kpc'), return_projections = False):
        """Computes the line of sight velocity of particles inside a cylinder of radius rcyl aligned with the line of sight direction.
        """
        mask = np.linalg.norm(self.coords[:, 0:2] - self.cm[0:2], axis=1) < unyt_array(*rcyl)
        
        los_velocities = easy_los_velocity(self.vels[mask], self.los)
        losvel = np.std(los_velocities)
        
        if return_projections:
            return losvel, los_velocities
        else:
            return losvel

    def enclosed_mass(self, r0, center):
        """Computes the enclosed mass on a sphere centered on center, and with radius r0.
        """
        mask = np.linalg.norm(self.bcoords - center, axis=1) <= r0
        return self.bmasses[mask].sum()

    

    









