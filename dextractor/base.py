#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 20 10:03:19 2024

@author: asier
"""
import numpy as np
from unyt import unyt_array
from .config import config
from .class_methods import gram_schmidt, center_of_mass, refine_center, half_mass_radius


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
        self.basis = gram_schmidt(los)
        if self._parent:
            self._parent._set_los(los)  
            
        return None
    
    
    
class BaseParticleType:
    """BaseParticleType class that implements common methods and attributes for particle ensembles. These methods and attributes
    are accesible for all particle types and hence this class acts as a bridge between stars, darkmatter and gas, allowing 
    them to access properties of one another. This makes sense, as particles types in cosmological simulations are coupled to
    each other.
    
    It also simplifies the code, as a plethora of common methods are displaced to here.
    """
    _shared_attrs = {
        "darkmatter": {"rockstar_center": None, "rockstar_vel": None, "rvir": None, "rs": None, "c": None, 'vmax': None, 'vrms': None},
        "stars": {"ML": None},
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
        if pt not in cls._shared_attrs:
            raise ValueError(f"Unknown particle type: {pt}")
        
        for key, value in kwargs.items():
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
    

    









