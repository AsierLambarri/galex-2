#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 20 10:03:19 2024

@author: asier
"""
import numpy as np
from unyt import unyt_array, unyt_quantity

from .config import config



class BaseHaloObject:
    """BaseSimulationObject that contains information shared between all objects in a simulation.
    """
    _shared_attrs = {
        "darkmatter": {
            "quantities": {
                "cm": None,
                "vcm": None,
                "rh": None,
                "rh3d": None,
                "rs": None,
                "rvir": None,
                "vmax": None
            },
            "moments": {
                "sigma": None
            }
        },

        
        "stars": {
            "quantities": {
                "cm": None,
                "vcm": None,
                "rh": None,
                "rh3d": None,
                "rt": None
            },
            "moments": {
                "sigma_los": None
            }
        },

        
        "gas": {
            "quantities": {
                "cm": None,
                "vcm": None,
                "rh": None,
                "rh3d": None,
                "rt": None
            },
            "moments": {
                "sigma_los": None
            }            
        }
    }
        

    def __init__(self):
        self._parent = None  
        
        self.base_units = config.base_units
        self.ptypes = config.ptypes
        
        self.los = [1, 0, 0]
        self.basis = np.identity(3)
        self._old_to_new_base = np.identity(3)


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
                for category in list(cls._shared_attrs[pt].keys()):
                    if key in cls._shared_attrs[pt][category]:
                        cls._shared_attrs[pt][category][key] = cls.format_value(value)
                    else:
                        raise ValueError(f"Invalid shared attribute '{key}' for type '{pt}'")
        return None
    
    @classmethod
    def get_shared_attr(cls, pt, key=None, cat=None):
        """Get a specific shared attribute for a particle type.
        """
        if pt not in cls._shared_attrs:
            raise ValueError(f"Unknown particle type: {pt}")

        if cat is not None:
            return cls._shared_attrs[pt].get(cat)
        elif key is not None:
            for category in list(cls._shared_attrs[pt].keys()):
                if key in list(cls._shared_attrs[pt][category]):
                    return cls._shared_attrs[pt][category].get(key)
        else:
            return cls._shared_attrs[pt]

    @classmethod
    def update_shared_attr(cls, pt, key, value):
        """Update a specific shared attribute for a particle type.
        """
        for category in list(cls._shared_attrs[pt].keys()):
            if  key in list(cls._shared_attrs[pt][category].keys()):
                cls._shared_attrs[pt][category][key] = value
            else:    
                raise ValueError(f"Cannot update: '{key}' not valid for '{pt}'")

    @classmethod
    def list_shared_attributes(cls, pt, category):
        """List all shared attributes for a given particle type."""
        return list(cls._shared_attrs[pt].get(category, {}).keys())

    @classmethod
    def clean_shared_attrs(cls, pt):
        """Reset all shared attributes for a specific particle type to None."""
        if pt not in cls._shared_attrs:
            raise ValueError(f"Unknown particle type: {pt}")
        for category in list(cls._shared_attrs[pt].keys()):
            for key in list(cls._shared_attrs[pt][category].keys()):
                cls._shared_attrs[pt][category][key] = None
        return None


    
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
    
    
    








