import yt
import os
import time
import numpy as np
from tqdm import tqdm
from pytreegrav import Potential, PotentialTarget
from unyt import unyt_array, unyt_quantity
from unyt.physical_constants import G

from src.config import config


class ptype:
    """ParticleType class that contains coordinates, masses, velocities, IDs, angular momentum and methods to compute various other
    quantities of interest, that only depend on the given particle type (e.g. the total energy depends on the potential energy which depends on other part-types). 
    It is used to represent the various particle types contained in a snapshot, and that eventually are loaded via the Snapshot class.

    The object can be initialized by providing the particle data, particle type etcetera, or by masking another instance of ParticleType to only the desired data.
    """

    ##########################################################
    ###                                                    ###
    ##                    INNIT FUNCTION                    ##
    ###                                                    ###
    ##########################################################
    
    def __init__(self,
                 ad = None,
                 pt = None,
                 center = None,
                 softenings = None,
                 unit_system = {'dist': '1*kpc', 
                                'vel': '1*km/s', 
                                'time': '1*Gyr', 
                                'mass': '1*Msun', 
                                'temp': '1*K'
                               },
                 object = None,
                 object_mask = None,
                 data_order = None
                ):
        """Initialization of attributes.
        """
        if object_mask is not None:
            self.masses =  object.masses[object_mask]
            self.velocities =  object.velocities[object_mask]
            self.ids =  object.ids[object_mask]
            self.angular_momentums =  object.angular_momentums[object_mask]
            self.coords = object.coords[object_mask]
            self.softs = object.softs[object_mask]

            self._units = object._units
            
        else:
            self._units = unit_system
            
            self.masses = ad[pt, 'Mass'][data_order].in_units(self._units['mass'])
            self.velocities = ad[pt, 'Velocities'][data_order].in_units(self._units['vel'])
            self.ids = np.array(ad[pt, 'ParticleIDs'][data_order].value, dtype="int")
            self.angular_momentums = ad[pt, 'particle_angular_momentum'][data_order].in_units(f"({self._units['mass']})*({self._units['dist']})**2/({self._units['time']})")
            if center is not None:
                self.coords = ad[pt, 'Coordinates'][data_order].in_units(self._units['dist']) - center.in_units(self._units['dist'])
            else:
                self.coords = ad[pt, 'Coordinates'][data_order].in_units(self._units['dist'])
                        
            if softenings is None:
                self.softs = unyt_array(np.zeros_like(self.masses), self._units['dist'])        
            else:
                self.softs = softenings[data_order].in_units(self._units['dist'])
    

            
        self._num_part = len(self.ids)
        
        self.bound_mask = np.full(self._num_part, True)
        self.bound_particles = self.ids[self.bound_mask]
        self.removed_mask = np.full(self._num_part, False)
        self.removed_particles = self.ids[self.removed_mask]
        
        self.cm = None
        self.v_cm = None
        
        self.L_orbital = None
        self.L_from_cm = None

        self.potentials = None
        self.ext_potentials = None
        self.total_cm_energies = None
        self.total_energies = None

