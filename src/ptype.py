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
        
        


