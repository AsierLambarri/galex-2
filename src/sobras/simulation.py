import os
import yt
import numpy as np
from yt.utilities.logger import ytLogger


from .loaders import parse_filename, sort_snaps



class Simulation:
    """Simulation object that stores general information about and derived quantities about the loaded simulation; i.e. files, file directories,
    halo catalogues, equivalence tables between outputs, redshifts and halo catalogue. Contains functions to process, store and read these quantities.
    Additionally, it enables selecting only a small subset of the simulation for latter processing. This is the most important piece of work that this
    class does.
    """
    def __init__(self,
                 snapshots_path,
                 halo_catalogue,
                 equivalence_table,
                 basename,
                 simulation_name="",
                 **kwargs
                ):
        
        ytLogger.setLevel(30)  

        self.pdir = snapshots_path
        self.cdir = halo_catalogue
        self.equivalence_table = 

        self.snapshots = equivalence_table[self.kw_fields['equivalences', 'snapshots']]




    def select_region(self,
                      z,
                      left_corner=None,
                      right_corner=None,
                      center=None,
                      radius=None
                     ):
        """Selects a region of the whole simulation to work with. Useful for when the simulation is too big or the user is only interested in working with a 
        certain region of the simulation, e.g. a cosmological simulation where more than one object of the wanted type is present. Regions can only be queried as
        rectangles or spheres.

        Parameters
        ----------
        z : float
            Redshift at which the region will be queried. It is rounded to the closest value between all the snapshots.
            
        left/right_corner : array[float|int], unyt_array[float|int] or equivalent, optional
            Left and Right corners of the bounding box. Default is None.

        center : array[float|int], unyt_array[float|int] or equivalent, optional
            Center of the bounding sphere.

        radius : float, unyt_quantity or equivalent
            Radius of bounding sphere.

        Returns
        -------
        region : Simulation.ZoomIn
        """
        

        

class ZoomInSim:

    def __init__(self,
                 snapshots_path,
                 halo_catalogue_path,
                 equivalence_table,
                 basename,
                 simulation_name="generic-name",
                 merger_tree=None,
                 **kwargs
                ):
    kw_equiv = {
                'snapnum': "snapid",
                'snapname': "snapshot"
               }    
    self.simname = simulation_name
    self.pdir = snapshots_path
    self.cdir = halo_catalogue    

    self.snapequiv = equivalence_table
    self.merger_tree = merger_tree

    self.snapshots = self.snapequiv[kw_equiv['snapname']]
        


























