import yt
import numpy as np
import pandas as pd
from unyt import unyt_array, unyt_quantity

from yt.utilities.logger import ytLogger


class RCTHalo:
    """Class containing information of a particular halo.
    """
    def __init__(self, 
                Table,
                sub_tree,
                units
                ):
        """Init function.
        """
        self.snapnum = np.array(Table[Table['Sub_tree_id'] == sub_tree].sort_values("Snapshot")['Snapshot'].values, dtype="int")
        self.z = np.array(Table[Table['Sub_tree_id'] == sub_tree].sort_values("Snapshot")['Redshift'].values, dtype="float")
        self.pos = Table[Table['Sub_tree_id'] == sub_tree].sort_values("Snapshot")[['position_x','position_y','position_z']].values * units['lencom']
        self.vel = Table[Table['Sub_tree_id'] == sub_tree].sort_values("Snapshot")[['velocity_x','velocity_y','velocity_z']].values * units['vel']
        self.rvir = Table[Table['Sub_tree_id'] == sub_tree].sort_values("Snapshot")['virial_radius'].values * units['lencom']
        




class ParticleTracker:
    """Particle tracking for Rockstar+Consistent-Trees identified halos.
    """
    def __init__(self,
                HaloTable,
                snap_dir,
                sub_tree_id,
                ts = None,
                uid = None,
                overlap = 10,
                verbose = False,
                logger = 30
                ):
        """Init Function.
        """
        ytLogger.setLevel(30)  
        if ts:
            self.ts = ts
        else:
            self.ts = yt.load(snap_dir)





        self.host = RCTHalo(HaloTable, 1, 
                            {'lencom': self.kpccm,
                             'len': self.kpc, 
                             'vel': self.kms,
                             'mass': self.msun
                            }
                           ) 
        self.satellite = RCTHalo(HaloTable, sub_tree_id, 
                            {'lencom': self.kpccm,
                             'len': self.kpc, 
                             'vel': self.kms,
                             'mass': self.msun
                            }
                           ) 

        self.fsnap = self.satellite.snapnum[-1]
        self.snaps_to_run = self.host.snapnum[np.where(self.host.snapnum == self.fsnap)[0][0]-overlap:]


        ytLogger.setLevel(logger)  


    def track_particles(self):
        """Tracks particles starting from fsnap-overlap until theend of the simulation, or until the halo is completely disrupted.
        """

        for snapnum in self.snaps_to_run:
            print(snapnum)
            ds = self.ts[snapnum]



        





        







        