import yt
import h5py
import numpy as np
import pandas as pd
from pytreegrav import Potential, PotentialTarget

from .config import config
from .class_methods import load_ftable


class Tracker:
    """Class that tracks halo particles after rockstar + consistent-trees looses the halos.
    Enables following the halos as they merge with much more massive halos. The main 
    purpose of this code is to track Dwarf Galaxies when they merge with MW-like galaxies.
    
    
    If a single halo is provided, it falls back to SingleTracker. If a list of halos is
    provided it tracks all at once. Gravitational computations are done with pytreegrav.
    
    When tracking, the code computes halo position and velocity using only DM particles, but 
    also computes center of stars, virial, tidal radius and bound fraction. The code checks
    for mergers by determining if |x_1 - x_2| < rh both with the main MW-like halo and 
    between tracked halos.
    
    Particles are selected as all those that satisfy: 1) Inside 2 * Rvir_sub or 
    2) particles inside of Rvir_sub when the halo was at 2*Rvis_main.
 
    The number of particles is fixed: mergers of subhalos can happen with subhalos present 
    in  the initial particle selection, but external particles cannot be accreted. For 
    dwarf galaxy tracking, this is a sensible approximation.
    """
    def __init__(self, 
                 sim_dir,
                 catalogue=None
                 ):
        """Init function.
        """
        self.sim_dir = sim_dir
        if catalogue is not None:
            self.set_catalogue(catalogue)
                
        
    @property
    def CompleteTree(self):
        return self._CompleteTree
    @property
    def PrincipalLeaf(self):
        if self._PrincipalLeaf is None:
            return None
        else:
            return self._PrincipalLeaf.sort_values("Snapshot", ascending=True)
    @property
    def equivalence_table(self):
        if self.PrincipalLeaf is None:
            return self._equiv
        else:
            return self._equiv[ self._equiv['snapshot'].isin(self.PrincipalLeaf['Snapshot']) ].sort_values("snapshot", ascending=True)
    @property
    def max_snap(self):
        return self.PrincipalLeaf["Snapshot"].max()
    @property
    def min_snap(self):
        return self.halos_tbt["Snapshot"].max()
    @property
    def snap_z_t_dir(self):
        return self._snap_z_time
    @property
    def tracked_halo_trees(self):
        return self._thtrees
        
    def _create_hdf5_file(self, fname): 
        """Creates an hdf5 file to store the data. 
        Each snapshot goes in a column.
        Each halo has a separate group.
        Each halo has four datasets: pIDs and Es for ds and stars.
        """
        self.f = h5py.File(fname, "a")
        
        for index, row in self.halos_tbt.sort_values("Sub_tree_id").iterrows():
            subgroup = f"sub_tree_{int(row['Sub_tree_id'])}"
            cols = self.max_snap - row["Snapshot"] + 1
            
            self.f.create_group(subgroup)
            self.f[subgroup].create_dataset(
                "darkmatter_ids", 
                shape=(0, cols), 
                maxshape=(None, cols), 
                dtype=np.int32,
                fillvalue=np.nan
            )
            self.f[subgroup].create_dataset(
                "darkmatter_energies", 
                shape=(0, cols), 
                maxshape=(None, cols), 
                dtype=np.float32,
                fillvalue=np.nan
            )
            self.f[subgroup].create_dataset(
                "stars_ids", 
                shape=(0, cols), 
                maxshape=(None, cols), 
                dtype=np.int32,
                fillvalue=np.nan
            )
            self.f[subgroup].create_dataset(
                "stars_energies", 
                shape=(0, cols), 
                maxshape=(None, cols), 
                dtype=np.float32,
                fillvalue=np.nan
            )
    def _create_snapztdir(self):
        """Creates _snap_z_time and adds column with relevant particle file directories.
        """
        self.snap_names = self.equivalence_table[ self.equivalence_table["snapshot"] >= self.halos_tbt["Snapshot"].min() ]["snapname"].values
        self.snap_dirs = [self.sim_dir + snap for snap in self.snap_names]
        self._snap_z_time = self.PrincipalLeaf[ self.PrincipalLeaf["Snapshot"] >= self.halos_tbt["Snapshot"].min() ][["Snapshot", "Redshift", "Time"]].reset_index(drop=True)
        self._snap_z_time["pdir"] = [self.sim_dir + snap for snap in self.snap_names]

    def _create_thtrees(self):
        """Creates a dictionary whose keys are the sub_trees of each of the tracked trees in pd.DataFrame format.
        When tracking, statistics like the center of mass, virial radius, half mass radius etc are appended at each time.
        """
        self._thtrees = {}
        columns = ["Sub_tree_id", "Snapshot", "Redshift", "Time", "mass", "virial_radius", "scale_radius", "tidal_radius", "vrms", "vmax", 
                  "position_x", "position_y", "position_z", "velocity_x", "velocity_y", "velocity_z", "R/Rvir", "stellar_mass"
                 ]
        for sn, subtree in self.halos_tbt[["Snapshot", "Sub_tree_id"]].values:
            cols = int(self.max_snap - sn + 1)
            self._thtrees[f"sub_tree_{int(subtree)}"] = pd.DataFrame(columns=columns, index=range(cols))
            self._thtrees[f"sub_tree_{int(subtree)}"]["Snapshot"] = list(range(int(sn), int(self.max_snap + 1)))
            self._thtrees[f"sub_tree_{int(subtree)}"]["Sub_tree_id"] = int(subtree)
        
    def set_catalogue(self, complete_tree):
        """Sets trees by loading the complete tree and separating the principal-leaf, main and satellite-trees. 
        Computations are only even done with the complete-tree. Assumes that "Sub_tree"s and "TreeNum"s are already
        computed. 
    
        Parameters
        ----------
        complete_tree : str or pd.DataFrame
            Complete tree to be set in place
            
        """
        if isinstance(complete_tree, str):
            self._CompleteTree = pd.read_csv(complete_tree)
        elif isinstance(complete_tree, pd.DataFrame):
            self._CompleteTree = complete_tree
    
        host_subid, tree_num = self.CompleteTree.sort_values(['mass', 'Snapshot'], ascending = (False, True))[["Sub_tree_id", "TreeNum"]].values[0]
        
        self._PrincipalLeaf = self.CompleteTree[self.CompleteTree["Sub_tree_id"] == host_subid]
                
    def set_equivalence(self, equiv):
        """Loads and sets equivalence table.
        """
        if isinstance(equiv, str):
            try:
                self._equiv = load_ftable(equiv)
            except:
                self._equiv = pd.read_csv(equiv)
                
        elif isinstance(equiv, pd.DataFrame):
            self._equiv = equiv
    
        else:
            raise AttributeError(f"Could not set equivalence table!")
            
    def dump_to_hdf5(self, subtree, index, data):
        """Dumps particle IDs, its potential, kinetic and total energies.
        """
        subgroup = f"sub_tree_{int(subtree)}"
        group = self.f[subgroup]
        for field in ["darkmatter_ids", "stars_ids", "darkmatter_energies", "stars_energies"]:
            dset = group[field]
            dset.resize((len(data[field]), dset.shape[1]))
            dset[:, index] = data[field]
            
        return None
    
    def dump_to_csv(self, subtree, snapnum, data):
        """Dumps a csv with halo position, velocity, virial radius etc with the same format as rockstar+consistent trees 
        files created using MergerTrees class
        """
        return None
    
    
    def track_halos(self, halos_tbt, output, **kwargs):
        """Tracks the halos_tbt (to-be-tracked, not throw-back-thursday) halos. Each halos starting poit can be a different snapshot
        """

        if not isinstance(self.CompleteTree, pd.DataFrame) or not isinstance(self.equivalence_table, pd.DataFrame):
            raise ValueError(f"Either CompleteTree or equivalence_table are not pandas dataframes! or are not properly set!")
        
        self.halos_tbt = halos_tbt.sort_values("Snapshot")
        self._create_snapztdir()
        self._create_thtrees()
        self._create_hdf5_file(fname=output)

        active_halos = np.array([], dtype=int)
        for index, row in self.snap_z_t_dir.iterrows():
            snapshot, redshift, time, pdir = row["Snapshot"], row["Redshift"], row["Time"], row["pdir"]
            
            if snapshot in self.halos_tbt["Snapshot"].values: 
                active_halos = np.array(
                    active_halos, 
                    self.halos_tbt[self.halos_tbt["Snapshot"] == snapshot]["Sub_tree_id"]
                )
            if active_halos.shape[0] != 0:
                ds = config.loader
            else:
                raise Exception(f"You dont have any active halos at this point! (snapshot {snapshot}). This means that something went wrong!")

            for subtree in active_halos:
                #pot + trackeo
                #add particles to self.f
                #compute statistics
                #add statistics to csv
            for subtree in active_halos:
                #check for merging with main halo or merges between them using csvs. Now all of them are up to date.
            








                
        self.f.close()
        return active_halos
                
            
            

    
        
        
        

        
    

































































