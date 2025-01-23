import yt
import h5py
import numpy as np
import pandas as pd
from pathlib import Path
import warnings

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
    
    def compute_particle_gravity(self, **kwargs):
        """Adds gravitational potential for stars.
        """
        def _grav_stars(field, data):
            from pytreegrav import PotentialTarget    

            return data["stars", "mass"] * unyt_array(
                PotentialTarget(
                    pos_source=np.concatenate( (data["stars", "coordinates"], data["darkmatter", "coordinates"]) ).to("kpc"), 
                    pos_target=data["stars", "coordinates"].to("kpc"), 
                    m_source=np.concatenate( (data["stars", "mass"], data["darkmatter", "mass"]) ).to("Msun"), 
                    softening_target=data["stars", "softening"].to("kpc"),
                    softening_source=np.concatenate( (data["stars", "softening"], data["darkmatter", "softening"]) ).to("kpc"), 
                    G=4.300917270038e-06,
                    theta=0.6,
                    parallel=True,
                    quadrupole=True
                ), 
                "km**2/s**2"
            ) 
        def _grav_dm(field, data):
            from pytreegrav import PotentialTarget    

            return data["darkmatter", "mass"] * unyt_array(
                PotentialTarget(
                    pos_source=np.concatenate( (data["stars", "coordinates"], data["darkmatter", "coordinates"]) ).to("kpc"), 
                    pos_target=data["darkmatter", "coordinates"].to("kpc"), 
                    m_source=np.concatenate( (data["stars", "mass"], data["darkmatter", "mass"]) ).to("Msun"), 
                    softening_target=data["darkmatter", "softening"].to("kpc"),
                    softening_source=np.concatenate( (data["stars", "softening"], data["darkmatter", "softening"]) ).to("kpc"), 
                    G=4.300917270038e-06,
                    theta=0.6,
                    parallel=True,
                    quadrupole=True
                ), 
                "km**2/s**2"
            )
        
        self.ds.add_field(
            ("stars", "grav_potential"),
            function=_grav_stars,
            sampling_type="local",
            units="Msun*km**2/s**2",
            force_override=True
        )
        self.ds.add_field(
            ("stars", "grav_potential"),
            function=_grav_dm,
            sampling_type="local",
            units="Msun*km**2/s**2",
            force_override=True
        )
    
    def compute_particle_kinetic(self, v_cm, **kwargs):
        """Computes the kinetic energy of the particles with respect to the center of mass.
        """
        self.ds.add_field(
            ("stars", "kinetic_energy"),
            function=lambda field, data: 0.5 * data["stars", "mass"] * np.linalg.norm(data["stars","velocity"] - v_cm, axis=1) ** 2,
            sampling_type="local",
            units="Msun*km**2/s**2",
            force_override=True
        )
        self.ds.add_field(
            ("darkmatter", "kinetic_energy"),
            function=lambda field, data: 0.5 * data["darkmatter", "mass"] * np.linalg.norm(data["darkmatter","velocity"] - v_cm, axis=1) ** 2,
            sampling_type="local",
            units="Msun*km**2/s**2",
            force_override=True
        )

    def compute_particle_energy(self, **kwargs):
        """Computes the kinetic energy of the particles with respect to the center of mass.
        """
        self.ds.add_field(
            ("stars", "total_energy"),
            function=lambda field, data: data["stars", "kinetic_energy"] + data["stars", "grav_potential"],
            sampling_type="local",
            units="Msun*km**2/s**2",
            force_override=True
        )
        self.ds.add_field(
            ("darkmatter", "total_energy"),
            function=lambda field, data: data["darkmatter", "kinetic_energy"] + data["darkmatter", "grav_potential"],
            sampling_type="local",
            units="Msun*km**2/s**2",
            force_override=True
        )
    
    def select_particles_init(self, center, vel, rvir, RRvir):
        """Does the particle selection at the first time. If RRvir > 2 it does
        selection mode 1, else does selection mode 2
        """
        if RRvir > 2:
            self.ds.add_field(
                ("stars", "r"),
                function=lambda field, data: np.linalg.norm(data["stars", "coordinates"] - center, axis=1),
                sampling_type="local",
                units="Msun*km**2/s**2",
                force_override=True
            )
            self.ds.add_field(
                ("darkmatter", "r"),
                function=lambda field, data: np.linalg.norm(data["darkmatter", "coordinates"] - center, axis=1),
                sampling_type="local",
                units="Msun*km**2/s**2",
                force_override=True
            )
            self.compute_particle_kinetic(vel)
            self.compute_particle_energy(v_cm)
            
            yt.add_particle_filter(
                "stars_selection", 
                function=lambda pfilter, data: (data[pfilter.filtered_type, "total_energy"] < 0) or (data[pfilter.filtered_type, "r"] < rvir), 
                filtered_type="stars", 
                requires=["total_energy"]
            )
            yt.add_particle_filter(
                "darkmatter_selection", 
                function=lambda pfilter, data: (data[pfilter.filtered_type, "total_energy"] < 0) or (data[pfilter.filtered_type, "r"] < rvir), 
                filtered_type="darkmatter", 
                requires=["total_energy"]
            )
            
            sp = ds.sphere(center, 2 * rvir)
            
            

        else:
            raise Exception(f"You must provide RRvir>2")
            
        return particle_id_E 
        
    def track_halos(self, halos_tbt, output, **kwargs):
        """Tracks the halos_tbt (to-be-tracked, not throw-back-thursday) halos. Each halos starting poit can be a different snapshot
        """

        if not isinstance(self.CompleteTree, pd.DataFrame) or not isinstance(self.equivalence_table, pd.DataFrame):
            raise ValueError("Either CompleteTree or equivalence_table are not pandas dataframes! or are not properly set!")
        
        self.halos_tbt = halos_tbt.sort_values("Snapshot")
        self._create_snapztdir()
        self._create_thtrees()

        file_path = Path(output)
        if file_path.is_file():
            outpùt2 = output[:-5] + "_v2" + output[-5:]
            warnings.warn(f"The file {output} already exists", RuntimeWarning)

        self._create_hdf5_file(fname=output)

        active_halos = set()
        terminated_halos = set()
        live_halos = set()
        
        for index, row in self.snap_z_t_dir.iterrows():
            snapshot, redshift, time, pdir = row["Snapshot"], row["Redshift"], row["Time"], row["pdir"]
            
            inserted_halos = self.halos_tbt[self.halos_tbt["Snapshot"] == snapshot]["Sub_tree_id"].astype(int).values
            active_halos.update(inserted_halos)
                
            live_halos = active_halos - terminated_halos
            if not live_halos:
                continue
            if not active_halos:
                raise Exception(f"You dont have any active halos at this point! (snapshot {snapshot}). This means that something went wrong!")
            else:
                self.ds = config.loader(pdir)
                self.compute_particle_gravity()


            for subtree in live_halos:
                if subtree in inserted_halos:
                    member_particles = self.select_particles(
                        center=self.ds.arr(self.halos_tbt[self.halos_tbt["Sub_tree_id"] == subtree][["position_x", "position_y", "position_z"]].astype(float).values, 'kpccm'),
                        vel=self.ds.arr(self.halos_tbt[self.halos_tbt["Sub_tree_id"] == subtree][["velocity_x", "velocity_y", "velocity_z"]].astype(float).values, 'km/s'),
                        rvir=self.ds.quan(self.halos_tbt[self.halos_tbt["Sub_tree_id"] == subtree]["virial_radius"].astype(float).values, 'kpccm'),
                        RRvir=self.ds.quan(self.halos_tbt[self.halos_tbt["Sub_tree_id"] == subtree]["R/Rvir"].astype(float).values, '')
                    )
                else:
                    pass
                #pot + trackeo
                #add particles to self.f
                #compute statistics
                #add statistics to csv
#            for subtree in live_halos:
                #check for merging with main halo or merges between them using csvs. Now all of them are up to date.

            del self.ds

        self.f.close()
        return active_halos, live_halos, terminated_halos
                
            
            

    
        
        
        

        
    

































































