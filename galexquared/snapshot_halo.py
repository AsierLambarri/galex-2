import yt
import numpy as np
from unyt import unyt_array, unyt_quantity

from .config import config
from .base import BaseHaloObject
from .particle_type import Component
from .class_methods import create_sph_dataset, softmax, create_subset_mask

class SnapshotHalo(BaseHaloObject):
    """zHalo class that implements a variety of functions to analyze the internal structure of a halo at a certain redshift, and the galaxy that
    is contained within it, such as computing their respective moments (x_cm and v_cm), projected and deprojected half-mass radii, total and LOS
    velocity dispersions, surface and volumetric density profiles, computation of dynamical and M_x masses and much more!

    Initialization arguments
    ------------------------
    fn : str
        Path to file containing the halo.

    ds : yt.dataset, optional
        Pre-loaded yt-dataset containing (or not, if you mess up) the desired halo. Useful for avoiding the I/O  overhead of loading the same snapshot multiple times
        when you want to load a bunch of halos in the same snapshot.
        
    snapnum : int or float, optional BUT REQUIRED if loading with subtree or haloid
        Snapshot number of the snapshot where the halo is located.

    subtree, haloid, uid : int or float, optional but AT LEAST ONE is required
        Subtree; uid of the oldest progenitor of the branch identified in the merger tree, haloid; unique halo id at a specific snapshot and 
        universal id of the halo. Only one is required. IF LOADING WITH SUBTREE OR HALOID, the snapshot number is required.

    **kwargs: dictionary of keyword arguments that can be passed when loading the halo. They are the following:

        · units : dict[key : str, value : str]
            Default are Rockstar units: mass : Msun, time : Gyr, length : kpccm, and velocity : km/s.


    
    """
    def __init__(self,
                 fn,
                 from_catalogue=None,
                 center=None,
                 radius=None,
                 dataset=None,
                 **kwargs
                ):
        """Initialize function.
        """
        super().__init__()

        self.fn = fn
        if from_catalogue is not None:
            self._cathalo = from_catalogue.iloc[0]
            self.sp_radius = unyt_quantity(self._cathalo['virial_radius'] / (1 + self._cathalo['Redshift']), 'kpc')
            self.sp_center = unyt_array(self._cathalo[['position_x', 'position_y', 'position_z']].values.astype(float) / (1 + self._cathalo['Redshift']), 'kpc')
        elif center is not None and radius is not None:
            self.sp_center = unyt_array(center[0], center[1])
            self.sp_radius = unyt_quantity(radius[0], radius[1])        
        else:
            raise ValueError("You did not provide neither (center, radius) or from_catalogue!!")

        self._kwargs = kwargs
        self._load_and_parse_data(dataset)
        self.arr = self._ds.arr
        self.quant = self._ds.quan        
        if from_catalogue is not None:
            self._setup_catalogue_shared_params()
        else:
            self.sp_center = self.arr(center[0], center[1])
            self.sp_radius = self.quant(radius[0], radius[1])             


    @property
    def p(self):
        return self.get_shared_attr("halo", cat="properties")
    @property
    def q(self):
        return self.get_shared_attr("halo", cat="quantities")
    @property
    def m(self):
        return self.get_shared_attr("halo", cat="moments")
    @property
    def _ds(self):
        return self.get_shared_attr("halo", cat="data", key="data_set")
    @property
    def _data(self):
        return self.get_shared_attr("halo", cat="data", key="data_source")


        
    def _setup_catalogue_shared_params(self):
        """Loads the parameters present in the catalogue.
        """
        fields = {
            "rockstar_center": self.arr(self._cathalo[['position_x', 'position_y', 'position_z']].values.astype(float), 'kpccm').to("kpc"),
            "rockstar_velocity": self.arr(self._cathalo[['velocity_x', 'velocity_y', 'velocity_z']].values.astype(float), 'km/s'),
            "rockstar_rs": self.quant(self._cathalo['scale_radius'], 'kpccm').to("kpc"),
            "rockstar_rvir": self.quant(self._cathalo['virial_radius'], 'kpccm').to("kpc"),
            "rockstar_vmax": self.quant(self._cathalo['vmax'], 'km/s'),
            "rockstar_vrms": self.quant(self._cathalo['vrms'], 'km/s')
        }
        self.set_shared_attrs("halo", fields)
        self.sp_center = fields["rockstar_center"]
        self.sp_radius = fields["rockstar_rvir"]
    
    def _load_and_parse_data(self, dataset):
        """
        Calls the module-level parser to parse the loaded dataset. This function is directly linked to the loader, sinde it works with the object
        type provided on it. If you attempt to change one, you must change both and make sure that the data is returned in the correct format 
        and specified order:
            
                                                                 units : dict[str : str]
                        config.parser(fn, center, radius) -----> metadata : dict[str : str | float | unyt_quantity]
                                                                 data : hashable as data[particle_type, field]
                                                                 
        As the particle type names change from one scheme to another, these MUST be set by the user beforehand as: config.particles : dict[str : str], where the equivalences between
        gas, dm and star particles are listed.
                                                                 

        Returns
        -------
        base_units : dict[str : str]
        metadata : dict[str : str | float | unyt_quantity]
        parsed_data : returned object is of the type returned by (user) specified config.parser
        
        Each return is saved as an attribute of the config, zHalo or zHalo.ptype instance: parsed_data is saved inside ptype, for each particle type. base_units are saved inside config and
        the metadata is saved inside zHalo.
        """
        ds = config.loader(self.fn) if dataset is None else dataset
        data = ds.sphere(self.sp_center, self.sp_radius)
        self.update_shared_attr("halo", "data_set", ds)
        self.update_shared_attr("halo", "data_source", data)
        self._all_data = self._ds.all_data()

        metadata = {
            'redshift': self._ds.current_redshift,
            'scale_factor': 1 / (self._ds.current_redshift + 1),
            'time': self._ds.current_time,
            'H0': self._ds.cosmology.hubble_constant,
            'omega_matter': self._ds.cosmology.omega_matter,
            'omega_lambda': self._ds.cosmology.omega_lambda,
            'omega_radiation': self._ds.cosmology.omega_radiation,
            'omega_curvature': self._ds.cosmology.omega_curvature,
            'omega': self._ds.cosmology.omega_matter + self._ds.cosmology.omega_lambda +
            self._ds.cosmology.omega_radiation + self._ds.cosmology.omega_curvature
        }
        
        self._metadata = metadata
     
        self.stars = Component("stars", **self._kwargs["stars_params"] if "stars_params" in self._kwargs else {})
        self.darkmatter = Component("darkmatter", **self._kwargs["dm_params"] if "dm_params" in self._kwargs else {})
        self.gas = Component("gas", **self._kwargs["gas_params"] if "gas_params" in self._kwargs else {})

        self.set_shared_attrs("halo", self._kwargs["halo_params"] if "halo_params" in self._kwargs else None)
        self.set_shared_attrs("halo", self._metadata)

        self.stars.sp_center, self.stars.sp_radius = self.sp_center, self.sp_radius
        self.darkmatter.sp_center, self.darkmatter.sp_radius = self.sp_center, self.sp_radius
        self.gas.sp_center, self.gas.sp_radius = self.sp_center, self.sp_radius

    def _update_data(self):
        """Updates dataset after adding fields.
        """
        self._all_data = self._ds.all_data()
        self.update_shared_attr("halo", "data_source", self._ds.sphere(self.sp_center, self.sp_radius))

    def _compute_grav_stars(self, **kwargs):
        """Adds gravitational potential for stars.
        """
        theta = kwargs.get("theta", 0.7)
        parallel = kwargs.get("parallel", True)
        quadrupole = kwargs.get("quadrupole", True)
    
        def _grav_function(field, data):
            from pytreegrav import PotentialTarget
            global parallel
            global quadrupole
            global theta
            
            return data["stars", "mass"] * unyt_array(
                PotentialTarget(
                    pos_source=np.concatenate(( data["gas", "coordinates"], data["stars", "coordinates"], data["darkmatter", "coordinates"] )).to("kpc"), 
                    pos_target=data["stars", "coordinates"].to("kpc"), 
                    m_source=np.concatenate(( data["gas", "mass"], data["stars", "mass"], data["darkmatter", "mass"] )).to("Msun"), 
                    softening_target=data["stars", "softening"].to("kpc"),
                    softening_source=np.concatenate(( data["gas", "softening"], data["stars", "softening"], data["darkmatter", "softening"] )).to("kpc"), 
                    G=4.300917270038e-06,
                    theta=0.6,
                    parallel=True,
                    quadrupole=True
                ), 
                "km**2/s**2"
            )
        
        self._ds.add_field(
            ("stars", "grav_potential"),
            function=_grav_function,
            sampling_type="local",
            units="Msun*km**2/s**2",
            force_override=True
        )

    def _compute_grav_gas(self, **kwargs):
        """Adds gravitational potential for stars.
        """
        theta = kwargs.get("theta", 0.7)
        parallel = kwargs.get("parallel", True)
        quadrupole = kwargs.get("quadrupole", True)
    
        def _grav_function(field, data):
            from pytreegrav import PotentialTarget
            global parallel
            global quadrupole
            global theta
            
            return data["gas", "mass"] * unyt_array(
                PotentialTarget(
                    pos_source=np.concatenate(( data["gas", "coordinates"], data["stars", "coordinates"], data["darkmatter", "coordinates"] )).to("kpc"), 
                    pos_target=data["gas", "coordinates"].to("kpc"), 
                    m_source=np.concatenate(( data["gas", "mass"], data["stars", "mass"], data["darkmatter", "mass"] )).to("Msun"), 
                    softening_target=data["gas", "softening"].to("kpc"),
                    softening_source=np.concatenate(( data["gas", "softening"], data["stars", "softening"], data["darkmatter", "softening"] )).to("kpc"), 
                    G=4.300917270038e-06,
                    theta=0.6,
                    parallel=True,
                    quadrupole=True
                ), 
                "km**2/s**2"
            )
        
        self._ds.add_field(
            ("gas", "grav_potential"),
            function=_grav_function,
            sampling_type="local",
            units="Msun*km**2/s**2",
            force_override=True
        )

    def _compute_grav_dm(self, **kwargs):
        """Adds gravitational potential for stars.
        """
        theta = kwargs.get("theta", 0.7)
        parallel = kwargs.get("parallel", True)
        quadrupole = kwargs.get("quadrupole", True)
    
        def _grav_function(field, data):
            from pytreegrav import PotentialTarget
            global parallel
            global quadrupole
            global theta
            
            return data["darkmatter", "mass"] * unyt_array(
                PotentialTarget(
                    pos_source=np.concatenate(( data["gas", "coordinates"], data["stars", "coordinates"], data["darkmatter", "coordinates"] )).to("kpc"), 
                    pos_target=data["darkmatter", "coordinates"].to("kpc"), 
                    m_source=np.concatenate(( data["gas", "mass"], data["stars", "mass"], data["darkmatter", "mass"] )).to("Msun"), 
                    softening_target=data["darkmatter", "softening"].to("kpc"),
                    softening_source=np.concatenate(( data["gas", "softening"], data["stars", "softening"], data["darkmatter", "softening"] )).to("kpc"), 
                    G=4.300917270038e-06,
                    theta=0.6,
                    parallel=True,
                    quadrupole=True
                ), 
                "km**2/s**2"
            )
        
        self._ds.add_field(
            ("darkmatter", "grav_potential"),
            function=_grav_function,
            sampling_type="local",
            units="Msun*km**2/s**2",
            force_override=True
        )

    def _compute_total_E(self):
        """Adds total energy field
        """
        self._ds.add_field(
            ("stars", "total_energy"),
            function=lambda field, data: data["stars", "grav_potential"] + data["stars", "kinetic_energy"],
            sampling_type="local",
            units="Msun*km**2/s**2",
            force_override=True
        )
        self._ds.add_field(
            ("gas", "total_energy"),
            function=lambda field, data: data["gas", "grav_potential"] + data["gas", "kinetic_energy"] + data["gas", "thermal_energy"],
            sampling_type="local",
            units="Msun*km**2/s**2",
            force_override=True
        )
        self._ds.add_field(
            ("darkmatter", "total_energy"),
            function=lambda field, data: data["darkmatter", "grav_potential"] + data["darkmatter", "kinetic_energy"],
            sampling_type="local",
            units="Msun*km**2/s**2",
            force_override=True
        )        
        self._update_data()

    def info(self, get_str=False):
        """Prints information about the loaded halo: information about the position of the halo in the simulation
        and each component; dark matter, stars and gas: CoM, v_CoM, r1/2, sigma*, and properties of the whole halo such as
        the Dynamical Mass, half-light radius etc.

        Parameters
        ----------
        get_str : bool, optional
            Whether to print the information or to return a string.

        Returns
        -------
        info : str
        """
        output = []
        
        output.append(f"\ngeneral information")
        output.append(f"{'':-<21}")
        output.append(f"{'snapshot path':<20}: {'/'.join(self.fn.split('/')[:-1])}")
        output.append(f"{'snapshot file':<20}: {self.fn.split('/')[-1]}")
        output.append(f"{'redshift':<20}: {self.redshift:.4f}")
        output.append(f"{'scale_factor':<20}: {self.scale_factor:.4f}")
        output.append(f"{'age':<20}: {self.time:.4f}")
        output.append(f"{'cut-out center':<20}: [{self.sp_center[0].value:.2f}, {self.sp_center[1].value:.2f}, {self.sp_center[2].value:.2f}]  {self.sp_center.units}")
        output.append(f"{'cut-out radius':<20}: {self.sp_radius:.2f}")
        output.append(f"{'dm':<20}: {self.darkmatter.masses.sum():.3e}")
        output.append(f"{'stars':<20}: {self.stars.masses.sum():.3e}")
        output.append(f"{'gas':<20}: {self.gas.masses.sum():.3e}")
        output.append(f"{'Mdyn':<20}: {None if self.Mhl is None else f'{self.Mhl:.3e}'}")


        output.append(f"\ncoordinate basis")
        output.append(f"{'':-<21}")
        output.append(f"{'type':<20}: orthonormal")
        output.append(f"{'line-of-sight':<20}: {self.los}")
        output.append(f"{'u1':<20}: [{self.basis[0,0]:.2f}, {self.basis[1,0]:.2f}, {self.basis[2,0]:.2f}]")
        output.append(f"{'u2':<20}: [{self.basis[0,1]:.2f}, {self.basis[1,1]:.2f}, {self.basis[2,1]:.2f}]")
        output.append(f"{'u3':<20}: [{self.basis[0,2]:.2f}, {self.basis[1,2]:.2f}, {self.basis[2,2]:.2f}]")

        str_main = "\n".join(output)
        str_stars = self.stars.info(get_str=True)
        str_darkmatter = self.darkmatter.info(get_str=True)
        str_gas = self.gas.info(get_str=True)

        str_info = "\n".join([str_main, str_stars, str_darkmatter, str_gas])
        if get_str:
            return str_info
        else:
            print(str_info)
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
        
        Changes basis and los instances in the class instance and all its childs. All vectorial quantities get expressed in the new
        coordinate basis.
        """
        self._set_los(los)            
        self.stars.set_line_of_sight(los)
        self.darkmatter.set_line_of_sight(los)
        self.gas.set_line_of_sight(los)

        for q in ["cm", "vcm", "rockstar_velocity", "rockstar_center"]:
            self.update_shared_attr(
                "halo",
                q,
                self._old_to_new_base @ self.q[q] if self.q[q] is not None else None
            )       

    def enclosed_mass(self, center, radius, components):
        """Computes the dynamical mass: the mass enclonsed inside the 3D half light radius of stars. Right now, the half-mass-radius is
        used, under the assumptionn that M/L does not vary much from star-particle to star-particle.

        Parameters
        ----------
        components : list[str]
            Components to use: stars, gas and/or darkmatter.

        Returns
        -------
        mdyn : float
        """
        if components == "all": components=["stars", "darkmatter", "gas"]
        if components == "particles": components=["stars", "darkmatter"]

        mgas, mstars, mdm = unyt_quantity(0, "Msun"), unyt_quantity(0, "Msun"), unyt_quantity(0, "Msun") 
        
        if "stars" in components:
            mstars = self.stars.enclosed_mass(radius, center)
        if "gas" in components:
            mgas = self.gas.enclosed_mass(radius, center)
        if "darkmatter" in components:
            mdm = self.darkmatter.enclosed_mass(radius, center)

        return  mstars + mdm + mgas

    def compute_gravitational_energy(self, **kwargs):
        """Adds gravitational energy to stars, dm and gas. Fields are computed on demand by yt.
        """
        self._compute_grav_dm(**kwargs)
        self._compute_grav_stars(**kwargs)
        self._compute_grav_gas(**kwargs)
        self._update_data()
   
    def compute_kinetic_energy(self, v_cm=None, **kwargs):
        """Adds kinetic energy to yt dataset.
        """
        if v_cm is None:
            _, v_cm = self.darkmatter.refined_center6d(method=kwargs.get("method", "adaptative"), **kwargs)
        
        self._ds.add_field(
            ("stars", "kinetic_energy"),
            function=lambda field, data: 0.5 * data["stars", "mass"] * np.linalg.norm(data["stars","velocity"] - v_cm, axis=1) ** 2,
            sampling_type="local",
            units="Msun*km**2/s**2",
            force_override=True
        )
        self._ds.add_field(
            ("gas", "kinetic_energy"),
            function=lambda field, data: 0.5 * data["gas", "mass"] * np.linalg.norm(data["gas","velocity"] - v_cm, axis=1) ** 2,
            sampling_type="local",
            units="Msun*km**2/s**2",
            force_override=True
        )
        self._ds.add_field(
            ("darkmatter", "kinetic_energy"),
            function=lambda field, data: 0.5 * data["darkmatter", "mass"] * np.linalg.norm(data["darkmatter","velocity"] - v_cm, axis=1) ** 2,
            sampling_type="local",
            units="Msun*km**2/s**2",
            force_override=True
        )        
        self._update_data()
   
    def compute_energy(self,  
                       cm_subset=["darkmatter"],
                       cm_weighting="softmax",
                       refine=False,
                       verbose=False,
                       **kwargs
                      ):  
        """Computes total energy of particles, taking into account gas thermal energy. Since the process is dependent of v_cm,
        a refinement process can be carried out to converge on v_cm.
        """
        T = kwargs.get("T", "adaptative")
        f = kwargs.get("f", 0.1)
        nbound = kwargs.get("nbound", 32)
        delta = kwargs.get("delta", 1E-3)

        self.compute_gravitational_energy(**kwargs)
        vcm = self.darkmatter.q["vcm"]
        cm = self.darkmatter.q["cm"]

        for i in range(100):
            self.compute_kinetic_energy(v_cm=vcm)
            self._compute_total_E()
            E = self.darkmatter["total_energy"]
            
            
            if np.all(E >= 0):
                raise Exception("All the particles are unbound in the halo!")
            else:
                bound_subset_mask = E < 0
                
            if not refine:
                break
                
            if cm_weighting.lower() == "most-bound":
                N = int(np.rint(np.minimum(f * np.count_nonzero(bound_subset_mask), nbound)))
                most_bound_mask = create_subset_mask(E, np.ones_like(E, dtype=bool), N)
                
                new_cm = np.average(
                    self.darkmatter["coordiantes"][most_bound_mask], 
                    axis=0, 
                    weights=self.darkmatter["mass"][most_bound_mask]
                )
                new_vcm = np.average(
                    self.darkmatter["velocity"][most_bound_mask], 
                    axis=0, 
                    weights=self.darkmatter["mass"][most_bound_mask]
                )  
            elif cm_weighting.lower() == "softmax":                
                w = E[bound_subset_mask]/E[bound_subset_mask].min()
                
                if T == "adaptative":
                    T = np.abs(self.darkmatter["kinetic_energy"][bound_subset_mask].mean()/E[bound_subset_mask].min())
                    
                new_cm = np.average(
                    self.darkmatter["coordinates"][bound_subset_mask], 
                    axis=0, 
                    weights=softmax(w, T)
                )
                new_vcm = np.average(
                    self.darkmatter["velocity"][bound_subset_mask], 
                    axis=0, 
                    weights=softmax(w, T)
                )               
            else:
                raise ValueError("Weighting mode doesnt exist!")

            delta_cm = np.sqrt(np.sum((new_cm - cm)**2, axis=0)) / np.linalg.norm(cm) < delta
            delta_vcm =  np.sqrt(np.sum((new_vcm - vcm)**2, axis=0)) / np.linalg.norm(vcm) < delta        
            
            if (delta_cm and delta_vcm):
                self.update_shared_attr("halo", "cm", new_cm)
                self.update_shared_attr("halo", "vcm", new_vcm)
                self.compute_kinetic_energy(v_cm=new_vcm)
                self._compute_total_E()

                break
            cm, vcm = new_cm, new_vcm

    def get_bound_halo(self):
        """Returns a SnapshotHalo instance where only bound particles are present. Only usable after running compute_energies and compute_bound_stars.
        """
        yt.add_particle_filter(
            "stars_bound", 
            function=lambda pfilter, data: data[pfilter.filtered_type, "total_energy"] < 0, 
            filtered_type="stars", 
            requires=["total_energy"]

        )
        yt.add_particle_filter(
            "gas_bound", 
            function=lambda pfilter, data: data[pfilter.filtered_type, "total_energy"] < 0, 
            filtered_type="gas", 
            requires=["total_energy"]

        )
        yt.add_particle_filter(
            "darkmatter_bound", 
            function=lambda pfilter, data: data[pfilter.filtered_type, "total_energy"] < 0, 
            filtered_type="darkmatter", 
            requires=["total_energy"]

        )
        self._ds.add_particle_filter("stars_bound")
        self._ds.add_particle_filter("gas_bound")
        self._ds.add_particle_filter("darkmatter_bound")
        self._update_data()
        return None
            
    def plot(self, normal="z", catalogue=None, smooth_particles = False, **kwargs):
        """Plots the projected darkmatter, stars and gas distributions along a given normal direction. The projection direction can be changed with
        the "normal" argument. If catalogue is provided, halos of massgreater than 5E7 will be displayed on top of the darkmatter plot.

        OPTIONAL Parameters
        ----------
        normal : str or 
            Projection line of sight, either "x", "y" or "z".
        catalogue : pd.DataFrame
            Halo catalogue created using galex.extractor.mergertrees.

        Returns
        -------
        fig : matplotlib figure object
        """
        import cmyt
        from matplotlib import gridspec
        import matplotlib.pyplot as plt
        from mpl_toolkits.axes_grid1 import AxesGrid, make_axes_locatable
        from matplotlib.patches import Circle, Rectangle
        from mpl_toolkits.axes_grid1.inset_locator import mark_inset
        
        try:
            import smplotlib
        except:
            pass

        plt.rcParams['axes.linewidth'] = 1.2
        plt.rcParams['xtick.major.width'] = 1.1
        plt.rcParams['xtick.minor.width'] = 1.1
        plt.rcParams['ytick.major.width'] = 1.1
        plt.rcParams['ytick.minor.width'] = 1.1
        
        plt.rcParams['xtick.major.size'] = 7 * 1.5
        plt.rcParams['ytick.major.size'] = 7 * 1.5
        
        plt.rcParams['xtick.minor.size'] = 5 
        plt.rcParams['ytick.minor.size'] = 5
        
        ds = self._ds
        center = self.arr(self.sp_center)
        radius = self.quant(self.sp_radius)

        source_factor = 1.5 if "source_factor" not in kwargs.keys() else kwargs["source_factor"]
        draw_inset = True if "draw_inset" not in kwargs.keys() else kwargs["draw_inset"]
        
        if "zoom_factors_rvir" in kwargs.keys():
            zooms = np.sort(kwargs["zoom_factors_rvir"])[::-1][[0,1]]
            plot_radii = radius * zooms * 2
        else:
            plot_radii = radius * np.array([1 * 2])

        sp_source = ds.sphere(center, source_factor*plot_radii.max())
        
        plots = 0
        if not self.darkmatter.empty: plots += 1
        if not self.stars.empty: plots += 1
        if not self.gas.empty: plots += 1

        if plots == 0:
            raise ValueError(f"It Seems that all components are empty!")

        if normal=="x" or normal==[1,0,0]: cindex = [1,2]
        elif normal=="y" or normal==[0,1,0]: cindex = [2,0]
        elif normal=="z" or normal==[0,0,1]: cindex = [0,1]
        else: raise ValueError(f"Normal must along main axes. You provided {normal}.")


        if smooth_particles:
            ds_dm = create_sph_dataset(
                ds,
                self.ptypes["darkmatter"],
                data_source=sp_source,
                n_neighbours=100 if "n_neighbours" not in kwargs.keys() else kwargs["n_neighbours"][0] if isinstance(kwargs["n_neighbours"], list) else kwargs["n_neighbours"],
                kernel="wendland2" if "kernel" not in kwargs.keys() else kwargs["kernel"],
            )
            dm_type = ("io", "density") 

            ds_st = create_sph_dataset(
                ds,
                self.ptypes["stars"],
                data_source=sp_source,
                n_neighbours=100 if "n_neighbours" not in kwargs.keys() else kwargs["n_neighbours"][1] if isinstance(kwargs["n_neighbours"], list) else kwargs["n_neighbours"],
                kernel="wendland2" if "kernel" not in kwargs.keys() else kwargs["kernel"],
            )
            star_type = ("io", "density")

            sp_source_dm = ds_dm.sphere(center, source_factor*plot_radii.max())
            sp_source_st = ds_st.sphere(center, source_factor*plot_radii.max())
        else:
            dm_type = (self.ptypes["darkmatter"], "mass")  
            star_type = (self.ptypes["stars"], "mass")            

        gas_type = (self.ptypes["gas"], "density")        
            
        if "cm" not in kwargs:
            cmyt.arbre.set_bad(cmyt.arbre.get_under())
            cm = cmyt.arbre
        else:
            cm = kwargs["cm"]

            
        fig, axes = plt.subplots(
            len(plot_radii),
            plots,
            figsize=(6*plots*1.2*1.05, 6*len(plot_radii)*1.2),
            sharey="row",
            constrained_layout=False
        )
        grid = axes.flatten()

        stars_cen = self.basis @ self.stars.q["cm"]
        dm_cen = self.basis @ self.darkmatter.q["cm"]
        gas_cen = self.basis @ self.gas.q["cm"]
        plot_centers = [center, 0.5 * (stars_cen + dm_cen)]
        
        
        ip = 0
        for jp, (pcenter, plot_radius) in enumerate(zip(plot_centers, plot_radii)):
            center_dist =  (pcenter - plot_centers[0])[cindex]

            ext = (-plot_radius.to("kpc")/2 + center_dist[0], 
                   plot_radius.to("kpc")/2 + center_dist[0], 
                   -plot_radius.to("kpc")/2 + center_dist[1], 
                   plot_radius.to("kpc")/2 + center_dist[1]
            )



            tmp_stars_cen = (stars_cen - pcenter)[cindex] + center_dist
            tmp_dm_cen = (dm_cen - pcenter)[cindex] + center_dist
            tmp_gas_cen = (gas_cen - pcenter)[cindex] + center_dist
            tmp_cen = (center - pcenter)[cindex] + center_dist
            
            if not self.darkmatter.empty: 
                if smooth_particles:            
                    p = yt.ProjectionPlot(ds_dm, normal, dm_type, center=pcenter, width=plot_radius, data_source=sp_source_dm)
                    p.set_unit(dm_type, "Msun/kpc**2")
                    frb = p.data_source.to_frb(plot_radius, 800)
    
                else:
                    p = yt.ParticleProjectionPlot(ds, normal, dm_type, center=pcenter, width=plot_radius, density=True, data_source=sp_source, deposition="ngp" if "deposition" not in kwargs.keys() else kwargs["deposition"])
                    p.set_unit(dm_type, "Msun/kpc**2")
                    frb = p.frb
    
    
                ax = grid[ip]
                data = frb[dm_type]
                data[data == 0] = data[data != 0 ].min()
                pc_dm = ax.imshow(data.to("Msun/kpc**2"), cmap=cm, norm="log", extent=ext)
                divider = make_axes_locatable(ax)
                cax = divider.append_axes("right", size="5%", pad=0)
                cbar_dm = plt.colorbar(pc_dm, cax=cax)
                cbar_dm.set_label(r'Projected Dark Matter Density $[Msun/kpc^2]$', fontsize=20)
                cbar_dm.ax.tick_params(labelsize=25)

                
                ax.scatter(*tmp_cen, color="black", marker="1", s=370, zorder=20)
                if jp==0:
                    rvir_circ = Circle((0,0), radius.to("kpc").value, facecolor="none", edgecolor="black")
                    ax.add_patch(rvir_circ)
                    ax.scatter(*tmp_stars_cen, color="red", marker="*", s=150)
                    ax.scatter(*tmp_dm_cen, color="black", marker="+", s=300)
                if jp==1:
                    ax.scatter(*tmp_stars_cen, color="red", marker="*", s=300)
                    ax.scatter(*tmp_dm_cen, color="black", marker="+", s=370)
                    ax.scatter(*tmp_gas_cen, color="orange", marker="o")
                    rhf_circ = Circle(tmp_stars_cen.to("kpc").value, self.stars.half_mass_radius()[0].to("kpc").value, facecolor="none", edgecolor="red")
                    ax.add_patch(rhf_circ)
                
                ax.set_aspect('equal')
                ax.tick_params(axis='both', which='both', labelbottom=True, labelsize=20)    
                if ip < plots:
                    ax.set_title(kwargs["titles"][0] if "titles" in kwargs.keys() else None, fontsize=25)

                ax.set_xlim(ext[0], ext[1])
                ax.set_ylim(ext[2], ext[3])
                ip += 1
    
            
            if not self.stars.empty:
                if smooth_particles: 
                    p = yt.ProjectionPlot(ds_st, normal, star_type, center=pcenter, width=plot_radius, data_source=sp_source_st)
                    p.set_unit(star_type, "Msun/kpc**2")
                    frb = p.data_source.to_frb(plot_radius, 800)

                else:
                    p = yt.ParticleProjectionPlot(ds, normal, star_type, center=pcenter, width=plot_radius, density=True, data_source=sp_source, deposition="ngp" if "deposition" not in kwargs.keys() else kwargs["deposition"])
                    p.set_unit(star_type, "Msun/kpc**2")    
                    frb = p.frb

    
                ax = grid[ip]
                data = frb[star_type]
                data[data == 0] = data[data != 0 ].min()
                pc_st = ax.imshow(data.to("Msun/kpc**2"), cmap=cm, norm="log", vmin=data.to("Msun/kpc**2").max().value/1E4, extent=ext)
                divider = make_axes_locatable(ax)
                cax = divider.append_axes("right", size="5%", pad=0)
                cbar_st = fig.colorbar(pc_st, cax=cax)
                cbar_st.set_label(r'Projected Stellar Density $[Msun/kpc^2]$', fontsize=22)
                cbar_st.ax.tick_params(labelsize=25)

                ax.scatter(*tmp_cen, color="black", marker="1", s=370, zorder=20)
                if jp==0:
                    rvir_circ = Circle((0,0), radius.to("kpc").value, facecolor="none", edgecolor="black")
                    ax.add_patch(rvir_circ)
                    ax.scatter(*tmp_stars_cen, color="red", marker="*", s=150)
                    ax.scatter(*tmp_dm_cen, color="black", marker="+", s=300)
                if jp==1:
                    ax.scatter(*tmp_stars_cen, color="red", marker="*", s=300)
                    ax.scatter(*tmp_dm_cen, color="black", marker="+", s=370)
                    ax.scatter(*tmp_gas_cen, color="orange", marker="o")
                    rhf_circ = Circle(tmp_stars_cen.to("kpc").value, self.stars.half_mass_radius()[0].to("kpc").value, facecolor="none", edgecolor="red")
                    ax.add_patch(rhf_circ)


                
                ax.set_aspect('equal')
                ax.tick_params(axis='both', which='both', labelbottom=True, labelsize=20)
                if ip < plots:
                    ax.set_title(kwargs["titles"][1] if "titles" in kwargs.keys() else None, fontsize=25)

                ax.set_xlim(ext[0], ext[1])
                ax.set_ylim(ext[2], ext[3])               
                ip += 1
    
            
            if not self.gas.empty: 
                p = yt.ProjectionPlot(ds, normal, gas_type, center=pcenter, width=plot_radius, data_source=sp_source)
                p.set_unit(gas_type, "Msun/kpc**2")
                frb = p.data_source.to_frb(plot_radius, 800)
                
                ax = grid[ip]
                density_data = frb[gas_type]
                density_data[density_data == 0] = density_data[density_data != 0 ].min()
                p_gas = ax.imshow(density_data.to("Msun/kpc**2"), cmap=cm, norm="log", extent=ext)
                divider = make_axes_locatable(ax)
                cax = divider.append_axes("right", size="5%", pad=0)
                cbar_gas = fig.colorbar(p_gas, cax=cax)
                cbar_gas.set_label(r'Projected Gas Density $[Msun/kpc^2]$', fontsize=22)

                ax.scatter(*tmp_cen, color="black", marker="1", s=370, zorder=20)
                if jp==0:
                    rvir_circ = Circle((0,0), radius.to("kpc").value, facecolor="none", edgecolor="black")
                    ax.add_patch(rvir_circ)
                    ax.scatter(*tmp_stars_cen, color="red", marker="*", s=150)
                    ax.scatter(*tmp_dm_cen, color="black", marker="+", s=300)
                    ax.scatter(*tmp_gas_cen, color="orange", marker="o")
                if jp==1:
                    ax.scatter(*tmp_stars_cen, color="red", marker="*", s=300)
                    ax.scatter(*tmp_dm_cen, color="black", marker="+", s=370)
                    ax.scatter(*tmp_gas_cen, color="orange", marker="o")
                    rhf_circ = Circle(tmp_stars_cen.to("kpc").value, self.stars.half_mass_radius()[0].to("kpc").value, facecolor="none", edgecolor="red")
                    ax.add_patch(rhf_circ)

                
                cbar_gas.ax.tick_params(labelsize=25)
                ax.set_aspect('equal')
                ax.tick_params(axis='both', which='both', labelbottom=True, labelsize=20)
                if ip < plots:
                    ax.set_title(kwargs["titles"][2] if "titles" in kwargs.keys() else None, fontsize=25)

                ax.set_xlim(ext[0], ext[1])
                ax.set_ylim(ext[2], ext[3])
                ip += 1

        c = "white" if smooth_particles else "darkgreen"
        low_m = 9E7 if "low_mass" not in kwargs.keys() else kwargs["low_mass"]
        high_m = np.inf if "high_mass" not in kwargs.keys() else kwargs["high_mass"]
        annotation_style = "circle" if "annotation_style" not in kwargs.keys() else kwargs["annotation_style"]
        if catalogue is not None:
            dist = np.linalg.norm(ds.arr(catalogue[['position_x', 'position_y', 'position_z']].values, 'kpccm') - center.to("kpccm"), axis=1)
            filtered_halos = catalogue[
                (dist < plot_radii.max()) & 
                (catalogue['mass'] > low_m) & 
                (catalogue['mass'] < high_m) & 
                (dist > 0.1)
            ]
            for i in range(0, len(filtered_halos)):
                sub_tree_id = filtered_halos['Sub_tree_id'].iloc[i]
                halo_pos = ds.arr(filtered_halos.iloc[i][['position_x', 'position_y', 'position_z']].values, 'kpccm').to('kpc') - center
                virial_radius = ds.quan(filtered_halos.iloc[i]['virial_radius'], 'kpccm').to('kpc')

                if annotation_style == "circle" or annotation_style == "all":
                    extra_halo = Circle(halo_pos[cindex], 0.5*virial_radius, facecolor="none", edgecolor=c)
                    axes[0, 0].add_patch(extra_halo)
                    
                if annotation_style == "center" or annotation_style == "all":
                    axes[0, 0].scatter(*halo_pos[cindex], marker="v", edgecolor=c, s=90, color="none")

                if annotation_style == "all":
                    axes[0, 0].text(halo_pos[cindex][0], halo_pos[cindex][1] - 0.033*virial_radius, int(sub_tree_id), fontsize=14, ha="center", va="top", color=c)    

            

        for ax in axes[-1,:]:
            ax.set_xlabel('x [kpc]', fontsize=20)

        for ax in axes[:, 0]:
            ax.set_ylabel('y [kpc]', fontsize=20)

        if len(plot_radii) != 1 and draw_inset:
            mark_inset(
                axes[0, -1], 
                axes[-1, -1], 
                loc1=1, loc2=2, 
                facecolor="none",
                edgecolor="black"
            )
    
            plt.subplots_adjust(
                hspace=-0.45,
                wspace=0.1
            )
        else:
            plt.subplots_adjust(
                hspace=0.3,
                wspace=0.1
            )
        plt.tight_layout()
        plt.close()
        return fig     
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    

    
    
    
    
    
    
    
    
    
    











