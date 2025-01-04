import os
import yt
import shutil
import numpy as np
from unyt import unyt_array, unyt_quantity

from .config import config
from .base import BaseSimulationObject
from .particle_type import StellarComponent, DarkComponent, GasComponent
from .class_methods import bound_particlesBH, bound_particlesAPROX, density_profile, velocity_profile, create_sph_dataset, gram_schmidt

class SnapshotHalo(BaseSimulationObject):
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


    



    SOME OF THESE ARGUMENTS WILL BE INHERITED FROM A BIGGER CLASS AT SOME POINT.
    """
    def __init__(self,
                 fn,
                 center,
                 radius,
                 dataset=None,
                 **kwargs
                ):
        """Initialize function.
        """
        super().__init__()

        ### File initialization information ###
        self.fn = fn
        self.sp_center = unyt_array(center[0], center[1])
        self.sp_radius = unyt_quantity(radius[0], radius[1])        
        self.bound_method = None

        self._kwargs = kwargs
        self._fields_loaded = {}
        self._dynamic_fields = {
            "all_coords" : "coords",
            "all_vels" : "vels",
            "all_masses" : "masses",
            "particle_type" : "particle_type"
        }
        
        self.parse_dataset(dataset)
        self._cm, self._vcm = None, None

    
    
    @property
    def time(self):
        """Cosmic time
        """
        if self._time is not None:
            return self._time.in_units(f"{self.units['time']}")
        else:
            return None
    @property  
    def redshift(self):
        """Redshift
        """
        return self._redshift
    @property  
    def scale_factor(self):
        """scale_factor
        """
        return self._scale_factor
    @property  
    def hubble_constant(self):
        """hubble_constant
        """
        return self._hubble_constant
    @property  
    def omega_matter(self):
        """omega_matter
        """
        return self._omega_matter
    @property  
    def omega_lambda(self):
        """omega_lambda
        """
        return self._omega_lambda
    @property  
    def omega_radiation(self):
        """omega_radiation
        """
        return self._omega_radiation
    @property  
    def omega(self):
        """omega
        """
        return self._omega
    

    @property
    def cm(self):
        return self._cm.in_units(self.units['length'])
    @cm.setter
    def cm(self, value):
        if not isinstance(value, unyt_array):
            raise ValueError("vcm must be a number unyt_array.")
        if str(value.units.dimensions) != '(length)':
            raise ValueError(f"Your units are not correct: {str(value.units.dimensions)} != length")
        
        self._cm = value
    @property
    def vcm(self):
        return self._vcm.in_units(self.units['velocity'])
    @vcm.setter
    def vcm(self, value):
        if not isinstance(value, unyt_array):
            raise ValueError("vcm must be a number unyt_array.")
        if str(value.units.dimensions) != '(length)/(time)':
            raise ValueError(f"Your units are not correct: {str(value.units.dimensions)} != velocity")
        
        self._vcm = value

    @property
    def Mhl(self):
        self.stars.half_mass_radius()
        return self.enclosed_mass(
            self.stars.cm,
            self.stars.rh3d,
            ["stars", "darkmatter", "gas"]
        )
    @Mhl.setter
    def Mhl(self, value):
        if not isinstance(value, unyt_quantity):
            raise ValueError("vcm must be a number unyt_array.")
        if str(value.units.dimensions) != '(mass)':
            raise ValueError(f"Your units are not correct: {str(value.units.dimensions)} != mass")
        
        self._Mhl = value

    

    
    def _update_kwargs(self, default_kw, new_kw):
        """Update default kwargs with user provided kwargs.
        """
        for key, value in new_kw.items():
            if isinstance(value, dict) and key in default_kw and isinstance(default_kw[key], dict):
                self._update_kwargs(default_kw[key], value)
            else:
                default_kw[key] = value

        self._kwargs = default_kw
        return None
   
    
    
    
        
            
    def info(self, get_str = False):
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

        output.append(f"\nunits")
        output.append(f"{'':-<21}")
        output.append(f"{'length_unit':<20}: {self.units['length']}")
        output.append(f"{'velocity_unit':<20}: {self.units['velocity']}")
        output.append(f"{'mass_unit':<20}: {self.units['mass']}")
        output.append(f"{'time_unit':<20}: {self.units['time']}")
        output.append(f"{'comoving':<20}: {self.units['comoving']}")

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

    def load_dataset(self):
        """Calls the module-level loader function to load the dataset. The module-level loader can be set as
        import pkg; pkg.loader = loader_function. Afterwards, the whole module has access to it. The function must be
        user written (it can use anything; h5py, yt, pynbody, ...), this is done to allow for more flexibillity when
        loading the data, as the I/O sections of the analysis pipelines are usually time consuming.

        The function must accept a file-name/path and return, in an XXX structure the following quantities, at minimum:
        
        .....
        .....
        .....

        Additional quantities can also be passed and will be loaded with the rest (e.g. element fractions for gas, 
        luminosities for stars, etc.).

        Afterwards, only the particles that are part of the halo are selected.
        
        Returns
        -------
        dataset : returned object is of the type returned by (user) specified config.loader
        """

        assert self.loader, "The module-level loader is not set! You can set it up as: import pkg; pkg.config.loader = loader_function before you start your scripts or use the default one.\nBeware of the requirements that this loader must fulfill!"
    
        return self.loader(self.fn, config.code)
        
        
    def parse_dataset(self, dataset):
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
        assert self.parser, "The module-level loader is not set! You can set it up as: import pkg; pkg.config.parser = parser_function before you start your scripts or use the default one.\nBeware of the requirements that this parser must fulfill!"
            
        base_units, metadata, hashable_data = self.parser(self.load_dataset() if dataset is None else dataset, 
                                                            self.sp_center, self.sp_radius
                                                           )    
        self._data = hashable_data
        self._metadata = metadata
        for meta, value in metadata.items():
            if meta in ['redshift',
                        'scale_factor',
                        'time',
                        'hubble_constant',
                        'omega_matter',
                        'omega_lambda',
                        'omega_radiation',
                        'omega_curvature',
                        'omega']:
                
                setattr(self, '_'+meta, value)
            else:
                setattr(self, meta, value)

            
        if self.base_units is None:
            self.base_units = base_units
        
        self.stars = StellarComponent(hashable_data, **self._kwargs)
        self.darkmatter = DarkComponent(hashable_data, **self._kwargs)
        self.gas = GasComponent(hashable_data, **self._kwargs)

        self.stars._sp_center, self.stars._sp_radius = self.sp_center, self.sp_radius
        self.darkmatter._sp_center, self.darkmatter._sp_radius = self.sp_center, self.sp_radius
        self.gas._sp_center, self.gas._sp_radius = self.sp_center, self.sp_radius
        return None
        
        

    def set_units(self, new_units):
        """Sets units for zHalo and all particle types it contains.
        """
        self._set_units(new_units)
        self.stars._set_units(new_units)
        self.darkmatter._set_units(new_units)
        self.gas._set_units(new_units)
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

        if self._cm is not None:
            self._cm = self._old_to_new_base @ self._cm
        if self._vcm is not None:
            self._vcm = self._old_to_new_base @ self._vcm
            
        return None

    def enclosed_mass(self, center, radius, components, only_bound=False):
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
            mstars = self.stars.enclosed_mass(radius, center, only_bound=only_bound)
        if "gas" in components:
            mgas = self.gas.enclosed_mass(radius, center, only_bound=only_bound)
        if "darkmatter" in components:
            mdm = self.darkmatter.enclosed_mass(radius, center, only_bound=only_bound)

        return  mstars + mdm + mgas

    def compute_bound_particles(self,
                                method="BH",
                                components=["stars","gas","darkmatter"],
                                cm_subset=["darkmatter"],
                                weighting="softmax",
                                verbose=False,
                                **kwargs
                               ):
        """Computes the kinetic, potential and total energies of the particles in the specified components and determines
        which particles are bound (un-bound) as E<0 (E>=0). Energies are stored as attributes. The gravitational potential
        canbe calculated in two distinct ways:

                1) Barnes-Hut tree-code implemented in pytreegrav, with allowance for force softening.
                2) By approximating the potential as for a particle of mass "m" located at "r" as:
                                         pot(r) = -G* M(<r) * m / |r - r_cm|
                   this requires knowledge of the center-of-mass position to a good degree.

        Given that the initial center-of-mass position and velocity might be relativelly unknown (as the whole particle
        distribution is usually not a good estimator for these), the posibility of performing a refinement exist, where several
        iterationsa performed until the center-of-mass position and velocity converge `both` to delta. The enter-of-mass 
        position and velocity can be calculated using the N-most bound particles of using softmax weighting.

        OPTIONAL Parameters
        ----------
        method : str
            Method of computation. Either BH or APROX. Default: BH
        weighting : str
            SOFTMAX or MOST-BOUND. Names are self, explanatory. Default: softmax.
        components : list[str]
            Components to use: stars, gas and/or darkmatter. Default: all
        verbose : bool
            Verbose. Default: False

        KEYWORD ARGUMENTS
        -----------------
        cm, vcm : array
            Optional initial center-of-mass position and velocity.
        softenings : list[tuple[float, str]] or list[float]
            Softening for each particle type. Same shape as components. Default: [0,0,0]
        theta : float
            Opening angle for BH. Default: 0.7
        refine : bool
            Whether to refine. Default: False
        delta : float
            Converge tolerance for refinement. Default: 1E-5
        nbound : int
        Controls how many particles are used when estimating CoM properties through MOST-BOUND.
        T : int
        Controls how many particles are used when estimating CoM properties through SOFTMAX.
        parallel : bool
            Whether to parallelize BH computation. Default: True
        quadrupole : bool
            Whether to use quardupole approximation istead of dipole. Default: True

        Returns
        -------
        None
        """
        self.stars._delete_bound_fields()
        self.gas._delete_bound_fields()
        self.darkmatter._delete_bound_fields()
        
        if components == "particles":
            components = ["stars", "darkmatter"]
        elif components == "all":
            components = ["stars", "darkmatter", "gas"]

        if cm_subset == "particles":
            cm_subset = ["stars", "darkmatter"]
        elif cm_subset == "all":
            cm_subset = ["stars", "darkmatter", "gas"]
            
        masses = unyt_array(np.empty((0,)), "Msun")
        coords = unyt_array(np.empty((0,3)), "kpc")
        vels = unyt_array(np.empty((0,3)), "km/s")
        softenings = unyt_array(np.empty((0,)), "kpc")
        
        particle_types = np.empty((0,))

        for component in components:
            if getattr(self, component).empty == False:
                getattr(self, component)._delete_bound_fields()
                
                N = len(getattr(self, component).masses)
                masses = np.concatenate((
                    masses, getattr(self, component).masses.to("Msun")
                ))
                coords = np.vstack((
                    coords, getattr(self, component).coords.to("kpc")
                ))
                vels = np.vstack((
                    vels, getattr(self, component).vels.to("km/s")
                ))
                softenings = np.concatenate((
                    softenings, getattr(self, component).softs.to("kpc")
                ))              #unyt_array(np.full(N, psoft.to("kpc")), 'kpc')
                particle_types = np.concatenate((
                    particle_types, np.full(N, component)
                ))

        thermal_energy = unyt_array(np.zeros_like(masses).value, "Msun * km**2/s**2")
        if "gas" in components:
            thermal_energy[particle_types == "gas"] = getattr(self, "gas").thermal_energy.to("Msun * km**2/s**2")

        
        
        particle_subset = np.zeros_like(particle_types, dtype=bool)
        for sub in cm_subset:
            particle_subset = particle_subset | (particle_types == sub)

        if method.lower() == "bh":
            E, kin, pot, cm, vcm = bound_particlesBH(
                coords,
                vels,
                masses,
                softs=softenings,
                extra_kin=thermal_energy,
                cm=None if "cm" not in kwargs else unyt_array(*kwargs['cm']) if isinstance(kwargs['cm'], tuple) and len(kwargs['cm']) == 2 else kwargs['cm'],
                vcm=None if "vcm" not in kwargs else unyt_array(*kwargs['vcm']) if isinstance(kwargs['vcm'], tuple) and len(kwargs['vcm']) == 2 else kwargs['vcm'],
                cm_subset=particle_subset,
                weighting=weighting,
                refine=True if "refine" not in kwargs.keys() else kwargs["refine"],
                delta=1E-5 if "delta" not in kwargs.keys() else kwargs["delta"],
                f=0.1 if "f" not in kwargs.keys() else kwargs["f"],
                nbound=32 if "nbound" not in kwargs.keys() else kwargs["nbound"],
                T=0.22 if "T" not in kwargs.keys() else kwargs["T"],
                return_cm=True,
                verbose=verbose,

            )
        elif method.lower() == "aprox":
            E, kin, pot, cm, vcm = bound_particlesAPROX(
                coords,
                vels,
                masses,
                extra_kin=thermal_energy,
                cm=None if "cm" not in kwargs else unyt_array(*kwargs['cm']) if isinstance(kwargs['cm'], tuple) and len(kwargs['cm']) == 2 else kwargs['cm'],
                vcm=None if "vcm" not in kwargs else unyt_array(*kwargs['vcm']) if isinstance(kwargs['vcm'], tuple) and len(kwargs['vcm']) == 2 else kwargs['vcm'],
                cm_subset=particle_subset,
                weighting=weighting,
                refine=True if "refine" not in kwargs.keys() else kwargs["refine"],
                delta=1E-5 if "delta" not in kwargs.keys() else kwargs["delta"],
                f=0.1 if "f" not in kwargs.keys() else kwargs["f"],
                nbound=32 if "nbound" not in kwargs.keys() else kwargs["nbound"],
                T=0.22 if "T" not in kwargs.keys() else kwargs["T"],
                return_cm=True,
                verbose=verbose,

            )

        
        
        self.cm = cm
        self.vcm = vcm
        
        for component in components:     
            setattr(getattr(self, component), 'E', E[particle_types == component].copy())
            setattr(getattr(self, component), 'kin', kin[particle_types == component].copy())
            setattr(getattr(self, component), 'pot', pot[particle_types == component].copy())
            setattr(getattr(self, component), 'bound_method', f"grav-{method}".lower())

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

        plt.rcParams['axes.linewidth'] = 1.1
        plt.rcParams['xtick.major.width'] = 1.1
        plt.rcParams['xtick.minor.width'] = 1.1
        plt.rcParams['ytick.major.width'] = 1.1
        plt.rcParams['ytick.minor.width'] = 1.1
        
        plt.rcParams['xtick.major.size'] = 7 * 1.5
        plt.rcParams['ytick.major.size'] = 7 * 1.5
        
        plt.rcParams['xtick.minor.size'] = 5 
        plt.rcParams['ytick.minor.size'] = 5
        
        ds = self._data.ds
        center = ds.arr(self.sp_center)
        radius = ds.quan(self.sp_radius)

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

                
        fig, axes = plt.subplots(
            len(plot_radii),
            plots,
            figsize=(6*plots*1.2*1.05, 6*len(plot_radii)*1.2),
            sharey="row",
            constrained_layout=False
        )
        grid = axes.flatten()

        stars_cen = self.stars.cm
        dm_cen = self.darkmatter.cm
        gas_cen = self.gas.cm
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
                    dm_type = (self.ptypes["darkmatter"], self.fields["darkmatter"]["masses"])            
                    p = yt.ParticleProjectionPlot(ds, normal, dm_type, center=pcenter, width=plot_radius, density=True, data_source=sp_source, deposition="ngp" if "deposition" not in kwargs.keys() else kwargs["deposition"])
                    p.set_unit(dm_type, "Msun/kpc**2")
                    frb = p.frb
    
    
                ax = grid[ip]
                data = frb[dm_type]
                pc_dm = ax.imshow(data.to("Msun/kpc**2"), cmap=cmyt.arbre, norm="log", extent=ext)
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
                    rhf_circ = Circle(tmp_stars_cen.to("kpc").value, self.stars.half_mass_radius().to("kpc").value, facecolor="none", edgecolor="red")
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
                    star_type = (self.ptypes["stars"], self.fields["stars"]["masses"])
                    p = yt.ParticleProjectionPlot(ds, normal, star_type, center=pcenter, width=plot_radius, density=True, data_source=sp_source, deposition="ngp" if "deposition" not in kwargs.keys() else kwargs["deposition"])
                    p.set_unit(star_type, "Msun/kpc**2")    
                    frb = p.frb

    
                ax = grid[ip]
                data = frb[star_type]
                pc_st = ax.imshow(data.to("Msun/kpc**2"), cmap=cmyt.arbre, norm="log", extent=ext)
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
                    rhf_circ = Circle(tmp_stars_cen.to("kpc").value, self.stars.half_mass_radius().to("kpc").value, facecolor="none", edgecolor="red")
                    ax.add_patch(rhf_circ)


                
                ax.set_aspect('equal')
                ax.tick_params(axis='both', which='both', labelbottom=True, labelsize=20)
                if ip < plots:
                    ax.set_title(kwargs["titles"][1] if "titles" in kwargs.keys() else None, fontsize=25)

                ax.set_xlim(ext[0], ext[1])
                ax.set_ylim(ext[2], ext[3])               
                ip += 1
    
            
            if not self.gas.empty: 
                gas_type = (self.ptypes["gas"], self.fields["gas"]["dens"])        
                p = yt.ProjectionPlot(ds, normal, gas_type, center=pcenter, width=plot_radius, data_source=sp_source)
                p.set_unit(gas_type, "Msun/kpc**2")
                frb = p.data_source.to_frb(plot_radius, 800)
                
                ax = grid[ip]
                density_data = frb[gas_type]
                p_gas = ax.imshow(density_data.to("Msun/kpc**2"), cmap=cmyt.arbre, norm="log", extent=ext)
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
                    rhf_circ = Circle(tmp_stars_cen.to("kpc").value, self.stars.half_mass_radius().to("kpc").value, facecolor="none", edgecolor="red")
                    ax.add_patch(rhf_circ)

                
                cbar_gas.ax.tick_params(labelsize=25)
                ax.set_aspect('equal')
                ax.tick_params(axis='both', which='both', labelbottom=True, labelsize=20)
                if ip < plots:
                    ax.set_title(kwargs["titles"][2] if "titles" in kwargs.keys() else None, fontsize=25)

                ax.set_xlim(ext[0], ext[1])
                ax.set_ylim(ext[2], ext[3])
                ip += 1


        if catalogue is not None:
            dist = np.linalg.norm(ds.arr(catalogue[['position_x', 'position_y', 'position_z']].values, 'kpccm') - center.to("kpccm"), axis=1)
            filtered_halos = catalogue[(dist < radius.to("kpc")) & (catalogue['mass'] > 9E7) & (dist > 0)]
            for i in range(0, len(filtered_halos)):
                sub_tree_id = filtered_halos['Sub_tree_id'].iloc[i]
                halo_pos = ds.arr(filtered_halos.iloc[i][['position_x', 'position_y', 'position_z']].values, 'kpccm').to('kpc') - center
                virial_radius = ds.quan(filtered_halos.iloc[i]['virial_radius'], 'kpccm').to('kpc')
    
                extra_halo = Circle(halo_pos[cindex], 0.5*virial_radius, facecolor="none", edgecolor="white")
                axes[0, 0].add_patch(extra_halo)
    
            

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
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    

    
    
    
    
    
    
    
    
    
    











