import numpy as np
from scipy.spatial import KDTree
from unyt import unyt_array, unyt_quantity
from copy import copy

from .geometry import Geometry
from .class_methods import (
                            gram_schmidt, 
                            vectorized_base_change, 
                            center_of_mass_pos, 
                            center_of_mass_vel, 
                            refine_6Dcenter, 
                            half_mass_radius, 
                            easy_los_velocity,
                            softmax,
                            density_profile,
                            velocity_profile
                            )


class Component(Geometry):
    """ptype class that contains the particle data, for each particle type present/relevant to your analysis, in the simulation. 
    Fields are stored as attributes and have units thanks to unyt. The class might have as much fields as wished by the user,
    but a few are mandatory:
        
    Mandatory fields
    ----------------
    - coordinates : stored as c
    - velocities : stored as v
    - masses : stored as self["mass"]
    - IDs : stored as self.ids
    """ 
    def __init__(self,
                 tag,
                 data,
                 **kwargs
                 ):
        """Initializes the ptype class.
        """
        super().__init__()

        self.ptype = tag
        
        self._data = data
        self._ds = data.ds
        
        self.arr = self._ds.arr
        self.quant = self._ds.quan
        
        self._dq = {}
        
        if self["mass"].sum() == 0:
            self.empty = True
        else:
            self.empty = False
            
        if self.ptype != "gas":
            self._default_center_of_mass()
            
        
        
    @property
    def dq(self):
        return self._dq
    @dq.setter
    def dq(self, keyval):
        if not isinstance(keyval, dict):
            raise TypeError("Expected a dictionary.")

        self._dq.update(keyval)    

        
    def __getitem__(self, key):
        """Retrieve the value for the given key, dynamically computing it if it's a dynamic field.
        """
        if self.bound and self.ptype != "gas":
            pt = self.ptype + "_bound"
        else:
            pt = self.ptype   
        
        try:
            if key in ["coordinates", "velocity"]:
                return vectorized_base_change(np.linalg.inv(self.basis), self._data[pt, key])
            else:
                return self._data[pt, key]
        except:
            return self.dq[key]
        
    def _update_data(self, dataset, datasource):
        """Updates data after fields have been added
        """
        self._ds = dataset
        self._data = datasource
        
    def info(self, 
             get_str=False
            ):
        """Returns a pretty information summary.
        
        Parameters
        ----------
        get_str : bool
            Return str instead of print. Default: False

        Returns
        -------
        info : str, optionally

        """
        c = self["coordinates"]
        m = self["mass"]
        v = self["velocity"]
        
        
        output = []
        
        output.append(f"\n{self.ptype}")
        output.append(f"{'':-<21}")
        output.append(f"{'len':<20}: {len(c)}")
        output.append(f"{'pos[0]':<20}: [{c[0,0].value:.2f}, {c[0,1].value:.2f}, {c[0,2].value:.2f}] {c.units}")
        output.append(f"{'vel[0]':<20}: [{v[0,0].value:.2f}, {v[0,1].value:.2f}, {v[0,2].value:.2f}] {v.units}")
        output.append(f"{'mass[0]':<20}: {m[0]} {m.units}")
            
        output.append(f"{'cm':<20}: {self.q['cm']}")
        output.append(f"{'vcm':<20}: {self.q['vcm']}")
        output.append(f"{'rh, rh3d':<20}: {self.q['rh']}, {self.q['rh3d']}")
        if self.ptype == "darkmatter":
            output.append(f"{'rs, rvir':<20}: {self.q['rs']}, {self.q['rvir']}")
            output.append(f"{'vmax, sigma':<20}: {self.q['vmax']}, {self.m['sigma']}")
        if self.ptype == "stars":
            output.append(f"{'sigma, sigma_los':<20}: {self.m['sigma']}, {self.m['sigma_los']}")
            output.append(f"{'avg ML':<20}: {self.q['ML']}")
            
        if get_str:
            return "\n".join(output)
        else:
            print("\n".join(output))
            return None
    
    def _default_center_of_mass(self):
        """Computes coarse CoM using all the particles as 

                CoM = sum(mass * pos) / sum(mass)        
        """
        self.empty = True
        tmp_cm, tmp_vcm = None, None
        
        if self["mass"].sum() != 0:
            tmp_cm = center_of_mass_pos(self["coordinates"], self["mass"])
            tmp_vcm = center_of_mass_vel(
                self["coordinates"], 
                self["mass"],
                self["velocity"],
                R=(1E4, "Mpc")
            )            
            self.empty = False
            
        self.dq = {"cm" : tmp_cm}
        self.dq = {"vcm" : tmp_vcm}
        return None

    def _knn_distance(self, center_point, k):
        """Returns the distance from center_point to the k nearest neighbour.

        Parameters
        ----------
        center_point : array
            center point of the knn search.
        k : int
            number of neighbours

        Returns
        -------
        dist : foat
        """   
        if not self.empty:
            if not hasattr(self, "_KDTree"):
                self._KDTree = KDTree(self["coordinates"])        
            distances, _ = self._KDTree.query(center_point, k=k)
            return distances[-1] * self["coordinates"].units
        else:
            return AttributeError(f"Compoennt {self.ptype} is empty! and therefore there is no center-of-mass to refine!")
    
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
        
        Changes basis and los instances in the class instance. All vectorial quantities get expressed in the new
        coordinate basis.
        """
        self._set_los(los)
        self.dq = {"cm": self._old_to_new_base @ self["cm"] if self["cm"] is not None else None}
        self.dq = {"vcm": self._old_to_new_base @ self["vcm"] if self["vcm"] is not None else None}
        return None
       
    def refined_center6d(self, 
                         method="adaptative",
                         **kwargs
                        ):
        """Refined center-of-mass position and velocity estimation. 

        The center of mass of a particle distribution is not well estimated by the full particle ensemble, since the outermost particles
        must not be distributed symmetrically around the TRUE center of mass. For that reason, a closer-to-truth value for th CM can be
        obtained by disregarding these outermost particles.
        
        Here we implement four methods in which a more refined CM estimation can be obtained. All of them avoid using gravitational 
        porentials, as these are not available to observers. Only positions and masses (analogous to luminosities in some sense)
        are used:
            
            1. RADIAL-CUT: Discard all particles outside of rshell = rc_scale * rmax. 
            2. SHRINK-SPHERE: An iterative version of the SIMPLE method, where rshell decreases in steps, with rshell_i+1 = alpha*rshell_i,
                       until an speficief minimun number of particles nmin is reached. Adapted from Pwer et al. 2003.
            3. FRAC-MASS: A variant of the RADIAL-CUT method. Here the cut is done in mass instead of radius. The user can specify a
                       X-mass-fraction and the X-mass radius and its center are computed iterativelly unitl convergence. 
            4. ADAPTATIVE: Performs `SHRINK-SPHERE` if the number of particles is larger than 2*nmin. Otherwise: `RADIAL-CUT`.

        The last radius; r_last, trace of cm positions; trace_cm, number of iterations; iters and final numper of particles; n_particles 
        are stored alongside the center-of-mass.
        
        Center-of-mass velocity estimation is done is done with particles inside v_scale * r_last.
        
        OPTIONAL Parameters
        -------------------

        method : str, optional
            Method with which to refine the CoM: radial-cut/rcc, shrink-sphere/ssc, fractional-mass/mfc od adaptative. Default: ADAPTATIVE
            
        rc_scaling : float
            rshell/rmax for radial-cut method. Must be between 0 and 1. Default: 0.5.
        alpha : float
            Shrink factor for shrink-sphere method. Default: 0.7.
        nmin : int
            Target number of particles for shrink-sphere method. Default: 250.
        mfrac : float
            Mass-fraction for fractional-mass method. Default: 0.3.
        v_scale : float
            Last Radius scale-factor for velocity estimation. Default: 1.5.
            
        Returns
        -------
        cm, vcm : array
            Refined Center of mass and various quantities.
        """
        if self.empty:
            return AttributeError(f"Compoennt {self.ptype} is empty! and therefore there is no center-of-mass to refine!")

        if method.lower() == "pot-most-bound":
            bound_mask = self["total_energy"] < 0
            f = 0.1 if "f" not in kwargs.keys() else kwargs["f"]
            nbound = 32 if "nbound" not in kwargs.keys() else kwargs["nbound"]
            
            N = int(np.rint(np.minimum(f * np.count_nonzero(bound_mask), nbound)))
            most_bound_ids = np.argsort(self["total_energy"])[:N]
            most_bound_mask = np.zeros(len(self["total_energy"]), dtype=bool)
            most_bound_mask[most_bound_ids] = True
            
            tmp_cm = np.average(self["coords"][most_bound_mask], axis=0, weights=self["mass"][most_bound_mask])
            tmp_vcm = np.average(self["velocity"][most_bound_mask], axis=0, weights=self["mass"][most_bound_mask]) 
            
        elif method.lower() == "pot-softmax":
            bound_mask = self["total_energy"] < 0
            T = "adaptative" if "T" not in kwargs.keys() else kwargs["T"]
            
            w = self["total_energy"][bound_mask]/self["total_energy"][bound_mask].min()
            if T == "adaptative":
                T = np.abs(self["kinetic_energy"][bound_mask].mean()/self["total_energy"][bound_mask].min())
                
            tmp_cm = np.average(self["coordinates"][bound_mask], axis=0, weights=softmax(w, T))
            tmp_vcm = np.average(self["velocity"][bound_mask], axis=0, weights=softmax(w, T))
        
        else:                
            self._centering_results = refine_6Dcenter(
                self["coordinates"],
                self["mass"],
                self["velocity"],
                method=method,
                **kwargs
            )
    
            tmp_cm = self._centering_results['center']
            tmp_vcm = self._centering_results['velocity']

        self.dq = {"cm" : tmp_cm}
        self.dq = {"vcm" : tmp_vcm}     
        return tmp_cm, tmp_vcm
    
    def half_mass_radius(self, mfrac=0.5, lines_of_sight=None, project=False):
        """By default, it computes 3D half mass radius of a given particle ensemble. If the center of the particles 
        is not provided, it is estimated by first finding the median of positions and then computing a refined CoM using
        only the particles inside r < 0.5*rmax.
        
        There is also an argument to compute other ALPHA_mass_radius of an arbitrary mass fraction. The desired ALPHA_mass_radius  
        is computed via rootfinding using scipy's implementation of brentq method.
        
        OPTIONAL Parameters
        -------------------
        mfrac : float
            Mass fraction of desired radius. Default: 0.5 (half, mass radius).
        project: bool
            Whether to compute projected quantity or not.
        
        Returns
        -------
        MFRAC_mass_radius : float
            Desired mfrac mass fraction radius estimation. Provided in same units as pos, if any.
        """
        if self.empty:
            return AttributeError(f"Compoennt {self.ptype} is empty! and therefore there is no half mass radius to compute!")
        else:                
            if lines_of_sight is None:
                lines_of_sight = np.array([self.los])
            elif np.array(lines_of_sight).ndim == 1:
                lines_of_sight = np.array([lines_of_sight])
            elif np.array(lines_of_sight).ndim == 2:
                pass
            else:
                raise ValueError(f"Lines of sight does not have the correct number of dimensions. It should have ndims=2, yours has {np.array(lines_of_sight).ndim}")

            tmp_rh_arr = self.arr( -9999 * np.ones((lines_of_sight.shape[0])), self["coordinates"].units)
            for i, los in enumerate(lines_of_sight):
                gs = gram_schmidt(los)
                new_coords = vectorized_base_change(np.linalg.inv(gs), self["coordinates"])
                new_cm = np.linalg.inv(gs) @ self.q["cm"]
                
                tmp_rh_arr[i] = half_mass_radius(
                    new_coords, 
                    self["mass"], 
                    new_cm, 
                    mfrac, 
                    project=project
                )

        if np.abs(mfrac - 0.5) <= 0.01:
            if project:
                self.dq = {"rh" : tmp_rh_arr.mean()}
                self.dq = {"e_rh" : tmp_rh_arr.std()}
            else:
                self.dq = {"rh3d" : tmp_rh_arr.mean()}
                
        return tmp_rh_arr

    def los_dispersion(self, rcyl=(1, 'kpc'), lines_of_sight=None):
        """Computes the line of sight velocity dispersion:  the width/std of f(v)dv of particles iside rcyl along the L.O.S. This is NOT the
        same as the dispersion velocity (which would be the rms of vx**2 + vy**2 + vz**2). All particles are used, including non-bound ones,
        given that observationally they are indistinguishable.

        OPTIONAL Parameters
        ----------
        rcyl : float, tuple[float, str] or unyt_quantity
            Axial radius of the cylinder. Default: 1 kpc.
        return_projections : bool
            Whether to return projected velocities. Default: False.

        Returns
        -------
        stdvel : unyt_quantity
        los_velocities : unyt_array
        """
        
        if self.empty:
            return AttributeError(f"Compoennt {self.ptype} is empty! and therefore there is no line of sight velocity to compute!")
        else:   
            if self.bound and self.ptype != "gas":
                pt = self.ptype + "_bound"
            else:
                pt = self.ptype 
                
            if lines_of_sight is None:
                lines_of_sight = np.array([self.los])
            elif np.array(lines_of_sight).ndim == 1:
                lines_of_sight = np.array([lines_of_sight])
            elif np.array(lines_of_sight).ndim == 2:
                pass
            else:
                raise ValueError(f"Lines of sight does not have the correct number of dimensions. It should have ndims=2, yours has {np.array(lines_of_sight).ndim}")

            tmp_disp_arr = self.arr( -9999 * np.ones((lines_of_sight.shape[0])), self["velocity"].units)
            for i, los in enumerate(lines_of_sight):
                cyl = self._ds.disk(self.basis @ self.q["cm"], los, radius=rcyl, height=(np.inf, 'kpc'), data_source=self._data)
                tmp_disp_arr[i] = easy_los_velocity(cyl[pt, "velocity"], los).std()

        self.dq = {"sigma_los" : tmp_disp_arr.mean()}
        self.dq = {"e_sigma_los" : tmp_disp_arr.std()}
      
        return tmp_disp_arr

    def enclosed_mass(self, r0, center):
        """Computes the enclosed mass on a sphere centered on center, and with radius r0.

        Parameters
        ----------
        r0 : unyt_quantity
            Radius
        center : unyt_array
            Center

        Returns
        -------
        encmass : unyt_quantity
        """
        if self.empty:
            return AttributeError(f"Compoennt {self.ptype} is empty! and therefore there is no enclosed mass to compute!") 
        else:
            mask = np.linalg.norm(self["coordinates"] - center, axis=1) <= r0
            return self["mass"][mask].sum()
 
    def density_profile(self, 
                        center=None,
                        bins=None,
                        projected=False,
                        mask=None,
                        **kwargs
                       ):
        """Computes the average density profile of the particles. Returns r_i (R_i), rho_i and e_rho_i (Sigma_i, e_Sigma_i) for each bin. Center
        and bins are doanematically computed but can also be used specified. Profiles can be for all or bound particles. The smallest two bins
        can be combined into one to counteract lack of resolution. Density error is assumed to be poissonian.

        OPTIONAL Parameters
        ----------
        pc : str
            Particle-Component. Either bound or all.
        center : array
            Center of the particle distribution. Default: None.
        bins : array
            Array of bin edges. Default: None.
        projected : bool
            Whether to get the projected distribution at current LOS. Default: False.
        return_bins : bool
            Whether to return bin edges. Default: False
        
        Returns
        -------
        r, dens, e_dens, (bins, optional) : arrays of bin centers, density and errors (and bin edges)
        """
        if self.empty:
            return AttributeError(f"Compoennt {self.ptype} is empty! and therefore there is no density profile to compute!")

        
        if "new_data" in kwargs.keys():
            sp = kwargs["new_data"] if "sp" in kwargs["new_data"] else self._data.ds.sphere(
                                                                           self.sp_center if "center" not in kwargs["new_data"].keys() else kwargs["new_data"]["center"], 
                                                                           kwargs["new_data"]["radius"]
                                                                       )
            
            pos = vectorized_base_change(
                np.linalg.inv(self.basis), 
                sp[self.ptype, "coordinates"].to(self["coordinates"].units)
            )
            mass = sp[self.ptype, "mass"].to(self["mass"].units)
            
        else:
            pos = self["coordinates"]
            mass = self["mass"]

        mask = np.ones_like(mass.value, dtype=bool) if mask is None else mask
        pos = pos[mask]
        mass = mass[mask]
        
        center = self.refined_center6d(**kwargs["kw_center"])[0].to(pos.units) if center is None else center.to(pos.units)
        
        if projected:
            pos = pos[:, :2]
            center = center[:2]

        
        if bins is None:
            radii = np.linalg.norm(pos - center, axis=1)
            rmin, rmax, thicken, nbins = radii.min(), radii.max(), False, 10
            if "bins_params" in kwargs.keys():
                rmin = rmin if "rmin" not in kwargs["bins_params"].keys() else kwargs["bins_params"]["rmin"]
                rmax = rmax if "rmax" not in kwargs["bins_params"].keys() else kwargs["bins_params"]["rmax"]
                thicken = None if "thicken" not in kwargs["bins_params"].keys() else kwargs["bins_params"]["thicken"]
                nbins = 10 if "bins" not in kwargs["bins_params"].keys() else kwargs["bins_params"]["bins"]
                
            bins = np.histogram_bin_edges(
                np.log10(radii),
                bins=nbins,
                range=np.log10([rmin, rmax]) 
            )
            bins = 10 ** bins

            if thicken is not None:
                binindex = [i for i in range(len(bins)) if i==0 or i>thicken]
                bins = bins[binindex]

        result = density_profile(
            pos,
            mass,
            center=center,
            bins=bins
        )

        if projected and (result["dims"] != 2):
            raise Exception("You fucked up dimensions, bud!")

        result["bins"] = bins
        return result
    
    def velocity_profile(self, 
                         center=None,
                         v_center=None,
                         bins=None,
                         projected="none",
                         quantity="rms",
                         mask=None,
                         **kwargs
                        ):
        """Computes the average disperion velocity profile of the particles. Returns r_i (R_i), vrms_i and e_vrms_i for each bin. Center
        and bins are doanematically computed but can also be used specified. Profiles can be for all or bound particles. The smallest two bins
        can be combined into one to counteract lack of resolution. Density error is assumed to be poissonian.

        OPTIONAL Parameters
        ----------
        pc : str
            Particle-Component. Either bound or all.
        center, v_center : array
            Center of the particle distribution. Default: None.
        bins : array
            Array of bin edges. Default: None.
        projected : bool
            Whether to get the projected distribution at current LOS. Default: False.
        return_bins : bool
            Whether to return bin edges. Default: False
        
        Returns
        -------
        r, vrms, e_vrms, (bins, optional) : arrays of bin centers, density and errors (and bin edges)
        """
        if self.empty:
            return AttributeError(f"Compoennt {self.ptype} is empty! and therefore there is no velocity profile to compute!")

        if "new_data" in kwargs.keys():
            sp = kwargs["new_data"] if "sp" in kwargs["new_data"] else self._data.ds.sphere(
                                                                           self.sp_center if "center" not in kwargs["new_data"].keys() else kwargs["new_data"]["center"], 
                                                                           kwargs["new_data"]["radius"]
                                                                       )
            
            pos = vectorized_base_change(
                np.linalg.inv(self.basis), 
                sp[self.ptype, "coordinates"].to(self["coordinates"].units)
            )
            vels = vectorized_base_change(
                np.linalg.inv(self.basis), 
                sp[self.ptype, "velocity"].to(self["velocity"].units)
            ) 
            mass = sp[self.ptype, "mass"].to(self["mass"].units)
            
        else:
            pos = self["coordinates"]
            vels = self["velocity"]
            mass = self["mass"]
                

        mask = np.ones_like(mass.value, dtype=bool) if mask is None else mask
        pos = pos[mask]
        mass = mass[mask]
        vels = vels[mask]
        
        center = self.refined_center6d(**kwargs["kw_center"])[0].to(pos.units) if center is None else center.to(pos.units)
        v_center = self.refined_center6d(**kwargs["kw_center"])[1].to(vels.units) if v_center is None else v_center.to(vels.units)


        if projected != "none":
            pos = pos[:, :2]
            vels = vels[:, 0] 
            center = center[:2]
            v_center = v_center[0]



        
        if bins is None:
            radii = np.linalg.norm(pos - center, axis=1)
            rmin, rmax, thicken, nbins = radii.min(), radii.max(), False, 10
            if "bins_params" in kwargs.keys():
                rmin = rmin if "rmin" not in kwargs["bins_params"].keys() else kwargs["bins_params"]["rmin"]
                rmax = rmax if "rmax" not in kwargs["bins_params"].keys() else kwargs["bins_params"]["rmax"]
                thicken = None if "thicken" not in kwargs["bins_params"].keys() else kwargs["bins_params"]["thicken"]
                nbins = 10 if "bins" not in kwargs["bins_params"].keys() else kwargs["bins_params"]["bins"]
                
            bins = np.histogram_bin_edges(
                np.log10(radii), 
                bins=nbins,
                range=np.log10([rmin, rmax]) 
            )
            bins = 10 ** bins
            
            if thicken is not None:
                binindex = [i for i in range(len(bins)) if i==0 or i>thicken]
                bins = bins[binindex]

        if projected == "none":
            result = velocity_profile(
                pos,
                vels,
                center=center,
                v_center=v_center,
                bins=bins,
                projected=False,
                average="bins"
            )
        if projected == "radial-bins" or projected == "bins" :
            result = velocity_profile(
                pos,
                vels,
                center=center,
                v_center=v_center,
                bins=bins,
                projected=True,
                average="bins",
                quantity=quantity
            )
        elif projected == "apertures":
            result = velocity_profile(
                pos,
                vels,
                center=center,
                v_center=v_center,
                bins=bins,
                projected=True,
                average="apertures",
                quantity=quantity
            )

        result["bins"] = bins        
        return result



   
            
    














































