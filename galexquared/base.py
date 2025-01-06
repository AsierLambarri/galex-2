#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 20 10:03:19 2024

@author: asier
"""
import numpy as np
from unyt import unyt_array, unyt_quantity
from scipy.spatial import KDTree

from .config import config
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
                            velocity_profile,
                            random_vector_spherical
                            )


class BaseSimulationObject:
    """BaseSimulationObject that contains information shared between all objects in a simulation.
    """
    def __init__(self):
        self._parent = None  
        
        self.base_units = config.base_units
        self.units = config.working_units
        self.ptypes = config.ptypes
        self.fields = config.fields
        
        self.loader = config.default_loader
        self.parser = config.default_parser
        
        self.los = [1, 0, 0]
        self.basis = np.identity(3)
        self._old_to_new_base = np.identity(3)
        
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
    
    
    
class BaseComponent:
    """BaseParticleType class that implements common methods and attributes for particle ensembles. These methods and attributes
    are accesible for all particle types and hence this class acts as a bridge between stars, darkmatter and gas, allowing 
    them to access properties of one another. This makes sense, as particles types in cosmological simulations are coupled to
    each other.
    
    It also simplifies the code, as a plethora of common methods are displaced to here.
    """
    _shared_attrs = {
        "darkmatter": {
            "rockstar_center": None, 
            "rockstar_vel": None,
            "cm": None,
            "vcm": None,
            "rh": None,
            "rh3d": None,
            "rvir": None, 
            "rs": None, 
            "c": None, 
            "vmax": None, 
            "vrms": None
        },
        
        "stars": {
            "cm": None,
            "vcm": None,
            "rh": None,
            "rh3d": None,
            "ML": None,
            "sigma_los": None
        },

        "gas": {
            "cm": None,
            "vcm": None,
            "rh": None,
            "rh3d": None,
            "sigma_los": None
        }
        
    }



            

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
                if key in cls._shared_attrs[pt]:
                    cls._shared_attrs[pt][key] = cls.format_value(value)
                else:
                    raise ValueError(f"Invalid shared attribute '{key}' for type '{pt}'")

        return None
    
    @classmethod
    def get_shared_attr(cls, pt, key):
        """Get a specific shared attribute for a particle type.
        """
        if pt not in cls._shared_attrs:
            raise ValueError(f"Unknown particle type: {pt}")
        return cls._shared_attrs[pt].get(key)

    @classmethod
    def update_shared_attr(cls, pt, key, value):
        """Update a specific shared attribute for a particle type.
        """
        if (pt in cls._shared_attrs) and (key in cls._shared_attrs[pt]):
            cls._shared_attrs[pt][key] = value
        else:
            raise ValueError(f"Cannot update: '{key}' not valid for '{pt}'")

    @classmethod
    def list_shared_attributes(cls, pt):
        """List all shared attributes for a given particle type."""
        return list(cls._shared_attrs.get(pt, {}).keys())

    @classmethod
    def clean_shared_attrs(cls, pt):
        """Reset all shared attributes for a specific particle type to None."""
        if pt not in cls._shared_attrs:
            raise ValueError(f"Unknown particle type: {pt}")
        for key in cls._shared_attrs[pt].keys():
            cls._shared_attrs[pt][key] = None

        return None


    



    

    def _priv__getattr__(self, field_name):
        """Dynamical loader for accessing fields.
        """
        funits = {
            'coords': self.units['length'],
            'hsml': self.units['length'],
            'softs': self.units['length'],
            'masses': self.units['mass'],
            'masses_ini': self.units['mass'],
            'vels': self.units['velocity'],
            'ages': self.units['time'],
            'metallicities': self.units['dimensionless'],
            'thermal_energy': f"{self.units['mass']}*({self.units['velocity']})**2"
        }
        vec_fields = ['coords', 'vels', 'bcoords', 'bvels']
        if field_name  in self._dynamic_fields.keys():
            if field_name in self._fields_loaded:
                return self._fields_loaded[field_name].in_units(funits[field_name]) if field_name in funits else self._fields_loaded[field_name]
            else:
                field = (self._base_ptype, self._dynamic_fields[field_name])
                if field_name in vec_fields:
                    self._fields_loaded[field_name] = vectorized_base_change(np.linalg.inv(self.basis), self._data[field].in_units(funits[field_name]) if field_name in funits else self._data[field])
                else:
                    self._fields_loaded[field_name] = self._data[field].in_units(funits[field_name]) if field_name in funits else self._data[field]
                return self._fields_loaded[field_name]
                
        elif field_name  in ['b'+ f for f in list(self._dynamic_fields.keys())]:
            if field_name in self._fields_loaded:
                return self._fields_loaded[field_name].in_units(funits[field_name[1:]]) if field_name[1:] in funits else self._fields_loaded[field_name]
            else:
                field = (self._base_ptype, self._dynamic_fields[field_name[1:]])
                if field_name in vec_fields:
                    self._fields_loaded[field_name] = vectorized_base_change(np.linalg.inv(self.basis), self._data[field][self._bmask].in_units(funits[field_name[1:]]) if field_name[1:] in funits else self._data[field][self._bmask])
                else:
                    self._fields_loaded[field_name] = self._data[field][self._bmask].in_units(funits[field_name[1:]]) if field_name[1:] in funits else self._data[field][self._bmask]
                return self._fields_loaded[field_name]
        
        try:
            return self.__getattribute__(field_name)
        except AttributeError:
            raise AttributeError(f"Field '{field_name}' not found for particle type {self.ptype}. "+ f"Available fields are: {list(self._dynamic_fields.keys())}")
        
        AttributeError(f"Field {field_name} not found for particle type {self.ptype}. Available fields are: {list(self._dynamic_fields.keys())}")
        return None

    def _delete_vectorial_fields(self):
        """Deletes vectorial fields: a.k.a. coordinates and velocities from cache when called. Used to force the dynamical field loader to re-load vectorial
        fields after a new line of sight has been set, so that the fields are properly oriented.
        """
        f = ['coords', 'vels', 'bcoords', 'bvels']
        for key in f:
            try:      
                del self._fields_loaded[key]
            except:
                continue

        return None

    def _delete_bound_fields(self):
        """Deletes bound fields stored in cache. Used to force the dynamical field loader to re-load these fields after a new boundness has been computed.
        """
        for key in list(self._fields_loaded.keys()):
            if key.startswith("b"):
                del self._fields_loaded[key]

        return None
        




    
    def _default_center_of_mass(self):
        """Computes coarse CoM using all the particles as 

                CoM = sum(mass * pos) / sum(mass)        
        """
        if self.masses.sum() != 0:
            self.cm = center_of_mass_pos(
                self.coords, 
                self.masses
            )
            self.vcm = center_of_mass_vel(
                self.coords, 
                self.masses,
                self.vels,
                R=(1E4, "Mpc")
            )
        else:
            self.empty = True
            self.cm = None 
            self.vcm = None
        return None



    
    def knn_distance(self, center_point, k):
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
        self._KDTree = KDTree(self.coords)        
        distances, _ = tree.query(center_point, k=k)
        
        return distances[-1] * self.coords.units

        
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
        self._delete_vectorial_fields()
        if self.cm is not None:
            self.cm = self._old_to_new_base @ self.cm
        if self.vcm is not None:
            self.vcm = self._old_to_new_base @ self.vcm

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
        cm : array
            Refined Center of mass and various quantities.
        """
        if self.empty:
            return AttributeError(f"Compoennt {self.ptype} is empty! and therefore there is no center-of-mass to refine!")
            
        if method.lower() == "pot-most-bound":
            bound_mask = self.E < 0
            f = 0.1 if "f" not in kwargs.keys() else kwargs["f"]
            nbound = 32 if "nbound" not in kwargs.keys() else kwargs["nbound"]
            
            N = int(np.rint(np.minimum(f * np.count_nonzero(bound_mask), nbound)))
            most_bound_ids = np.argsort(self.E)[:N]
            most_bound_mask = np.zeros(len(self.E), dtype=bool)
            most_bound_mask[most_bound_ids] = True
            
            self.cm = np.average(self.coords[most_bound_mask], axis=0, weights=self.masses[most_bound_mask])
            self.vcm = np.average(self.vels[most_bound_mask], axis=0, weights=self.masses[most_bound_mask]) 
            
        elif method.lower() == "pot-softmax":
            bound_mask = self.E < 0
            T = "adaptative" if "T" not in kwargs.keys() else kwargs["T"]
            
            w = self.E[bound_mask]/self.E[bound_mask].min()
            if T == "adaptative":
                T = np.abs(self.kin[bound_mask].mean()/self.E[bound_mask].min())
                
            self.cm = np.average(self.coords[bound_mask], axis=0, weights=softmax(w, T))
            self.vcm = np.average(self.vels[bound_mask], axis=0, weights=softmax(w, T))               


        else:
            self._centering_results = refine_6Dcenter(
                self.bcoords,
                self.bmasses,
                self.bvels,
                method=method,
                **kwargs
            )
    
            self.cm = self._centering_results['center']
            self.vcm = self._centering_results['velocity']
        return self.cm, self.vcm
    
    def half_mass_radius(self, mfrac=0.5, project=False, only_bound=True):
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
            
        if only_bound:
            if True in self._bmask:
                rh = half_mass_radius(
                    self.bcoords, 
                    self.bmasses, 
                    self.cm, 
                    mfrac, 
                    project=project
                )
            else:
                return AttributeError(f"Compoennt {self.ptype} has no bound mass! and therefore there is no half-mass radius to compute!")
        else:
            rh = half_mass_radius(
                self.coords, 
                self.masses, 
                self.cm, 
                mfrac, 
                project=project
            )
        
        if project:
            self.rh = rh
        else:
            self.rh3d = rh
            
        return rh


    def los_dispersion(self, rcyl=(1, 'kpc'), return_projections=False, only_bound=False):
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
            return AttributeError(f"Compoennt {self.ptype} is empty! and therefore there is no line of sight velocity to compute")

        mask = np.linalg.norm(self.coords[:, 0:2] - self.cm[0:2], axis=1) < unyt_array(*rcyl)

        if only_bound:
            if True in self._bmask:
                los_velocities = easy_los_velocity(self.vels[mask & self._bmask], self.los)
            else:
                return AttributeError(f"Compoennt {self.ptype} has no bound mass! and therefore there is no line of sight velocity to compute!")

        else:
            los_velocities = easy_los_velocity(self.vels[mask], self.los)

        
        losvel = np.std(los_velocities)
        
        if return_projections:
            return losvel, los_velocities
        else:
            return losvel

    def enclosed_mass(self, r0, center, only_bound=False):
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
            
        if only_bound:
            if True in self._bmask:
                mask = np.linalg.norm(self.bcoords - center, axis=1) <= r0
                return self.bmasses[mask].sum()
            else:
                return AttributeError(f"Compoennt {self.ptype} has no bound mass! and therefore there is no enclosed mass to compute!")
        else:
            mask = np.linalg.norm(self.coords - center, axis=1) <= r0
            return self.masses[mask].sum()

    
    def density_profile(self, 
                        pc="bound",
                        center=None,
                        bins=None,
                        projected=False,                      
                        return_bins=False,
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

        
        if "new_data_params" in kwargs.keys():
            if "sp" in kwargs["new_data_params"].keys():
                sp = kwargs["new_data_params"]["sp"]
            else:
                sp = self._data.ds.sphere(
                    self._sp_center if "center" not in kwargs["new_data_params"].keys() else kwargs["new_data_params"]["center"], 
                    kwargs["new_data_params"]["radius"]
                )

            pos = vectorized_base_change(
                np.linalg.inv(self.basis), 
                sp[self._base_ptype, self._dynamic_fields["coords"]].in_units(self.coords.units)
            )
            mass = sp[self._base_ptype, self._dynamic_fields["masses"]].in_units(self.masses.units)
            
        else:
            if pc == "bound":
                if np.any(self._bmask) == False:
                    return AttributeError(f"Compoennt {self.ptype} has no bound mass! and therefore there is no density profile to compute!")
                else:
                    pos = self.bcoords
                    mass = self.bmasses
            elif pc == "all":
                pos = self.coords
                mass = self.masses
                
        
        if center is None:
            center = self.refined_center6d(**kwargs["kw_center"])[0].to(pos.units)

            
        else:
            if isinstance(center, tuple):
                center = unyt_array(*center).to(pos.units)
    

        
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
            raise Exception("you fucked up dimensions, bud")

        if return_bins:
            return result["r"], result["rho"], result["e_rho"], bins
        else:
            return result["r"], result["rho"], result["e_rho"]

    
    def velocity_profile(self, 
                         pc="bound",
                         center=None,
                         v_center=None,
                         bins=None,
                         projected="none",
                         quantity="rms",
                         return_bins=False,
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
            
        if "new_data_params" in kwargs.keys():
            if "sp" in kwargs["new_data_params"].keys():
                sp = kwargs["new_data_params"]["sp"]
            else:
                sp = self._data.ds.sphere(
                    self._sp_center if "center" not in kwargs["new_data_params"].keys() else kwargs["new_data_params"]["center"], 
                    kwargs["new_data_params"]["radius"]
                )
                
            pos = vectorized_base_change(
                np.linalg.inv(self.basis), 
                sp[self._base_ptype, self._dynamic_fields["coords"]].in_units(self.coords.units)
            )
            vels = vectorized_base_change(
                np.linalg.inv(self.basis), 
                sp[self._base_ptype, self._dynamic_fields["vels"]].in_units(self.vels.units)
            )
        else:
            if pc == "bound":
                if np.any(self._bmask) == False:
                    return AttributeError(f"Compoennt {self.ptype} has no bound mass! and therefore there is no velocity profile to compute!")
                else:    
                    pos = self.bcoords
                    vels = self.bvels
            elif pc == "all":
                pos = self.coords
                vels = self.vels

        
        
        if center is None:
            center, v_center = self.refined_center6d(**kwargs["kw_center"])
            center, v_center = center.to(pos.units), v_center.to(vels.units)
        else:
            if isinstance(center, tuple):
                center = unyt_array(*center).to(pos.units)
            if isinstance(v_center, tuple):
                v_center = unyt_array(*v_center).to(vels.units)
        

        if projected != "none":
            pos = pos[:, :2]
            vels = easy_los_velocity(vels - v_center, [1,0,0])
            center = center[:2]
            v_center = np.array([0])



        
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

        
        if return_bins:
            return result["r"], result["v"], result["e_v"], bins
        else:
            return result["r"], result["v"], result["e_v"]



   
        









