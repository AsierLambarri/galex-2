import os
import yt
import shutil
import numpy as np
from unyt import unyt_array, unyt_quantity

from src.config import config



class zHalo:
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
    #__slots__ = ['info', 'load_dataset']
    def __init__(self,
                 fn,
                 center,
                 radius,
                 data=None,
                 **kwargs
                ):
        """Initialize function.
        """
        units = config.units

        default_kwargs = {'units': units,
                         }
        self.__update_kwargs__(default_kwargs, kwargs)

        ###File initialization information
        self.fn = fn
        self.sp_center = unyt_array(center[0], center[1])
        self.sp_radius = unyt_quantity(radius[0], radius[1])
        self.z = 0
        self.a = 1
        self.t = unyt_quantity(14, "Gyr")
        self.data = self.load_dataset()
        ###Properties of stars
        #self.stars = load_stars
        #self.dm = load_dm
        #self.gas = load_gas

        ###Coordinate system
        self.los = [1, 0, 0]
        self.basis = np.identity(3)




    ####################################
    ###                              ###
    ##            UTILITIES           ##
    ###                              ###
    ####################################
    
    def __update_kwargs__(self, default_kw, new_kw):
        """Update default kwargs with user provided kwargs.
        """
        for key, value in new_kw.items():
            if isinstance(value, dict) and key in default_kw and isinstance(default_kw[key], dict):
                self.__update_kwargs__(default_kw[key], value)
            else:
                default_kw[key] = value

        self.__kwargs__ = default_kw
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
        output.append(f"{'redshift':<20}: {self.z}")
        output.append(f"{'scale_factor':<20}: {self.a}")
        output.append(f"{'age':<20}: {self.t}")
        output.append(f"{'cut-out center':<20}: [{self.sp_center[0].value:.2f}, {self.sp_center[1].value:.2f}, {self.sp_center[2].value:.2f}]  {self.sp_center.units}")
        output.append(f"{'cut-out radius':<20}: {self.sp_radius:.2f}")
        output.append(f"{'dm':<20}: yes")
        output.append(f"{'stars':<20}: yes")
        output.append(f"{'gas':<20}: yes")

        output.append(f"\nunits")
        output.append(f"{'':-<21}")
        output.append(f"{'length_unit':<20}: { self.__kwargs__['units']['length']}")
        output.append(f"{'velocity_unit':<20}: { self.__kwargs__['units']['velocity']}")
        output.append(f"{'mass_unit':<20}: { self.__kwargs__['units']['mass']}")
        output.append(f"{'time_unit':<20}: { self.__kwargs__['units']['time']}")
        output.append(f"{'comoving':<20}: { self.__kwargs__['units']['comoving']}")

        output.append(f"\ncoordinate basis")
        output.append(f"{'':-<21}")
        output.append(f"{'type':<20}: orthonormal")
        output.append(f"{'line-of-sight':<20}: {self.los}")
        output.append(f"{'u1':<20}: {self.basis[:,0]}")
        output.append(f"{'u2':<20}: {self.basis[:,1]}")
        output.append(f"{'u3':<20}: {self.basis[:,2]}")


        if get_str:
            return None
        else:
            print("\n".join(output))
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
        """

        if config.loader is None:
            raise Exception("The module-level loader is not set! You can set it up as: import pkg;"+
                            " pkg.config.loader = loader_function before you start your scripts. Beware of the requirements that this loader must fulfill!")
        else:
            return config.loader(self.fn)
        

































