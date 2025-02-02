import numpy as np
from .class_methods import gram_schmidt, vectorized_base_change



def half_mass_radius(pos, mass, cm, mfrac=0.5, lines_of_sight=None, project=False):
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
        if lines_of_sight is None:
            lines_of_sight = np.array([[1,0,0]])
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
            new_cm = np.linalg.inv(gs) @ cm
            
            tmp_rh_arr[i] = half_mass_radius(
                new_coords, 
                mass, 
                new_cm, 
                mfrac, 
                project=project
            )

        return tmp_rh_arr