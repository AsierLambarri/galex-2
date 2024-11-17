import numpy as np
from unyt import unyt_array, unyt_quantity
from scipy.stats import binned_statistic
from scipy.spatial.transform import Rotation
from scipy.optimize import root_scalar
from numba import (
    vectorize,
    float32,
    float64,
    njit,
    jit,
    prange,
    get_num_threads,
    typed
)

def velocity_profile(pos,
                     vel,
                     mass=None,
                     center = None,
                     v_center = None,
                     bins = None
                    ):
    """Computes the velocity dispersion profile for different radii. The radii are the centers of the bins.

    Parameters
    ----------
    mass : array
        Array of masses of the particles. Shape(N,).
    vel : array
        Array of velocities. Shape(N,3).
    pos : array
        Array of positions of particles.
    bins : array
        Bin edges to use. Recommended to provide externally and to perform logarithmic binning.
    center : array
        Center of particle distribution.
    v_center : array
        Center of mass velocity. If none is provided, it is estimated with all the particles inside X kpc of center.
        
    Returns
    -------
    return_dict : dict
        Dictionary with radii, rho, e_rho, m_enclosed and center.
    """


    if center is not None:
        pass
    else:
        center = np.median(pos, axis=0)
        radii = np.linalg.norm(pos - center, axis=1)
        maskcen = radii < 0.5*radii.max()
        center = np.average(pos[maskcen], axis=0, weights=mass[maskcen])

    if v_center is not None:
        pass
    else:
        v_center = np.average(pos[np.linalg.norm(pos - center, axis=1) < 0.7], axis=0, weights=[np.linalg.norm(pos - center, axis=1) < 0.7])


    
    coords = pos - center
    radii = np.linalg.norm(coords, axis=1)
    magvel = np.linalg.norm(vel - v_center, axis=1)
    if bins is not None:
        vmean, redges, bn = binned_statistic(radii, magvel, statistic="mean", bins=bins)
        npart, _, _ = binned_statistic(radii, magvel, statistic="count", bins=bins)

    else:
        mask = radii <= unyt_quantity(1, pos.units)
    
        bins_lowradii = np.histogram_bin_edges(radii[mask])
        bins_highradii = np.histogram_bin_edges(radii[~mask])

        binedges = np.concatenate([bins_lowradii, bins_highradii[1:]])

        vmean, redges, bn = binned_statistic(radii, magvel, statistic="mean", bins=binedges)
        npart, _, _ = binned_statistic(radii, magvel, statistic="count", bins=bins)
    
    redges = redges * coords.units
    vmean = vmean * vel.units
    e_vmean = vmean / np.sqrt(npart)
    
    rcoords = (redges[1:] + redges[:-1])/2

    return_dict = {'r': rcoords,
                   'vmean': vmean,
                   'e_vmean': e_vmean,
                   'center': center,
                   'v_center': v_center
                  }
    
    return return_dict


def velocity_distribution(vel_r,
                          v_center,
                          ):
    """Returns the velocity distribution f(v,[r,r+dr])dv corresponding to the particles of a shperical shell with
    limiting radii [r,r+dr].

    Parameters
    ----------
    vel_r : array
        Velocities of particles inside [r,r+dr].
    v_center : array
        Velocity of the center
    """
    return None












