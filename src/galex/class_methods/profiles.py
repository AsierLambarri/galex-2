import numpy  as np
from scipy.stats import binned_statistic

def density_profile(pos, 
                    mass, 
                    center = None, 
                    bins = None, 
                    b_to_a=1, 
                    c_to_a=1, 
                    A=[1,0,0]
                   ):
    """Computes the radial density profile of a set of particles with geometry given by an
    ellipsoid with {a,b,c} axes and the major axis in the direction of A_vec. The binning is
    performed over 

                    m = sqrt{ (x/a)^2 + (y/b)^2 + (z/c)^2 }

    Returns errors on each bin based on the assumption that the statistics are poissonian.

    Parameters
    ----------
    pos : array
        Array of positions of the particles. Shape(N,3).
    mass : array
        Array of masses. Shape(N,).
    bins : array
        Bin edges to use. Recommended to provide externally and to perform logarithmic binning.
    center : array
        Center of particle distribution.
        
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
        
        
    coords = pos - center
    radii = np.linalg.norm(coords, axis=1)
    if bins is not None:
        mi, redges, bn = binned_statistic(radii, mass, statistic="sum", bins=bins)
        npbin, _, _ = binned_statistic(radii, mass, statistic="count", bins=bins)

    else:
        mi, redges, bn = binned_statistic(radii, mass, statistic="sum", bins=np.histogram_bin_edges(radii))
        npbin, _, _ = binned_statistic(radii, mass, statistic="count", bins=np.histogram_bin_edges(radii))
    
    redges = redges * coords.units
    mi = mi * mass.units

    if pos.shape[1] == 3:
        volumes = 4 * np.pi / 3 * ( redges[1:]**3 - redges[:-1]**3 )
    if pos.shape[1] == 2:
        volumes = np.pi * ( redges[1:]**2 - redges[:-1]**2 )

    
    rcoords = (redges[1:] + redges[:-1])/2
    dens = mi / volumes
    error = dens / np.sqrt(npbin)
    
    return_dict = {'r': rcoords,
                   'rho': dens,
                   'e_rho': error,
                   'm_enc': mi,
                   'center': center,
                   'dims': pos.shape[1]
                  }
    return return_dict








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
        vmean, redges, bn = binned_statistic(radii, magvel, statistic="mean", bins=np.histogram_bin_edges(radii))
        npart, _, _ = binned_statistic(radii, magvel, statistic="count", bins=np.histogram_bin_edges(radii))
    
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




