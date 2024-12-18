import numpy  as np
from scipy.stats import binned_statistic

from .utils import easy_los_velocity

def density_profile(pos, 
                    mass, 
                    center = None, 
                    bins = None
                   ):
    """Computes the average radial density profile of a set of particles over a number of
    bins. The radii are the centers of the bins. 

    Returns errors on each bin based on the assumption that the statistics are poissonian.
    
    The routine works seamlessly for 3D and 2D projected data, as the particle mass is scalar,
    the only difference radicates in the meaning of "pos" (3D-->r vs 2D-->R_los).
    
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
    npbin[npbin == 0] = 1E20
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
                     bins = None,
                     average="bins",
                    ):
    """Computes the velocity dispersion profile for different radii. The radii are the centers of the bins. 
    
    As the projection of a vectorial quantity is not as straightforward as that of mass profiles, the projection
    is done inside the function and is controlled by the los argument

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
    average : str
        "bins" to average over bins in pos, "apertures" to average over filled apertures.
    
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
        v_center = np.average(vel[np.linalg.norm(pos - center, axis=1) < 0.7], axis=0, weights=mass[np.linalg.norm(pos - center, axis=1) < 0.7])


    coords = pos - center
    radii = np.linalg.norm(coords, axis=1)
    magvel = np.linalg.norm(vel - v_center, axis=1)

    if average == "bins":
        if bins is not None:
            vmean, redges, bn = binned_statistic(radii, magvel, statistic="mean", bins=bins)
            npart, _, _ = binned_statistic(radii, magvel, statistic="count", bins=bins)
    
        else:
            vmean, redges, bn = binned_statistic(radii, magvel, statistic="mean", bins=np.histogram_bin_edges(radii))
            npart, _, _ = binned_statistic(radii, magvel, statistic="count", bins=np.histogram_bin_edges(radii))
        
        redges = redges * coords.units
        rcoords = (redges[1:] + redges[:-1])/2
        
    elif average == "apertures":
        if bins is None:
            bins = np.histogram_bin_edges(radii)
        
        R_aperture = 0.5 * (np.array(bins[:-1]) + np.array(bins[1:]))
                
        cumulative_means = []
        particle_counts = []
        for i in range(len(R_aperture)):
            r_ap = R_aperture[i]
            mask = radii <= r_ap
            mask_bin = (bins[i] <= radii ) & (radii <= bins[i+1])
            
            cumulative_mean = np.mean(magvel[mask]) if np.any(mask_bin) else np.nan
            cumulative_means.append(cumulative_mean)
            
            particle_count = np.sum(mask)
            particle_counts.append(particle_count)


        rcoords = R_aperture * coords.units
        vmean = np.array(cumulative_means)
        npart = np.array(particle_counts)
    
    
    vmean = vmean * vel.units
    if 0 in npart:
        npart[npart == 0] = np.inf
        
    e_vmean = vmean / np.sqrt(npart)
    

    return_dict = {'r': rcoords,
                   'vrms': vmean,
                   'e_vrms': e_vmean,
                   'center': center,
                   'v_center': v_center
                  }
    
    return return_dict




