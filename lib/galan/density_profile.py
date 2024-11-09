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
        mask = radii <= unyt_quantity(1, pos.units)
    
        bins_lowradii = np.histogram_bin_edges(radii[mask])
        bins_highradii = np.histogram_bin_edges(radii[~mask])

        binedges = np.concatenate([bins_lowradii, bins_highradii[1:]])

        mi, redges, bn = binned_statistic(radii, mass, statistic="sum", bins=binedges)
        npbin, _, _ = binned_statistic(radii, mass, statistic="count", bins=binedges)
    
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

def refine_center(pos, 
                  mass, 
                  scaling=0.5,
                  method="simple",
                  delta=1E-2,
                  alpha=0.95,
                  m=2,
                  nmin=100,
                  mfrac=0.5
                 ):
    """Refined CM position estimation. 

    The CoM of a particle distribution is not well estimated by the full particle ensemble, 

    Parameters
    ----------
    pos : array
        Array of positions.
    mass : array
        Array of masses.
    method : str, optional
        Method with which to refine the CoM: simple, interative or iterative-hm.
    delta : float
        Tolerance for stopping re-refinement. Default 1E-2.
    spacing : float
        Spacing between two consecutive shell radii when refining the CoM. A good guess is the softening length. Default is 0.08 in provided units.

    Returns
    -------
    centering_results : array
        Refined Center of mass and various quantities.
    """    
    lengthunit = pos.units
    pos = pos.value
    mass = mass.value
    
    center = np.median(pos, axis=0)
    
    

    if method == "hm":
        centering_results = __cm_mfrac_method__(pos, mass, center,
                                                delta, m, mfrac)    
    if method == "simple":
        centering_results = __cm_simple_method__(pos, mass, center, scaling)
        
    if method == "iterative":
        centering_results = __cm_iterative_method__(pos, mass, center,
                                                    delta, alpha, m, nmin)  
    if method == "iterative-hm":
        centering_results = __cm_iterative_mfrac_method__(pos, mass, center,
                                                          delta, m, nmin, alpha)  
    return centering_results




def __cm_simple_method__(pos,
                         mass, 
                         center,
                         scaling
                        ):

    """Computes the CoM by first finding the median of positions and then computing a refined CoM using
    only the particles inside r < 0.5*rmax. Usually is robust enough for estimations with precision of
    0.1-0.5 POS units.

    This approach has been inspired by Mark Grudic.
    """
    trace_cm = np.array([center])
    trace_delta = np.array([1])

    radii = np.linalg.norm(pos - center, axis=1)
    rmax = radii.max()
    center_new = np.average(pos[radii < scaling * rmax], axis=0, weights=mass[radii < scaling * rmax])
    centering_results = {'center': center_new ,
                         'delta': np.linalg.norm(center_new - center),
                         'r_last': scaling * rmax ,
                         'iters': 2,
                         'trace_delta':  np.linalg.norm(center_new - center) ,
                         'trace_cm': np.append(trace_cm, [center_new], axis=0),
                         'n_particles': len(pos[radii < scaling * rmax]),
                         'converged': True
                        }

    return centering_results

def __cm_mfrac_method__(pos,
                        mass,
                        center,
                        delta,
                        m,
                        mfrac
                        ):
    """Computes the center of mass by computing the half mass radius iteratively until the center of mass
    of the enclosed particles converges to DELTA in M consecutive iterations. This method is less affected by particle 
    number because the CoM estimation is always done with half the particles of the ensemble, this way, we avoid
    the inconveniences of the iterative method in that the cm can be more robustly computed for ensembles with N PART
    much smaller: around 10 to 20 particles are enough to reliably estimate the CoM.

    The half-mass radius is computed with the ``half_mass_radius`` routine.

    As the CoM estimation gets more precise whe using the most central particles, but suffers when having few particles,
    you can change the mass fraction with which to estimate the CoM (i.e. you can choose if you want the half-mass radius,
    quarter-mass radius, 75-percent-mass radius etc.).

    The routine has a maximum of 100 iterations to avoid endless looping.
    """

    trace_cm = np.array([center])
    trace_delta = np.array([1])
    converged = False
    for i in range(100):
        rhalf = half_mass_radius(pos, mass,  center=center, mfrac=mfrac)
        mask = np.linalg.norm(pos - center, axis=1) <= rhalf
        npart = len(pos[mask])
        
        center_new = np.average(pos[mask], axis=0, weights=mass[mask])

        diff = np.sqrt( np.sum((center_new - center)**2, axis=0) )           
        trace_cm = np.append(trace_cm, [center_new], axis=0)
        trace_delta = np.append(trace_delta, diff)
        
        if np.all(trace_delta[-m:] < delta):
            converged = True            
            break
            
        else:
            center = center_new

    centering_results = {'center': center_new ,
                         'delta': diff,
                         'r_last': rhalf ,
                         'iters': i + 1,
                         'trace_delta': trace_delta ,
                         'trace_cm': trace_cm,
                         'n_particles': npart,
                         'converged': converged
                        }
    return centering_results

def __cm_iterative_method__(pos,
                            mass,
                            center,
                            delta,
                            alpha, 
                            m, 
                            nmin
                           ):
    """Iterative method where the center of mass is computed using the particles inside an ever-shrinking
    sphere, until subsequent CoM estimations converge to DELTA in M consecutive iterations, the number of enclosed
    particles is smaller than NMIN or the radius reaches 0.3 in units of POS. The radius of the sphere shrinks 
    by 1-alpha in each iteration. More generally: r_i = alpha^i * r_0 where r_0 is RADII.max().

    For systems with few particles (this is left to user discretion, usually we take <100 particles or <5*nmin) this
    routine struggles to converge bellow DELTA=5E-2 because the removal of a single particle may cause the CoM to 
    move by more than said amount. This is an inherent problem to this method when applied to low particle count systems
    because each shrinkage removes some particles. In said cases, the routine may not converge, in which case,
    the user is splicitly told that the subroutine did not converge.
    """
    radii = np.linalg.norm(pos - center, axis=1)
    rmax, rmin = 1.001 * radii.max(), 0.1
    
    n = int( -np.log(rmax/rmin)/np.log(alpha))
    ri = rmax * alpha**np.linspace(0,n, n+1)
    
    trace_cm = np.array([center])
    trace_delta = np.array([1])
    
    for i, rshell in enumerate(ri):
        shellmask = np.sqrt(np.sum((pos - center)**2, axis=1)) <= rshell
        
        if len(pos[shellmask]) <= nmin:
            final_cm = center
            centering_results = {'center': final_cm ,
                                 'delta': diff,
                                 'r_last': rshell ,
                                 'iters': i ,
                                 'trace_delta': trace_delta ,
                                 'trace_cm': trace_cm,
                                 'n_particles': len(pos[shellmask]),
                                 'converged': False
                                }
            break
            
        center_new = np.average(pos[shellmask], axis=0, weights=mass[shellmask])
        diff = np.sqrt( np.sum((center_new - center)**2, axis=0) )           
        trace_cm = np.append(trace_cm, [center_new], axis=0)
        trace_delta = np.append(trace_delta, diff)
        
        if np.all(trace_delta[-m:] < delta):
            final_cm = center_new
            centering_results = {'center': final_cm ,
                                 'delta': diff,
                                 'r_last': rshell ,
                                 'iters': i + 1,
                                 'trace_delta': trace_delta ,
                                 'trace_cm': trace_cm,
                                 'n_particles': len(pos[shellmask]),
                                 'converged': True 
                                }
            break
            
        else:
            center = center_new


    return centering_results                


def __cm_iterative_mfrac_method__(pos,
                                  mass,
                                  center,
                                  delta,
                                  m,
                                  nmin,
                                  alpha
                                 ):
    """This method is a combination of the mfrac and iterative method. In each iteration, the center of mass for a 
    mfrac mass sphere is computed using the mfrac method, until convergence. This is iterated for decreasing values of
    mfrac until convergence.

    The last value of mfrac is determined by the MINIMUM NUMBER OF PARTICLES required by the user and convergence is stablished
    when M consecutive iterations have converged to DELTA. The MFRAC gets reduced by 1-alpha for each iteration.
    """
    if nmin >= len(mass):
        raise Exception(f"The particle ensemble you provided doesnt have enough particles! You specified {nmin} minimum particles but has {len(mass)}.")
        
    nmass = mass.sum() / len(mass)
    min_mass_frac = 1.2 * nmass * nmin / mass.sum()
    
    n = int( -np.log(0.75/min_mass_frac)/np.log(alpha))
    mfracs = 0.75 * alpha**np.linspace(0,n, n+1)
    
    trace_cm = np.array([center])
    trace_delta = np.array([1])  
    converged = False
    for i, mfrac in enumerate(mfracs):
        inter_cent = __cm_mfrac_method__(pos, mass, center, 1E-1 * delta, m, mfrac)
        center_new = inter_cent['center']
        rshell = inter_cent['r_last']
        npart = inter_cent['n_particles']
        
        diff = np.sqrt( np.sum((center_new - center)**2, axis=0) )           
        trace_cm = np.append(trace_cm, [center_new], axis=0)
        trace_delta = np.append(trace_delta, diff)
        
        if np.all(trace_delta[-m:] < delta):
            converged = True
            break
            
        else:
            center = center_new

    centering_results = {'center': center_new ,
                         'delta': diff,
                         'r_last': rshell ,
                         'iters': i + 1,
                         'trace_delta': trace_delta ,
                         'trace_cm': trace_cm,
                         'n_particles': npart,
                         'converged': converged 
                        }
    
    return centering_results  
    



def refine_centerGRAV(pos,
                      mass,
                      vel,
                      ids,
                      soft = None,
                      delta=1E-2,
                      refine=True,
                      nmin=32,
                      f=0.1,
                      theta=0.7
                     ):
    """Computes the Center of Mass using Mike Grudic's and Alexander Gurvich's PYTREEGRAV Barnes-Hut code. PYTREEGRAV 
    is used to obtain the most bound particles, the center of mass and center of mass velocity are computed with those 
    particles. The number of particles used for this is goberned by F and NMIN as MIN(F*NTOT, NMIN). The estimation can be refined
    until it absolutely converges to DELTA.

    This routine works the same as bound_particlesBH (they should be merged at some point) but instead returs the CoM and Vcm. 

    This function computes both quantities simultaneously for the DM and STARS profile, as  the gravitational system is formed by both 
    components.

    Parameters
    ----------
    pos : array
        Array of positions.
    mass : array
        Array of masses.
    vel : array
        Array of velocities.
    ids : array
        Array of particle ids. Used to identify STAR and DM particles.
    delta : float
        Tolerance for stopping re-refinement. Default 1E-2. Absolute.
    refine : bool
        Whether to refine
    nmin : int
        Number of particles used to estimate the cm and vcm.
    f : float
        Fraction of particles used to estimate the cm and vcm.
    theta : float
        BH code theta parameter.

    Returns
    -------
    result : dict
        Dictionary with results.
    """


    cm = np.average(pos.in_units("kpc"), axis=0, weights=mass) if cm is None else cm
    vcm = np.average(vel.in_units("km/s"), axis=0, weights=mass) if vcm is None else vcm

    softenings = soft.in_units("kpc") if soft is not None else None
    
    pot, tree = Potential(pos.in_units("kpc"), mass.in_units("Msun"), softenings,
                          parallel=True, quadrupole=True, G=4.300917270038e-06, theta=theta, return_tree=True)

    for i in range(100):
        abs_vel = np.sqrt( (vel[:,0]-vcm[0])**2 +
                           (vel[:,1]-vcm[1])**2 + 
                           (vel[:,2]-vcm[2])**2
                       )
    
        kin = 0.5 * mass.in_units("Msun") * abs_vel**2
            
        pot = mass.in_units("Msun") * unyt_array(pot, 'km**2/s**2')
        E = kin + pot
        bound_mask = E < 0
        
        
        N = int(np.rint(np.minimum(0.1 * np.count_nonzero(bound_mask), 32)))
        most_bound_ids = np.argsort(E)[:N]
        most_bound_mask = np.zeros(len(E), dtype=bool)
        most_bound_mask[most_bound_ids] = True
        
        new_cm = np.average(pos[most_bound_mask].in_units("kpc"), axis=0, weights=mass[most_bound_mask])
        new_vcm = np.average(vel[most_bound_mask].in_units("km/s"), axis=0, weights=mass[most_bound_mask])        
     
        delta_cm = np.sqrt(np.sum((new_cm - cm)**2, axis=0)) / np.linalg.norm(cm) < delta
        delta_vcm =  np.sqrt(np.sum((new_vcm - vcm)**2, axis=0)) / np.linalg.norm(vcm) < delta        
        
        if not refine or (delta_cm and delta_vcm):

            
            return (bound_mask, most_bound_mask, ids[bound_mask], ids[most_bound_mask]) if ids is not None else (bound_mask, most_bound_mask)

        cm, vcm = copy(new_cm), copy(new_vcm)





def half_mass_radius(pos, 
                     mass, 
                     center=None,
                     mfrac=0.5
                    ):
    """By default, it computes half mass radius of a given particle ensemble. If the center of the particles 
    is not provided, it is estimated by first finding the median of positions and then computing a refined CoM using
    only the particles inside r < 0.5*rmax.

    There is also an argument to compute other ALPHA_mass_radius of an arbitrary mass fraction. The desired ALPHA_mass_radius  
    is computed via rootfinding using scipy's implementation of brentq method.

    Parameters
    ----------
    pos : array
        Array of positions.
    mass : array
        Array of masses.
    center : array, optional
        Center of mass position. If not provided it is estimated as explained above.
    mfrac : float, optional
        Mass fraction of desired radius. Default: 0.5 (half, mass radius).

    Returns
    -------
    MFRAC_mass_radius : float
        Desired mfrac mass fraction radius estimation. Provided in same units as pos, if any.
    """
    if mfrac > 1:
        raise Exception(f"Mass fraction MFRAC must be between 0 and 1! Your input was {mfrac:.2f} > 1")
    if mfrac < 0:
        raise Exception(f"Mass fraction MFRAC must be between 0 and 1! Your input was {mfrac:.2f} < 0")
        
    if center is not None:
        pass
    else:
        center = np.median(pos, axis=0)
        radii = np.linalg.norm(pos - center, axis=1)
        center = np.average(pos[radii < 0.5 * radii.max()], axis=0, weights=mass[radii < 0.5 * radii.max()])
  
    coords = pos - center
    radii = np.sqrt(np.sum(coords**2, axis=1))
    
    halfmass_zero = root_scalar(lambda r: mass[radii < r].sum()/mass.sum() - mfrac, method="brentq", bracket=[0, radii.max()])

    if halfmass_zero.flag != "converged":
        raise Exception(f"Could not converge on {mfrac:.2f} mass radius!")
    else:
        try:
            return halfmass_zero.root * coords.units
        except:
            return halfmass_zero.root
            








"""
def refine_center(pos, 
                  mass, 
                  scaling = 0.5,
                  iterate=False, 
                  delta=1E-2,
                  alpha=0.95,
                  m=2,
                  nmin=100,
                  
                 ):
    Refined CM position estimation. The center can be computed by just two interations, first
    by estimating the center as the median of positions and then by cutting at r < 0.5*rmax and computing
    the center of mass explicitly with those particles.

    It can also be refined, in which case the first temptative 'median' CoM is refined by computing the CoM is
    shrinking spherical shells until either less than 7 particles are found inside or |cm1-cm2|<delta.

    Parameters
    ----------
    pos : array
        Array of positions.
    mass : array
        Array of masses.
    rerefine : bool
        Wether to refine the refined CoM (sic!). Default True.
    delta : float
        Tolerance for stopping re-refinement. Default 1E-2.
    spacing : float
        Spacing between two consecutive shell radii when refining the CoM. A good guess is the softening length. Default is 0.08 in provided units.

    Returns
    -------
    centering_results : array
        Refined Center of mass and various quantities.
        
    lengthunit = pos.units
    pos = pos.value
    mass = mass.value
    
    center = np.median(pos, axis=0)
    radii = np.linalg.norm(pos - center, axis=1)
    
    rmax, rmin = 1.001 * radii.max(), 0.3
    
    n = int( -np.log(rmax/rmin)/np.log(alpha))
    ri = rmax * alpha**np.linspace(0,n, n+1)
    
    trace_cm = np.array([center])
    trace_delta = np.array([1])


    if len(pos) < 5*nmin:
        iterate = False
    
    if iterate:
        #@njit(fastmath=True) 
        for i, rshell in enumerate(ri):
            shellmask = np.sqrt(np.sum((pos - center)**2, axis=1)) <= rshell
            
            if len(pos[shellmask]) <= nmin:
                final_cm = center
                centering_results = {'center': final_cm ,
                                     'delta': diff,
                                     'r_last': rshell ,
                                     'iters': i ,
                                     'trace_delta': trace_delta ,
                                     'trace_cm': trace_cm,
                                     'n_particles': len(pos[shellmask]),
                                     'converged': False
                                    }
                return centering_results                
                
            center_new = np.average(pos[shellmask], axis=0, weights=mass[shellmask])
            diff = np.sqrt( np.sum((center_new - center)**2, axis=0) )           
            trace_cm = np.append(trace_cm, [center_new], axis=0)
            trace_delta = np.append(trace_delta, diff)
            
            if np.all(trace_delta[-m:] < delta):
                final_cm = center_new
                centering_results = {'center': final_cm ,
                                     'delta': diff,
                                     'r_last': rshell ,
                                     'iters': i + 1,
                                     'trace_delta': trace_delta ,
                                     'trace_cm': trace_cm,
                                     'n_particles': len(pos[shellmask]),
                                     'converged': True 
                                    }
                return centering_results                
                
            else:
                center = center_new
                
    else:
        center_new = np.average(pos[radii < scaling * rmax], axis=0, weights=mass[radii < scaling * rmax])

        
        centering_results = {'center': center_new ,
                             'delta': np.linalg.norm(center_new - center),
                             'r_last':radii < scaling * rmax ,
                             'iters': 2,
                             'trace_delta':  np.linalg.norm(center_new - center) ,
                             'trace_cm': np.append(trace_cm, [center_new], axis=0),
                             'n_particles': len(pos[radii < scaling * rmax]),
                             'converged': True
                            }
        return centering_results

    centering_results = {'center': center_new ,
                         'delta': diff,
                         'r_last': rshell ,
                         'iters': i + 1,
                         'trace_delta': trace_delta ,
                         'trace_cm': trace_cm,
                         'n_particles': len(pos[shellmask]),
                         'converged': False
                        }
    return centering_results
"""


        
