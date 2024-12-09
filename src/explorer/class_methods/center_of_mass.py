import numpy as np
from unyt import unyt_quantity

from .half_mass_radius import half_mass_radius
from .utils import easy_los_velocity

def center_of_mass_pos(pos,
                       mass
                       ):
    """Computes coarse CoM using all the particles as 
    
                CoM = sum(mass * pos) / sum(mass)
        
    Parameters
    ----------
    pos : array-like[float], shape(N,dims)
        Positions of particles. First dimension of the array corresponds to each particle.
        Second dimension correspond to each coordiante axis.
    mass : array-like, shape(N,)
        Masses of particles.

    Returns
    -------
    CoM_pos : array-like
    """
    return np.average(pos, axis=0, weights=mass)

def center_of_mass_vel(pos,
                       mass,
                       vel,
                       center=None,
                       R=(0.7, 'kpc'),
                      ):
    """Computes the center of mass velocity as 

                    CoM = sum(mass * vel) / sum(mass)
                    
    using only particles inside a specified R around the estimated CoM position.

    Parameters
    ----------
    pos : array-like[float], shape(N,dims)
        Positions of particles. First dimension of the array corresponds to each particle.
        Second dimension correspond to each coordiante axis.
    mass : array-like, shape(N,)
        Masses of particles.
    vel : array-like, shape(N,)
        Velocities of particles.
    center : array-like, shape(N,)
        CoM position
    R : tuple or unyt_*
        Radius for selecting particles. Default 0.7 'kpc'.
        
    Returns
    -------
    CoM_vel : array-like
    """
    if center is None:
        center = center_of_mass_pos(pos, mass)
        
    mask = np.linalg.norm(pos - center, axis=1) < unyt_quantity(*R)
    return np.average(vel[mask], axis=0, weights=mass[mask])
    


        


def center_of_mass_vel_through_proj(pos,
                                    vel,
                                    center=None,
                                    rcyl=(1E4, "Mpc"),
                                    h=(1E4, "Mpc")
                                   ):
    """Computes the center of mass velocity by looking at the projected velocities through the x,y,z-axes of
    the provided coordinates. At each projection, only the particles within cylindrical distance rmax from the center
    and  a longitudinal distance h are taken into acocunt, to avoid picking up more than one kinematic component.
    Center-of-mass is taken to be the simple mean of each cylinder/projection.

    It is slightly different to center_of_mass_vel. Use not recomended.
    
    Parameters
    ----------
    pos : array
        Position array.
    vel : array
        Velocity array.
    center : array, optional
        Pre-computed center-of-mass.
    rcyl : tuple(float, str) or unyt_quantity
        Maximum radius.

    Returns
    -------
    cm_vel : array
    """
    mask_x = ( np.linalg.norm(pos[:,[1,2]] - center[1,2], axis=1) < unyt_array(*rcyl) ) & ( np.abs(pos[:,0] - center[0]) < unyt_array(*h) )
    mask_y = ( np.linalg.norm(pos[:,[0,2]] - center[0,2], axis=1) < unyt_array(*rcyl) ) & ( np.abs(pos[:,1] - center[1]) < unyt_array(*h) )
    mask_z = ( np.linalg.norm(pos[:,[0,1]] - center[0,1], axis=1) < unyt_array(*rcyl) ) & ( np.abs(pos[:,2] - center[2]) < unyt_array(*h) )

    los_velocities_x = np.mean(easy_los_velocity(vel[mask_x], [1,0,0]))
    los_velocities_y = np.mean(easy_los_velocity(vel[mask_y], [0,1,0]))
    los_velocities_z = np.mean(easy_los_velocity(vel[mask_z], [0,0,1]))
    try:
        return unyt_array([los_velocities_x, los_velocities_y, los_velocities_z], vel.units)
    except:
        return np.array([los_velocities_x, los_velocities_y, los_velocities_z])
     
    


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
        centering_results = _cm_mfrac_method(pos, mass, center,
                                             delta, m, mfrac)
    if method == "simple":
        centering_results = _cm_simple_method(pos, mass, center, scaling)
        
    if method == "iterative":
        centering_results = _cm_iterative_method(pos, mass, center,
                                                    delta, alpha, m, nmin)  
    if method == "iterative-hm":
        centering_results = _cm_iterative_mfrac_method(pos, mass, center,
                                                          delta, m, nmin, alpha)  
        
    centering_results['center'] *= lengthunit
    centering_results['trace_cm'] *= lengthunit
    centering_results['r_last'] *= lengthunit
    centering_results['delta'] *= lengthunit
    centering_results['trace_delta'] *= lengthunit
    
    return centering_results




def _cm_simple_method(pos,
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

def _cm_mfrac_method(pos,
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

def _cm_iterative_method(pos,
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
                                 'delta': diff ,
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


def _cm_iterative_mfrac_method(pos,
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
        inter_cent = _cm_mfrac_method(pos, mass, center, 1E-1 * delta, m, mfrac)
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




