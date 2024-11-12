import numpy as np


def random_vector_spherical(N):
    """
    Generate N uniformly distributed random points on the surface of a sphere of radius 1.
    
    Parameters
    ----------
    N : int
        Number of random poits to sample

    
    Returns
    -------
    points : array
        Array of shape (N,3) containing uniformly distributed points on the surface of a  sphere of radius 1.
    """
    np.random.seed()
    phi = np.random.uniform(0, 2 * np.pi, size=N)
    
    cos_theta = np.random.uniform(-1, 1, size=N)
    theta = np.arccos(cos_theta)
    
    x =  np.sin(theta) * np.cos(phi)
    y =  np.sin(theta) * np.sin(phi)
    z =  cos_theta
    
    return np.vstack((x, y, z)).T



def LOS_velocity(vel, los):
    """Computes the Line of Sight velocity of particles along a given LOS.

    Parameters
    ----------
    vel : array
        Array of velocities. Shape (N,3).
    los : array
        Array of shape (3,) representing the LOS. May not be normalized.

    Returns
    -------
    projected_vel : array
        Array of projected velocity magnitudes along the LOS.
    """

    los = los / np.linalg.norm(los)
    return np.dot(vel, los)
