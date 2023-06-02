import os
import time
import numpy as np
from multiprocessing import Pool
from scipy.spatial import cKDTree


def paralell_hist_loop(radii_and_indices, kdtree_center, particles_center, kdtree_to_check, particles_to_check, mins, maxs, N_radii, dr, eps, rho):
    """RDF histogram loop process for multiprocessing"""
    N, d = particles_center.shape
    g_r_partial = np.zeros(shape=(N_radii))

    for r_idx, r in radii_and_indices:
        r_idx = int(r_idx)
        # find all particles that are at least r + dr away from the edges of the box
        valid_idxs = np.bitwise_and.reduce([(particles_center[:, i]-(r+dr) >= mins[i]) & (particles_center[:, i]+(r+dr) <= maxs[i]) for i in range(d)])
        valid_particles_center = particles_center[valid_idxs]

        valid_idxs = np.bitwise_and.reduce([(particles_to_check[:, i]-(r+dr) >= mins[i]) & (particles_to_check[:, i]+(r+dr) <= maxs[i]) for i in range(d)])
        valid_particles_to_check = particles_to_check[valid_idxs]
        
        # compute n_i(r) for valid particles.
        for particle in valid_particles_center:
            n = kdtree_to_check.query_ball_point(particle, r+dr-eps, return_length=True) - kdtree_to_check.query_ball_point(particle, r, return_length=True)
            g_r_partial[r_idx] += n
        
        # normalize
        valid_particles = np.unique(np.concatenate((valid_particles_center, valid_particles_to_check), axis=0), axis=0)
        n_valid = len(valid_particles)
        shell_vol = (4/3)*np.pi*((r+dr)**3 - r**3) if d == 3 else np.pi*((r+dr)**2 - r**2)
        g_r_partial[r_idx] /= n_valid*shell_vol*rho
    
    return g_r_partial

def asym_rdf(particles_center, dr, particles_to_check=None, rho=None, rcutoff=0.9, eps=1e-15, parallel=True, progress=False):
    """
    Computes 2D or 3D radial distribution function g(r) of two sets of particle 
    coordinates of shape (N, d). Particle must be placed in a 2D or 3D cuboidal 
    box of dimensions [width x height (x depth)]. It supports comparison between 
    different particle lists to accomodate for rdf between different particle types 
    or the possibility to add periodic structures.
    
    Parameters
    ----------
    particles_center : (N, d) np.array
        Set of particle from which to compute the radial distribution function 
        g(r). These are the centered positions of each radial calculation. 
        Must be of shape (N, 2) or (N, 3) for 2D and 3D coordinates 
        repsectively.
    particles_to_check : (N, d) np.array, optional
        Set of particle from which to compute the radial distribution function 
        g(r). These are the positions which will be checked from ech point in particles_center
        Must be of shape (N, 2) or (N, 3) for 2D and 3D coordinates 
        repsectively. if left as None, it will behave exactly like rdf.
    dr : float
        Delta r. Determines the spacing between successive radii over which g(r)
        is computed.
    rho : float, optional
        Number density. If left as None, box dimensions will be inferred from 
        the particles and the number density will be calculated accordingly.
    rcutoff : float
        radii cutoff value between 0 and 1. The default value of 0.9 means the 
        independent variable (radius) over which the RDF is computed will range 
        from 0 to 0.9*r_max. This removes the noise that occurs at r values 
        close to r_max, due to fewer valid particles available to compute the 
        RDF from at these r values.
    eps : float, optional
        Epsilon value used to find particles less than or equal to a distance 
        in KDTree.
    parallel : bool, optional
        Option to enable or disable multiprocessing. Enabling this affords 
        significant increases in speed.
    progress : bool, optional
        Set to False to disable progress readout (only valid when 
        parallel=False).
        
    
    Returns
    -------
    g_r : (n_radii) np.array
        radial distribution function values g(r).
    radii : (n_radii) np.array
        radii over which g(r) is computed
    """

    if not isinstance(particles_center, np.ndarray):
        particles_center = np.array(particles_center)
    # assert particles array is correct shape
    shape_err_msg = 'particles_center should be an array of shape N x d, where N is \
                     the number of particles and d is the number of dimensions.'
    assert len(particles_center.shape) == 2, shape_err_msg
    # assert particle coords are 2 or 3 dimensional
    assert particles_center.shape[-1] in [2, 3], 'RDF can only be computed in 2 or 3 \
                                           dimensions.'

    if particles_to_check == None:
        particles_to_check = particles_center
    else:
        if not isinstance(particles_to_check, np.ndarray):
            particles_to_check = np.array(particles_to_check)
        # assert particles array is correct shape
        shape_err_msg = 'particles_to_check should be an array of shape N x d, where N is \
                        the number of particles and d is the number of dimensions.'
        assert len(particles_to_check.shape) == 2, shape_err_msg
        # assert particle coords are 2 or 3 dimensional
        assert particles_to_check.shape[-1] in [2, 3], 'RDF can only be computed in 2 or 3 \
                                            dimensions.'
    
    if not particles_center.shape == particles_to_check.shape:
        shape_err_msg = 'It is not possible to calculate RDF with  \
                        particles of differing dimensions.'
        assert not particles_center.shape == particles_to_check.shape, shape_err_msg

    start = time.time()

    #create a concatenated particles array to calc the correct box dimensions
    particles =  np.unique(np.concatenate((particles_center, particles_to_check), axis=0), axis=0)

    #calc max and min Dimension 
    mins = np.min(particles_center, axis=0)
    maxs = np.max(particles_center, axis=0)
    # translate particles such that the particle with min coords are closest to origin
    particles -= mins
    particles_center -= mins
    particles_to_check -= mins
    # dimensions of box
    dims = maxs - mins
    
    r_max = (np.min(dims) / 2)*rcutoff
    radii = np.arange(dr, r_max, dr)

    N, d = particles.shape
    if not rho:
        rho = N / np.prod(dims) # number density
    
    # create a KDTree for fast nearest-neighbor lookup of particles
    tree_center = cKDTree(particles_center)
    tree_to_check = cKDTree(particles_to_check)

    if parallel:
        N_radii = len(radii)
        radii_and_indices = np.stack([np.arange(N_radii), radii], axis=1)
        radii_splits = np.array_split(radii_and_indices, os.cpu_count(), axis=0)
        values = [(radii_splits[i], tree_center, particles_center, tree_to_check, particles_to_check, mins, maxs, N_radii, dr, eps, rho) for i in range(len(radii_splits))]
        with Pool() as pool:
            results = pool.starmap(paralell_hist_loop, values)
        g_r = np.sum(results, axis=0)
    else:
        g_r = np.zeros(shape=(len(radii)))
        for r_idx, r in enumerate(radii):
            # find all particles that are at least r + dr away from the edges of the box
            valid_idxs = np.bitwise_and.reduce([(particles_center[:, i]-(r+dr) >= mins[i]) & (particles_center[:, i]+(r+dr) <= maxs[i]) for i in range(d)])
            valid_particles_center = particles_center[valid_idxs]
            valid_idxs = np.bitwise_and.reduce([(particles_to_check[:, i]-(r+dr) >= mins[i]) & (particles_to_check[:, i]+(r+dr) <= maxs[i]) for i in range(d)])
            valid_particles_to_check = particles_to_check[valid_idxs]
            
            # compute n_i(r) for valid particles.
            for particle in valid_particles_center:
                n = tree_to_check.query_ball_point(particle, r+dr-eps, return_length=True) - tree_to_check.query_ball_point(particle, r, return_length=True)
                g_r[r_idx] += n
            
            # normalize
            valid_particles = np.unique(np.concatenate((valid_particles_center, valid_particles_to_check), axis=0), axis=0)
            n_valid = len(valid_particles)
            shell_vol = (4/3)*np.pi*((r+dr)**3 - r**3) if d == 3 else np.pi*((r+dr)**2 - r**2)
            g_r[r_idx] /= n_valid*shell_vol*rho

            if progress:
                print('Computing RDF     Radius {}/{}    Time elapsed: {:.3f} s'.format(r_idx+1, len(radii), time.time()-start), end='\r', flush=True)

    return g_r, radii
