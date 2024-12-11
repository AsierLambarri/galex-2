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







def __getattr__(self, field_name):
    """Dynamical loader for accessing fields.
    """
    
    if field_name  in self._dynamic_fields.keys():
        component_data = [
            self.stars.__getattr__(self._dynamic_fields[field_name]),
            self.gas.__getattr__(self._dynamic_fields[field_name]),
            self.darkmatter.__getattr__(self._dynamic_fields[field_name])
        ]
        #units = component_data[0].units
        self.particle_types = np.concatenate((
            np.full(len(component_data[0]),"stars"),
            np.full(len(component_data[1]),"gas"),
            np.full(len(component_data[2]),"darkmatter")
        ))
        return np.concatenate((
            component_data[0],
            component_data[1],
            component_data[2]
        )) 
            

    try:
        return self.__getattribute__(field_name)
    except AttributeError:
        raise AttributeError(f"Field '{field_name}' not found for particle type {self.ptype}. "+ f"Available fields are: {list(self._dynamic_fields.keys())}")
    
    AttributeError(f"Field {field_name} not found for particle type {self.ptype}. Available fields are: {list(self._dynamic_fields.keys())}")
    return None

