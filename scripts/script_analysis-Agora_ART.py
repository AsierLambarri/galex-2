import yt
import numpy as np
import pandas as pd
from astropy.table import Table
from unyt import unyt_quantity, unyt_array
from yt.utilities.logger import ytLogger

import concurrent.futures

from src import galex as dge
from src.galex.class_methods import random_vector_spherical

ytLogger.setLevel(30)

def load_ftable(fn):
    """Loads astropy tables formated with ascii.fixed_width and sep='\t'. These tables are human readable but
    a bit anoying to read with astropy because of the necessary keyword arguments. This functions solves that.
    Useful for tables that need to be accessed in scripts but be readable (e.g. csv are NOT human readable).

    Equivalent to : Table.read(fn, format="ascii.fixed_width", delimiter='\t')

    Parameters
    ----------
    fn : string, required
        Path to Formated Table file.

    Returns
    -------
    ftable : pd.DataFrame
    """
    return Table.read(fn, format="ascii.fixed_width", delimiter="\t").to_pandas()



dge.config.code = "ART"
files = ["ART_satellitesV5_Rvir1.csv", "ART_satellitesV5_Rvir1.5.csv", "ART_satellitesV5_Rvir2.0.csv", "ART_satellitesV5_Rvir3.0.csv", "ART_satellitesV5_Rvir4.0.csv", "ART_satellitesV5_Rvir5.0.csv"]

def analyzer(f):
    print(f"Analyzing {f}")
    tdir = "/home/asier/StructuralAnalysis/satellite_tables/"
    
    snapequiv = load_ftable("./ART_equivalence.dat")
    
    file = tdir + f
    
    candidates = pd.read_csv(file)
    
    candidates['delta_rel'] = pd.Series()
    candidates['stellar_mass'] = pd.Series()
    
    candidates['rh_stars_physical'] = pd.Series()
    candidates['rh_dm_physical'] = pd.Series()
    candidates['e_rh_stars_physical'] = pd.Series()
    candidates['e_rh_dm_physical'] = pd.Series()
    
    candidates['rh3D_stars_physical'] = pd.Series()
    candidates['rh3D_dm_physical'] = pd.Series()
    
    candidates['Mhl'] = pd.Series()
    
    candidates['sigma*'] = pd.Series()
    candidates['e_sigma*'] = pd.Series()
    
    for index, halo in candidates.iterrows():
        try:
            fn = snapequiv[snapequiv['snapshot'] == halo['Snapshot']]['snapname'].values[0]
            z = halo['Redshift']
        
            halocen = ([halo['position_x']/(1+z), halo['position_y']/(1+z), halo['position_z']/(1+z)], 'kpc')
            halovel = ([halo['velocity_x'], halo['velocity_y'], halo['velocity_z']], 'km/s')
            
            rvir = (halo['virial_radius'] / (1+z), 'kpc')
            rs = (halo['scale_radius'] / (1+z), 'kpc')
            vmax = (halo['vmax'], 'km/s')
            vrms = (halo['vrms'], 'km/s')
    
            print(f"#####   {halo['Sub_tree_id']}:snapshot-{halo['Snapshot']}   #####\n")
            halo_instance = dge.SnapshotHalo("/media/asier/EXTERNAL_USBA/Cosmo_v18/" + fn, center=halocen, radius=rvir,
                                  dm_params={'rockstar_center': halocen, 'rockstar_vel': halovel, 'rvir': rvir, 'rs': rs, 'vmax': vmax, 'vrms': vrms},
                                  stars_params={'ML': (3, 'Msun/Lsun')}
                                 )
        except:
            continue
            
        lines_of_sight = random_vector_spherical(N=16)
    
        
    
        halo_instance.stars.compute_stars_in_halo(verbose=False)
        candidates.loc[index, "stellar_mass"] =  halo_instance.stars.bmasses.sum().in_units("Msun").value
        candidates.loc[index, "delta_rel"] = float(halo_instance.stars.delta_rel)
        halo_instance.stars.refined_center_of_mass(method="hm", mfrac=0.5, delta=1E-2)
    
    
        halo_instance.darkmatter.compute_bound_particles(method="APROX", refine=False)
        dm_mass = halo_instance.darkmatter.bmasses.sum().in_units("Msun").value
        halo_instance.darkmatter.refined_center_of_mass(method="iterative", alpha=0.95, delta=1E-2)
    
    
        candidates.loc[index, 'rh3D_stars_physical'] = halo_instance.stars.half_mass_radius().in_units("kpc").value
        candidates.loc[index, 'rh3D_dm_physical'] = halo_instance.darkmatter.half_mass_radius().in_units("kpc").value
        candidates.loc[index, 'Mhl'] = halo_instance.dynamical_mass().in_units("Msun").value
    
        rh_dm = []
        rh_st = []
        sigma = []
        
        for los in lines_of_sight:
            halo_instance.set_line_of_sight(los.tolist())
            
            rh_st.append(halo_instance.stars.half_mass_radius(project=True))
            rh_dm.append(halo_instance.darkmatter.half_mass_radius(project=True))
    
            sigma.append(halo_instance.stars.los_velocity())
    
        halo_instance.set_line_of_sight([1,0,0])

    
        candidates.loc[index, "rh_stars_physical"] = np.mean(rh_st)
        candidates.loc[index, "rh_dm_physical"] = np.mean(rh_dm)
        candidates.loc[index, "e_rh_stars_physical"] = np.std(rh_st)
        candidates.loc[index, "e_rh_dm_physical"] = np.std(rh_dm)
        candidates.loc[index, "sigma*"] = np.mean(sigma)
        candidates.loc[index, "e_sigma*"] = np.std(sigma)
    
        print(halo_instance.info())
    
    #cols = ['Halo_ID','Snapshot','Redshift','Time','Sub_tree_id','uid','desc_uid','mass','stellar_mass','Mdyn','virial_radius','scale_radius','rh3D_stars_physical','rh3D_dm_physical', 'sigma*','e_sigma*', 
    #        'rh_stars_physical','rh_dm_physical',
    #        'e_rh_stars_physical','e_rh_dm_physical','R/Rvir','vrms','vmax','position_x','position_y','position_z','velocity_x','velocity_y','velocity_z',
    #        'A[x]','A[y]','A[z]','b_to_a','c_to_a','T_U','Tidal_Force','Tidal_ID','Secondary','Halo_at_z0','has_stars','has_galaxy','delta_rel','num_prog'
    #       ]
    
    
    candidates.to_csv(file, index=False)
    return f"{file} processed"
        
    


 	 	 	 	 	 	




with concurrent.futures.ProcessPoolExecutor(max_workers=6) as executor:
    results = list(executor.map(analyzer, files))

print(results)
print(f"Finished")










