import yt
import numpy as np
import pandas as pd
from astropy.table import Table
from unyt import unyt_quantity, unyt_array
from yt.utilities.logger import ytLogger
from uncertainties import ufloat

import concurrent.futures

from src import explorer as dge
from src.explorer.class_methods import random_vector_spherical

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
files = ["ART_satellitesV5_Rvir1.0.csv", "ART_satellitesV5_Rvir1.5.csv", "ART_satellitesV5_Rvir2.0.csv", "ART_satellitesV5_Rvir3.0.csv", "ART_satellitesV5_Rvir4.0.csv", "ART_satellitesV5_Rvir5.0.csv"]

def analyzer(f):
    print(f"Analyzing {f}")
    tdir = "/home/asier/StructuralAnalysis/satellite_tables/"
    errors = []    
    snapequiv = load_ftable("./ART_equivalence.dat")
    
    file = tdir + f
    
    candidates = pd.read_csv(file)
    
    candidates['stellar_mass'] = pd.Series()
    candidates['dark_mass'] = pd.Series()
    candidates['gas_mass'] = pd.Series()

    candidates['rh3D_stars_physical'] = pd.Series()
    candidates['rh3D_dm_physical'] = pd.Series()
    candidates['rh3D_gas_physical'] = pd.Series()

    candidates['rh_stars_physical'] = pd.Series()
    candidates['rh_dm_physical'] = pd.Series()
    candidates['e_rh_stars_physical'] = pd.Series()
    candidates['e_rh_dm_physical'] = pd.Series()
    


    candidates['sigma*'] = pd.Series()
    candidates['e_sigma*'] = pd.Series()

    candidates['Mhl'] = pd.Series()

    candidates['Mdyn'] = pd.Series()
    candidates['e_Mdyn'] = pd.Series()

    candidates['delta_rel'] = pd.Series()

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

        if halo_instance.stars.masses.sum() == 0:
            errors.append(f"Sub_tree_id {halo['Sub_tree_id']} in snapshot-{halo['Snapshot']} has no stars at all.")
            continue
            

            
        halo_instance.compute_bound_particles(
            components=["stars", "darkmatter", "gas"], 
            method="bh", 
            weighting="softmax", 
            T="adaptative", 
            verbose=False,  
            cm=unyt_array(*halocen), 
            vcm=unyt_array(*halovel)
        )

        if halo_instance.stars.bmasses.sum() == 0:
            errors.append(f"Sub_tree_id {halo['Sub_tree_id']} in snapshot-{halo['Snapshot']} has no stars bound.")
            continue
                    
        halo_instance.darkmatter.refined_center6d(method="adaptative", nmin=100)
        halo_instance.stars.refined_center6d(method="adaptative", nmin=40)
        halo_instance.gas.cm, halo_instance.gas.vcm = halo_instance.darkmatter.cm, halo_instance.darkmatter.vcm



        
        lines_of_sight = random_vector_spherical(N=16, half_sphere=False)

        
        candidates.loc[index, "stellar_mass"] =  halo_instance.stars.bmasses.sum().in_units("Msun").value
        candidates.loc[index, "dark_mass"] =  halo_instance.darkmatter.bmasses.sum().in_units("Msun").value
        candidates.loc[index, "gas_mass"] =  halo_instance.gas.bmasses.sum().in_units("Msun").value

        try:
            candidates.loc[index, "delta_rel"] = float(halo_instance.stars.delta_rel)
        except:
            candidates.loc[index, "delta_rel"] = 0
    
        
    
        candidates.loc[index, 'rh3D_stars_physical'] = halo_instance.stars.half_mass_radius(only_bound=True).in_units("kpc").value
        candidates.loc[index, 'rh3D_dm_physical'] = halo_instance.darkmatter.half_mass_radius(only_bound=True).in_units("kpc").value
        candidates.loc[index, 'rh3D_gas_physical'] = halo_instance.gas.half_mass_radius(only_bound=True).in_units("kpc").value
        
        candidates.loc[index, 'Mhl'] = halo_instance.Mhl.in_units("Msun").value
    
        rh_dm = []
        rh_st = []
        sigma = []
        mdyn = []
        for los in lines_of_sight:
            halo_instance.set_line_of_sight(los.tolist())

            tmp_st = halo_instance.stars.half_mass_radius(project=True, only_bound=True)
            tmp_dm = halo_instance.darkmatter.half_mass_radius(project=True, only_bound=True)
            tmp_sig = halo_instance.stars.los_dispersion()
            
            rh_st.append(tmp_st)
            rh_dm.append(tmp_dm)
            sigma.append(tmp_sig)
            
            mdyn.append(580 * 1E3 * tmp_st.to("kpc") * tmp_sig.to("km/s")**2)

        
        halo_instance.set_line_of_sight([1,0,0])

    
        candidates.loc[index, "rh_stars_physical"] = np.mean(rh_st)
        candidates.loc[index, "rh_dm_physical"] = np.mean(rh_dm)
        candidates.loc[index, "e_rh_stars_physical"] = np.std(rh_st)
        candidates.loc[index, "e_rh_dm_physical"] = np.std(rh_dm)
        
        candidates.loc[index, "sigma*"] = np.mean(sigma)
        candidates.loc[index, "e_sigma*"] = np.std(sigma)

        candidates.loc[index, "Mdyn"] = np.mean(mdyn)
        candidates.loc[index, "e_Mdyn"] = np.std(mdyn)

        
        print(halo_instance.info())
        
    
    candidates.to_csv(file, index=False)
    return f"{file} processed", errors
        
    


 	 	 	 	 	 	




with concurrent.futures.ProcessPoolExecutor(max_workers=6) as executor:
    results = list(executor.map(analyzer, files))

print(f"\n\nWarnings occurred during execution:")
print(results,"\n")
print(f"Finished. Bye!")










