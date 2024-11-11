import os
import yt
import sys
import yaml
import shutil
import argparse
import numpy as np
import pandas as pd
from tqdm import tqdm
from limepy import limepy
from copy import copy, deepcopy
from astropy.table import Table
from unyt import unyt_array, unyt_quantity
import matplotlib.pyplot as plt
import smplotlib
import pprint
pp = pprint.PrettyPrinter(depth=4, width=10000)


from lib.loaders import load_ftable, load_halo_rockstar
from lib.mergertrees import compute_stars_in_halo, bound_particlesBH, bound_particlesAPROX
from lib.galan import half_mass_radius, density_profile, velocity_profile, refine_center, NFWc, random_vector_spherical, LOS_velocity

def clear_directory(directory_path):
    if not os.path.exists(directory_path):
        raise Exception("Directory does not exist.")
        
    for item in os.listdir(directory_path):
        item_path = os.path.join(directory_path, item)
        
        if os.path.isfile(item_path) or os.path.islink(item_path):
            os.remove(item_path)  
        elif os.path.isdir(item_path):
            shutil.rmtree(item_path)

    return None


def isolate_sequences(snap_numbers, index_list):
    differences = np.diff(snap_numbers)
    
    jump_indices = np.where(differences != 1)[0]
    
    sequences = []
    indices = []
    start_idx = 0
    for idx in jump_indices:
        sequences.append(snap_numbers[start_idx:idx + 1])
        indices.append(index_list[start_idx:idx + 1])
        start_idx = idx + 1
    
    sequences.append(snap_numbers[start_idx:])
    indices.append(index_list[start_idx:])
    
    return sequences, indices





parser = argparse.ArgumentParser(description='This script extracts suitable candidate galaxy information from a given simulation+Rockstar-ConsistentTrees dataset. The script finds candidate satelite galaxies as those at R/Rvir=1, and fitting the redshift and DM-mass criteria specified in the config yaml file.')
    
parser.add_argument('file', metavar='file.yml', type=str, help='Path to config yaml file. It must contain following entries: -dir_snapshots: snapshot directory, -RCT_paths: path RCT files for main and satellite trees, -snapequiv_path: path to equivalence between RCt snapshot numbers and snapshot ids. More information on the example config.yaml file that is provided.')
parser.add_argument('--verbose', action='store_true', help='Display while running for debugging or later visual inspection.')


args = parser.parse_args()






with open(args.file, "r") as file:
    config_yaml = yaml.safe_load(file)

code = config_yaml['code']
outdir = config_yaml['output_dir'] + "_" + code
if os.path.isdir(outdir):
    clear_directory(outdir)
else:
    os.mkdir(outdir)

os.mkdir(outdir + "/particle_data")


pdir = config_yaml['dir_snapshots']

snapequiv = load_ftable(config_yaml['snapequiv_path'])

MainTree = pd.read_csv(config_yaml['RCT_paths']['main'])
CompleteTree = deepcopy(MainTree)
for main_or_sat, path in config_yaml['RCT_paths'].items():
    if main_or_sat != 'main':   
        sat = pd.read_csv(path)
        CompleteTree = pd.concat([CompleteTree, sat])


mlow, mhigh = (
    float(config_yaml['constraints']['mlow']),
    float(config_yaml['constraints']['mhigh']) if str(config_yaml['constraints']['mhigh']) != "inf" else np.inf
)
stellar_low, stellar_high = (
    float(config_yaml['constraints']['stellar_low']),
    float(config_yaml['constraints']['stellar_high']) if str(config_yaml['constraints']['stellar_high']) != "inf" else np.inf
)
zlow, zhigh = (
    float(config_yaml['constraints']['zlow']),
    float(config_yaml['constraints']['zhigh']) if str(config_yaml['constraints']['zhigh']) != "inf" else np.inf
)
Rvir_extra = list([ R for R in config_yaml['Rvir'] if R != 1])

#constrainedTree = CompleteTree[(zlow <= CompleteTree['Redshift']) & (CompleteTree['Redshift'] <= zhigh) & 
#                               (mlow <= CompleteTree['mass']) & (CompleteTree['mass'] <= mhigh)].sort_values(by=['Sub_tree_id'], ascending=[True])


constrainedTree_R1 = CompleteTree[(np.abs(CompleteTree['R/Rvir'] - 1) < 0.10) & (mlow <= CompleteTree['mass']) & (CompleteTree['mass'] <= mhigh) &
                                  (zlow <= CompleteTree['Redshift']) & (CompleteTree['Redshift'] <= zhigh)].sort_values(by=['Snapshot', 'Sub_tree_id'], ascending=[True, True])

unique_subtrees = np.unique(constrainedTree_R1['Sub_tree_id'])

CrossingSats_R1 = pd.DataFrame(columns=constrainedTree_R1.columns)

verbose = False
for subtree in tqdm(unique_subtrees):
    crossing_subtree = constrainedTree_R1[constrainedTree_R1['Sub_tree_id'] == subtree]
    crossing_subtree.sort_values("Snapshot")

    crossings_list, indexes_list = isolate_sequences(crossing_subtree['Snapshot'].values, crossing_subtree.index)
    
    cross_redshifts = []
    cross_indexes = []
    for cross_snapnums, indexes in zip(crossings_list, indexes_list):
        crossing_subtree_subseq = crossing_subtree.loc[indexes]
        delta_RRvir = crossing_subtree_subseq['R/Rvir'].values - 1
        redshifts = crossing_subtree_subseq['Redshift'].values


        if True in (delta_RRvir > 0):
            RRvir_plus = delta_RRvir[delta_RRvir > 0]
            RRvir_plus_minimum = RRvir_plus.min()
            ipos_min = np.where(delta_RRvir == RRvir_plus_minimum)[0][0]       
            cross_redshifts.append(redshifts[ipos_min])
            cross_indexes.append(indexes[ipos_min])
        else:
            pass
    
    crossings_df = crossing_subtree.loc[cross_indexes]
    CrossingSats_R1 = pd.concat([CrossingSats_R1, crossings_df])
    
    if verbose:
        print(f"\nSUBTREE: {subtree}. First Crossing Redshift: {cross_redshifts}, {cross_indexes}")
        pp.pprint(crossing_subtree[['Halo_ID','Snapshot','Redshift','uid','mass','R/Rvir']])
        print(crossings_list, indexes_list)
        print(crossing_subtree.loc[cross_indexes][['Halo_ID','Snapshot','Redshift','uid','mass','R/Rvir']])


CrossingSats_R1['has_stars'] = pd.Series()
CrossingSats_R1['has_galaxy'] = pd.Series()
CrossingSats_R1['delta_rel'] = pd.Series()
CrossingSats_R1['stellar_mass'] = pd.Series()


for snapnum in np.unique(CrossingSats_R1['Snapshot'].values):
    filtered_halos  = CrossingSats_R1[(CrossingSats_R1['Snapshot'] == snapnum)]
    fn = snapequiv[snapequiv['snapid'] == snapnum]['snapshot'].value[0]
    ds = yt.load(pdir + fn)
    
    print(f"\n##########  {snapnum}  ##########")
    
    for index in filtered_halos.index:
        filtered_node = filtered_halos.loc[[index]]
        istars, mask_stars, sp, delta_rel = compute_stars_in_halo(filtered_node, ds, verbose=True)
        
        hasstars = np.count_nonzero(mask_stars) != 0
        hassgal = np.count_nonzero(mask_stars) >= 6
        
        CrossingSats_R1['has_stars'].loc[index] = hasstars 
        CrossingSats_R1['has_galaxy'].loc[index] = hassgal
        CrossingSats_R1['delta_rel'].loc[index] = float(delta_rel)
        if hasstars:
            print(sp['stars','particle_mass'][mask_stars].sum().in_units("Msun").value)
            CrossingSats_R1['stellar_mass'].loc[index] = sp['stars','particle_mass'][mask_stars].sum().in_units("Msun").value

        if hassgal:
            np.savetxt(outdir + "/" + f"particle_data/stars_{int(filtered_node['Sub_tree_id'].values)}.{int(snapnum)}.pids", istars, fmt="%i")            



        print(f"\n uid: {filtered_node['uid'].values}, subtree: {filtered_node['Sub_tree_id'].values}.")
        print(f"Stars found: {hasstars}.")
        print(f"Galaxies found: {hassgal}. Np: {np.count_nonzero(mask_stars)}.")  




CrossingSats_R1.to_csv(outdir + "/" + f"candidate_satellites_{code}.csv", index=False)

CrossingSats_R1 = CrossingSats_R1[CrossingSats_R1['has_galaxy']]


arrakihs_v2 = CrossingSats_R1[ (CrossingSats_R1['stellar_mass'] <= stellar_high) & (stellar_low <= CrossingSats_R1['stellar_mass']) ].sort_values("Redshift", ascending=False)



arrakihs_v2 = arrakihs_v2[~arrakihs_v2.duplicated(['Sub_tree_id'], keep='first').values].sort_values("Sub_tree_id")

arrakihs_v2.to_csv(outdir + "/" + f"ARRAKIHS_Infall_CosmoV18_{code}.csv", index=False)


arrakihs_v2['rh_stars_physical'] = pd.Series()
arrakihs_v2['rh_dm_physical'] = pd.Series()
arrakihs_v2['Mdyn'] = pd.Series()

for uid in arrakihs_v2['uid'].values:
    halo, sp = load_halo_rockstar(arrakihs_v2, snapequiv, pdir, uid=uid)
    halocen = halo[['position_x','position_y','position_z']].values[0] * sp.units.kpccm

    subid = int(halo['Sub_tree_id'].value[0])
    snap = int(halo['Snapshot'].value[0])

    st_pos = sp['darkmatter','particle_position'].in_units("kpc")
    st_vel = sp['darkmatter','particle_velocity'].in_units("km/s")
    st_mass = sp['darkmatter','particle_mass'].in_units("Msun")
    st_ids = sp['darkmatter', 'particle_index'].value

    dm_pos = sp['darkmatter','particle_position'].in_units("kpc")
    dm_vel = sp['darkmatter','particle_velocity'].in_units("km/s")
    dm_mass = sp['darkmatter','particle_mass'].in_units("Msun")
    dm_ids = sp['darkmatter', 'particle_index'].value

    bound, most_bound, iid, bound_iid = bound_particlesAPROX(dm_pos, dm_vel, dm_mass, ids=dm_ids, cm=halocen.in_units("kpc"), vcm=unyt_array(halo[['velocity_x', 'velocity_y', 'velocity_z']].values[0], 'km/s'), refine=False)

    print(f"Total mass: {mass[bound].sum()}.")
    print(f"CM is: {np.average(pos[most_bound], axis=0, weights=mass[most_bound]).in_units('kpccm')}")
    print(f"shoud: {halocen}")
    
    np.savetxt(outdir + "/" + f"particle_data/dm_{int(subid)}.{int(snap)}.pids", np.array([iid, np.isin(iid, bound_iid) * 1]).T, fmt="%i", delimiter=",")            

    st_bids = np.loadtxt(outdir + "/" + f"particle_data/stars_{subid}.{snap}.pids")
    dm_bpts = np.loadtxt(outdir + "/" + f"particle_data/dm_{subid}.{snap}.pids", delimiter=",")
    dm_bids =  dm_pts[:,0]

    dm_mask =  np.isin(dm_ids, dm_bids)
    st_mask =  np.isin(st_ids, st_bids)

    
    dm_rh, dm_center0 = half_mass_radius(dm_pos[dm_mask], dm_mass[dm_mask])
    st_rh, st_center0 = half_mass_radius(st_pos[st_mask], st_mass[st_mask])
    arrakihs_v2['rh_dm_physical'].loc[halo.index] = dm_rh.in_units('kpc').value
    arrakihs_v2['rh_stars_physical'].loc[halo.index] = st_rh.in_units('kpc').value

    
    dm_mdynMask = np.linalg.norm(dm_pos[dm_mask] - st_center0, axis=1) <= st_rh
    st_mdynMask = np.linalg.norm(st_pos[st_mask] - st_center0, axis=1) <= st_rh

    dm_mdyn_cont = dm_mass[dm_mask][dm_mdynMask].sum()
    st_mdyn_cont = st_mass[st_mask][st_mdynMask].sum()
    arrakihs_v2['Mdyn'].loc[halo.index] = dm_mdyn_cont + st_mdyn_cont

    print(f"Sub id: {subid}")
    print(f"-" * len(f"Sub id: {subid}" + "\n"))
    print(f"Stellar mass: {st_mass[st_mask].sum():.3e}. rH: {st_rh:.2f}.Half mass:  {st_mass[st_mdynMask].sum():.3e}. Error: {(st_mdyn_cont/st_mass[st_mask].sum() - 0.5):.3e}")
    print(f"DM mass: {dm_mass[dm_mask].sum():.3e}. Rockstar mass: {halo['mass'].value[0]:.3e}. rH: {dm_rh:.2f}. Mass inside st_rH:  {dm_mdyn_cont/dm_mass[dm_mask].sum():.3e}")
    print(f"Mdyn: {(dm_mdyn_cont + st_mdyn_cont):.3e}")




sys.exit()

