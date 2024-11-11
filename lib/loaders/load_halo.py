import yt
import numpy as np
import pandas as pd
from unyt import unyt_quantity, unyt_array

def load_halo_rockstar(
                    catalogue,
                    snapequiv,
                    path_to_sim,
                    snapnum = None,
                    subtree = None,
                    haloid = None,
                    uid = None,
                    **kwargs
                    ):
    """Loads the specified halo: it returns both the Merger catalogue row corresponding to said halo and a YT dataset
    of the region the halo occupies in the simulation. Various arguments can be passed as kwargs.

    Units are handled with YT.
    
    Parameters
    ----------
    catalogue : pd.DataFrame, required
        Pandas DataFrame containing the Rockstar + Consistent Trees Merger Tree catalogue. List of fields can be passed as kwarg.
        
    snapequiv : pd.DataFrame/astropy.table/similar, required
        Table containing equivalences between snapshot number (i.e. first, second, thenth etc. snapshot of the sim-run), and snapshot name.
        It is used to identify from which snapshot to load the region occupied by the halo.
        
    path_to_sim : string, required
        Path where the output of the simulation, i.e. the snapshots, are.

    snapnum : int or float, optional BUT REQUIRED if loading with subtree or haloid
        Snapshot number of the snapshot where the halo is located.

    subtree, haloid, uid : int or float, optional but AT LEAST ONE is required
        Subtree; uid of the oldest progenitor of the branch identified in the merger tree, haloid; unique halo id at a specific snapshot and 
        universal id of the halo. Only one is required. IF LOADING WITH SUBTREE OR HALOID, the snapshot number is required.

    **kwargs: dictionary of keywords that can be passed when loading the halo. They are the following:

        · catalogue_units : dict
            Default are Rockstar units: mass : Msun, time : Gyr, length : kpccm, and velocity : km/s.

        · max_radius : (float, string)
                Max radius to cut out from the simulation and return to the user. The string provides the units.
        
        · catalogue_fields : dict of dicts
            · id_fields : dict
                Default are Rockstar fields: subtree : Sub_tree_id, uid : uid, haloid : Halo_ID, snapnum : Snapshot.
                
            · position_fields : dict
                Defaults are Rockstar fields: coord_i : position_i. Position of the center of the Halo.
    
            · radii_fields : dict
                Defaults are scale_radius and virial_radius. For Scale and Virial NFW radii.
    
        · snapequiv_fields : dict
            Only concerns snapnum and snapshot name columns. Defaults are "snapid" and "snapshot", respectively.
        

    Returns
    -------
    halo : pd.DataFrame

    sp : yt.region.sphere
    """
    catalogue_units = {
        'mass': "Msun",
        'time': "Gyr",
        'length':"kpccm",
        'velocity': "km/s"
    }
    catalogue_fields = {
        'id_fields': {'haloid': "Halo_ID", 'uid': "uid", 'snapnum': "Snapshot", 'subtree': "Sub_tree_id"},
        'position_fields': {'position_x': "position_x", 'position_y': "position_y", 'position_z': "position_z"},
        'radii_fields': {'rs': "scale_radius", 'rvir': "virial_radius"}
                       }
    snapequiv_fields = {
        'snapnum': "snapid",
        'snapname': "snapshot"
    }
    print(snapequiv_fields)

    #TBD poder cambiarlos.

    idfields = catalogue_fields['id_fields']
    posfields = list(catalogue_fields['position_fields'].values())
    if snapnum is None:
        if uid is not None:
            halo = catalogue[catalogue[idfields['uid']] == uid]
            snapnum = halo[idfields['snapnum']].values[0]
        else:
            raise Exception("SNAPNUM not provided!!")

    else:
        if uid is not None:
            halo = catalogue[catalogue[idfields['uid']] == uid]
        if subtree is not None:
            halo = catalogue[(catalogue[idfields['subtree']] == subtree) & (catalogue[idfields['snapnum']] == snapnum)]
        if haloid is not None:
            halo = catalogue[(catalogue[idfields['haloid']] == haloid) & (catalogue[idfields['snapnum']] == snapnum)]

    file = snapequiv[snapequiv[snapequiv_fields['snapnum']] == snapnum][snapequiv_fields['snapname']].value[0]
    
    ds = yt.load(path_to_sim + f"/{file}")

    halocen = halo[posfields].values[0]
    halovir = halo[catalogue_fields['radii_fields']['rvir']].values[0] 

    if 'max_radius' in kwargs.keys(): sp = ds.sphere( (halocen, catalogue_units['length']), kwargs['max_radius'] )
    else: sp = ds.sphere( (halocen, catalogue_units['length']), (halovir, catalogue_units['length']) )


    return halo, sp

 




            