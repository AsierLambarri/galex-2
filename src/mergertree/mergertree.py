import os
import yt
import ytree
import numpy as np
import pandas as pd
from tqdm import tqdm
from unyt import unyt_quantity

from copy import copy, deepcopy

class MergerTree:
    """Easy manage of merger trees, with the specific purpose of extracting galaxies with desired parameters and qualities. Although
    it may do more things than just that depending on what the stage you catch me -the author-, in.

    For now, it only accepts consistent-trees output.
    """
    def __init__(self,
                 fn
                ):
        """Initialization function.
        """
        self.fn = fn
        self.arbor = self._load_ytree_tree()
        self.size = self.arbor.size
        self.all_fields = self.arbor.field_list
        self.minium_halomass = 1E7
        self.set_fields({
                'Halo_ID': 'halo_id',
                'Snapshot': 'Snap_idx',
                'Redshift': 'redshift',
                'Time': ('time', 'Gyr'),
                'uid': 'uid',
                'desc_uid': 'desc_uid',
                'mass': ('mass', 'Msun'),
                'num_prog': 'num_prog',
                'virial_radius': ('virial_radius', 'kpc'),
                'scale_radius': ('scale_radius', 'kpc'),
                'vrms': ('velocity_dispersion', 'km/s'),
                'vmax': ('vmax', 'km/s'),
                'position_x': ('position_x', 'kpc'),
                'position_y': ('position_y', 'kpc'),
                'position_z': ('position_z', 'kpc'),
                'velocity_x': ('velocity_x', 'km/s'),
                'velocity_y': ('velocity_y', 'km/s'),
                'velocity_z': ('velocity_z', 'km/s'),
                'A[x]': 'A[x]',
                'A[y]': 'A[y]',
                'A[z]': 'A[z]',
                'b_to_a': 'b_to_a',
                'c_to_a': 'c_to_a',
                'T_U': 'T_|U|',
                'Tidal_Force': 'Tidal_Force',
                'Tidal_ID': 'Tidal_ID'
        })


    def _load_ytree_tree(self):
        """Deletes arbor in temporary_arbor, if existing, and creates a new one for use in current instance.
        """
        a = ytree.load(self.fn)
        try:
            b = a.save_arbor()
            return ytree.load(b)
        except:
            ## IMPLEMENT SEARCH OF LAS GOOD TREE AND CUT THE WHOLE FOREST THERE
        
    def _compute_subtreid(self, old_df):
        """Computes subtree id for given merger-tree tree.
        """
        df = deepcopy(old_df)
        
        df['Sub_tree_id'] = np.zeros(len(df))
        for snapnum in range(int(df['Snapshot'].values.min()), int(df['Snapshot'].values.min())):
            Halo_ID_list = np.unique(df[(df['Snapshot']==snapnum)]['uid'])
            if snapnum == 0:
                index = df[(df['uid'].isin(Halo_ID_list))].index
                values = df[(df['uid'].isin(Halo_ID_list))]['uid']
                df.loc[index, 'Sub_tree_id'] = values
            else:
                Existing_halos = Halo_ID_list[np.isin(Halo_ID_list, df['desc_uid'])]
                New_halos = Halo_ID_list[~np.isin(Halo_ID_list, df['desc_uid'])]
                index_existing = df[(df['uid'].isin(Existing_halos))].sort_values('uid').index
                index_new = df[(df['uid'].isin(New_halos))].index
                values_existing = df[(df['desc_uid'].isin(Existing_halos))&
                                                 (df['Secondary']==False)].sort_values('desc_uid')['Sub_tree_id']
                values_new = df[(df['uid'].isin(New_halos))]['uid']
                df.loc[index_existing, 'Sub_tree_id'] = np.array(values_existing)
                df.loc[index_new, 'Sub_tree_id'] = np.array(values_new)

        return df
        
    def _compute_R_Rvir(self, old_df):
        """Computes R/Rvir for given nodes in in the tree.
        """
        df = deepcopy(old_df)

        df['R/Rvir'] = np.zeros(len(df))
        for snapnum in range(int(df['Snapshot'].values.min())):
            cx = self.MainTree[self.MainTree['Snapshot'] == snapnum].sort_values('mass')['position_x'].iloc[-1]
            cy = self.MainTree[self.MainTree['Snapshot'] == snapnum].sort_values('mass')['position_y'].iloc[-1]
            cz = self.MainTree[self.MainTree['Snapshot'] == snapnum].sort_values('mass')['position_z'].iloc[-1]
            cRvir = self.MainTree[self.MainTree['Snapshot'] == snapnum].sort_values('mass')['virial_radius'].iloc[-1]
            isnap = df[df['Snapshot'] == snapnum].index
            df.loc[isnap, 'R/Rvir'] = np.sqrt(   (df.loc[isnap, 'position_x'] - cx)**2 + 
                                                 (df.loc[isnap, 'position_y'] - cy)**2 + 
                                                 (df.loc[isnap, 'position_z'] - cz)**2 ) / cRvir
            
        return df




    def set_fields(self, fields_dict):
        """Sets the fields to be saved into a df
        """
        self.selected_fields = fields_dict
        return None

    def construc_df_tree(self, treenum, maingal = False):
        """Constructs merger-tree for a single tree.

        OPTIONAL parameters
        -------------------
        treenum : int
            Tree/arbor number inside the forest.s
        """
        mytree = self.arbor[treenum]
                
        Data = np.zeros((mytree.tree_size, len(self.selected_fields)))
        single_tree = pd.DataFrame(Data, columns = self.selected_fields.keys())

        if maingal:
            for count, node in enumerate(tqdm(mytree['tree'], desc=f"Traversing tree number {treenum}", unit=" nodes", ncols=200)):
                for myname, tablename in self.selected_fields.items():  
                    if type(tablename) == tuple:
                        single_tree.loc[count, myname] = node[tablename[0]].to(tablename[1])
                    else:
                        single_tree.loc[count, myname] = node[tablename]

        else:
            for count, node in enumerate(mytree['tree']):
                for myname, tablename in self.selected_fields.items():  
                    if type(tablename) == tuple:
                        single_tree.loc[count, myname] = node[tablename[0]].to(tablename[1])
                    else:
                        single_tree.loc[count, myname] = node[tablename]


        single_tree_final = single_tree.sort_values(['mass', 'Snapshot'], ascending = (False, True))
        single_tree_final['Secondary'] = single_tree_final.duplicated(['desc_uid', 'Snapshot'], keep='first')
        single_tree_final.sort_values('Snapshot')

        single_tree_final = self._compute_subtreid(single_tree_final)
        if maingal:
            single_tree_final['R/Rvir'] = 0 * np.ones_like(single_tree_final['Halo_ID'].values)
        else:
            single_tree_final = self._compute_R_Rvir(single_tree_final)

        single_tree_final['Halo_at_z0'] = mytree.uid * np.ones_like(single_tree_final['Halo_ID'].values)

        return single_tree_final        


    def construct_df_forest(self):
        """Constructs a data-frame based merger-tree forest for easy access
        """
        self.MainTree = self.construc_df_tree(0, maingal=True)

        z0_masses = np.array([tree['mass'].value/1E7 - 1 for tree in self.arbor[1:]])
        index_above = np.where((z0_masses > 0) == False)[0][0] + 1

        SatelliteTrees_final = pd.DataFrame()
        for sat_index in tqdm(range(1, index_above), desc=f"Traversing satellite trees", ncols=300):
            SatTree = self.construc_df_tree(sat_index, maingal=False)
            SatelliteTrees_final = pd.concat([SatelliteTrees_final, SatTree])

        self.SatelliteTrees = SatelliteTrees_final
        self.CompleteTree = pd.concat([self.MainTree, self.SatelliteTrees])

        return None

    def construct_equivalence_table(self):
        """
        """
        return None



    def select_halos(self, constraints):
        """
        """
        return None
    
    
    
