import os
import yt
import ytree
import warnings
import numpy as np
import pandas as pd
from tqdm import tqdm
from unyt import unyt_quantity

from copy import copy, deepcopy

from ..galex.class_methods import compute_stars_in_halo, load_ftable


class MergerTree:
    """Easy manage of merger trees, with the specific purpose of extracting galaxies with desired parameters and qualities. Although
    it may do more things than just that depending on what the stage you catch me -the author-, in.

    For now, it only accepts consistent-trees output.
    """
    def __init__(self,
                 fn,
                 arbor=None
                ):
        """Initialization function.
        """
        self.fn = fn
        self.arbor = self._load_ytree_tree() if arbor is None else arbor
        self.size = self.arbor.size
        self.all_fields = self.arbor.field_list
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
        self.min_halo_mass = 1E7

    
    def _load_ytree_tree(self):
        """Deletes arbor in temporary_arbor, if existing, and creates a new one for use in current instance.
        """
        sfn = "/".join(self.fn.split("/")[:-1]) + "/arbor/arbor.h5"
        
        if self.fn.endswith("arbor.h5"):
            return ytree.load(self.fn)
            
        elif os.path.exists(sfn):
            warnings.warn("h5 formatted arbor has been detected in the provided folder", UserWarning)
            return ytree.load(sfn)
            
        else:
            a = ytree.load(self.fn)
            fn = a.save_arbor(filename=sfn)
            return ytree.load(fn)
        
    def _compute_subtreid(self, old_df, maingal=False):
        """Computes subtree id for given merger-tree tree.
        """
        df = deepcopy(old_df)
        
        df['Sub_tree_id'] = np.zeros(len(df))
       
        for snapnum in range(self.snap_min, self.snap_max + 1):
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

        df['R/Rvir'] = pd.Series()
        for snapnum in tqdm(range(self.snap_min, self.snap_max + 1), desc=f"Computing R/Rvir", ncols=200):
            cx = self._PrincipalTree[self._PrincipalTree['Snapshot'] == snapnum]['position_x'].values
            cy = self._PrincipalTree[self._PrincipalTree['Snapshot'] == snapnum]['position_y'].values
            cz = self._PrincipalTree[self._PrincipalTree['Snapshot'] == snapnum]['position_z'].values
            cRvir = self._PrincipalTree[self._PrincipalTree['Snapshot'] == snapnum]['virial_radius'].values
            isnap = df[df['Snapshot'] == snapnum].index
            df.loc[isnap, 'R/Rvir'] = np.sqrt(   (df.loc[isnap, 'position_x'] - cx)**2 + 
                                                 (df.loc[isnap, 'position_y'] - cy)**2 + 
                                                 (df.loc[isnap, 'position_z'] - cz)**2 ) / cRvir
            
        return df

    def _has_stars_or_galaxy(self, CrossingSats_R1, pdir):
        """Finds stars and galaxies in seleted halos.
        """
        snapequiv = deepcopy(self.equivalence_table)
        
        for snapnum in np.unique(CrossingSats_R1['Snapshot'].values):
            filtered_halos  = CrossingSats_R1[(CrossingSats_R1['Snapshot'] == snapnum)]
            fn = snapequiv[snapequiv['snapid'] == snapnum]['snapshot'].values[0]
            ds = yt.load(pdir + fn)
            
            for index in filtered_halos.index:
                filtered_node = filtered_halos.loc[[index]]
                istars, mask_stars, sp, delta_rel = compute_stars_in_halo(filtered_node, ds, verbose=True)
                
                hasstars = np.count_nonzero(mask_stars) != 0
                hassgal = np.count_nonzero(mask_stars) >= 6
                
                CrossingSats_R1.loc[index, 'has_stars'] = hasstars 
                CrossingSats_R1.loc[index, 'has_galaxy'] = hassgal
                CrossingSats_R1.loc[index, 'delta_rel'] = float(delta_rel)
                if hasstars:
                    CrossingSats_R1.loc[index, 'stellar_mass'] = sp['stars','particle_mass'][mask_stars].sum().in_units("Msun").value
        
        
            ds.close()


    


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
            count = 0
            for node in tqdm(list(mytree['tree']), desc=f"Traversing Main Tree", ncols=200):
                for myname, tablename in self.selected_fields.items():  
                    if type(tablename) == tuple:
                        single_tree.loc[count, myname] = node[tablename[0]].to(tablename[1])
                    else:
                        single_tree.loc[count, myname] = node[tablename]
                count += 1
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
        
        if maingal:
            self.snap_min, self.snap_max = int(single_tree_final['Snapshot'].values.min()), int(single_tree_final['Snapshot'].values.max())

        single_tree_final = self._compute_subtreid(single_tree_final)

        if maingal:
            self._PrincipalTree = single_tree_final[single_tree_final['Sub_tree_id'] == 1]
            
        single_tree_final['Halo_at_z0'] = mytree.uid * np.ones_like(single_tree_final['Halo_ID'].values)
        single_tree_final['TreeNum'] =  int(treenum) * np.ones_like(single_tree_final['Halo_ID'].values)

        if self._computing_forest:
            pass
        else:
            single_tree_final = self._compute_R_Rvir(single_tree_final)
                    
        return single_tree_final        


    def construct_df_forest(self):
        """Constructs a data-frame based merger-tree forest for easy access
        """
        self._computing_forest = True
        
        MainTree = self.construc_df_tree(0, maingal=True) #self.
        MainTree = self._compute_R_Rvir(MainTree)

        z0_masses = np.array([tree['mass'].to(self.selected_fields['mass'][1]).value/self.min_halo_mass - 1 for tree in self.arbor[1:]])
        index_above = np.where((z0_masses > 0) == False)[0][0] + 1

        SatelliteTrees_final = pd.DataFrame()
        for sat_index in tqdm(range(1, index_above), desc=f"Traversing Satellite Trees", ncols=200):
            SatTree = self.construc_df_tree(sat_index, maingal=False)
            SatelliteTrees_final = pd.concat([SatelliteTrees_final, SatTree])



        SatelliteTrees_final = self._compute_R_Rvir(SatelliteTrees_final.reset_index(drop=True, inplace=False))

        self.MainTree = MainTree
        self.SatelliteTrees = SatelliteTrees_final
        self.CompleteTree = pd.concat([self.MainTree, self.SatelliteTrees])

        self._computing_forest = False

        return None

    def construct_equivalence_table(self):
        """work in progress
        """
        return None



    def select_halos(self, main_Rvir = 1, extra_Rvirs = None, **constraints):
        """Selects halos according to the provided constraints. Constraints are pased as kwargs. Sub_tree_id == 1 is avoided.

        Parameters
        ----------
        main_Rvir : float
            First radius in which to search. Default: R/Rvir=1.
        extra_Rvirs : array[float]
            Extra R/Rvir in which to search for the halos found at main_Rvir.
        constraints : kwargs
            Constraints to apply at main_Rvir. 

            List:
            ------------
            Redshift : [zlow, zmax]
            mass: [mlow, mhigh]
            Secondary : True or False
            Rvir_tol : 0.2

            stellar_mass : [n*low, n*high] -- number of particles is provided, not stellar_mass
        """
        CompleteTree = deepcopy(self.CompleteTree)
        zlow, zhigh, mlow, mhigh, secondary, st_low, st_high, Rvir_tol = -1, 1E4, -1, 1E20, True, -1, int(1E20), 0.2
        
        if 'Redshift' in constraints.keys():
            zlow, zhigh = constraints['Redshift']
        if 'mass' in constraints.keys():
            mlow, mhigh = constraints['mass']
        if 'Secondary' in constraints.keys():
            secondary = constraints['Secondary']
        if 'stellar_mass' in constraints.keys():
            st_low, st_high = constraints['stellar_mass']        
        if 'Rvir_tol' in constraints.keys():
           Rvir_tol = constraints['Rvir_tol'] 

        print(mlow, mhigh)
        constrainedTree_Rmain = CompleteTree[(np.abs(CompleteTree['R/Rvir'] - main_Rvir) < Rvir_tol) & 
                                             #(mlow <= CompleteTree['mass'])                      & (CompleteTree['mass'] <= mhigh) &
                                             (zlow <= CompleteTree['Redshift'])                  & (CompleteTree['Redshift'] <= zhigh)
                                            ].sort_values(by=['Snapshot', 'Sub_tree_id'], ascending=[True, True])

        #constrainedTree_Rmain['Priority'] = (constrainedTree_Rmain['R/Rvir'] > main_Rvir).astype(int)  # 1 if R/Rvir > 1, else 0
        
        #constrainedTree_Rmain_sorted = constrainedTree_Rmain.sort_values(by=['Sub_tree_id', 'Priority', 'R/Rvir', 'Redshift'], 
                                                                         #ascending=[True, False, True, False]
                                                                        #)
        
        #crossing_haloes_mainRvir = constrainedTree_Rmain_sorted.groupby('Sub_tree_id').first()



        
        return constrainedTree_Rmain
    
    
    
