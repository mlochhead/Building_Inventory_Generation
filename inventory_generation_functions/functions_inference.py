# Copyright (c) 2025, Meredith Lochhead
# All rights reserved.
#
# This source code is licensed under the BSD 3-Clause License found in the
# LICENSE file in the root directory of this source tree.


import json
import numpy as np 
import geopandas as gpd
import folium
import matplotlib.pyplot as plt
import pandas as pd
from itertools import product

import hazus_rulesets as hazrules



########
def update_res_occ(row):

    # MIXED USE
    if row['OccupancyClass_Best'] in ['RES3M', 'RES1M', 'RES3AM', 'RES3BM', 'RES3CM', 'RES3DM', 'RES3EM', 'RES3FM', 'RES1-2SNBM','RES1-1SNBM','RES1-2SNBM','RES1-2SWBM','RES1-3SNBM','RES1-3SWBM','RES1-SLNBM','RES1-SLWBM']:
        if (row['Units_Best']) <= 1: # Not currently allowing for mixed-use single family residnetial buildings 
            return "RES1"
        elif (row['Units_Best']) == 2: 
            return "RES3AM"
        elif (row['Units_Best'] >= 3) & (row['Units_Best'] <= 4):
            return "RES3BM"
        elif (row['Units_Best'] > 4) & (row['Units_Best'] < 10):
            return "RES3CM"
        elif (row['Units_Best'] >= 10) & (row['Units_Best'] < 20):
            return "RES3DM"
        elif (row['Units_Best'] >= 20) & (row['Units_Best'] <= 50):
            return "RES3EM"
        elif (row['Units_Best'] > 50):
            return "RES3FM"
        else: 
            return 'RES3M'
    
    # NOT MIXED USE 
    if row['OccupancyClass_Best'] in ['RES', 'RES1', 'RES3', 'RES3A', 'RES3D', 'RES3C', 'RES3B', 'RES3E', 'RES3F', 'RES1-2SNB','RES1-1SNB','RES1-2SNB','RES1-2SWB','RES1-3SNB','RES1-3SWB','RES1-SLNB','RES1-SLWB']:
        if (row['Units_Best']) <= 1: 
            return "RES1"
        elif (row['Units_Best']) == 2: 
            return "RES3A"
        elif (row['Units_Best'] >= 3) & (row['Units_Best'] <= 4):
            return "RES3B"
        elif (row['Units_Best'] > 4) & (row['Units_Best'] < 10):
            return "RES3C"
        elif (row['Units_Best'] >= 10) & (row['Units_Best'] < 20):
            return "RES3D"
        elif (row['Units_Best'] >= 20) & (row['Units_Best'] <= 50):
            return "RES3E"
        elif (row['Units_Best'] > 50):
            return "RES3F"
        else:
            return 'RES3'

    else: 
        return row['OccupancyClass_Best']
#######


###### CREATED NEW VERSION OF MODULATE_OCC -- TAKEN FROM BRAILS++ CODE 
def modulate_occ(s, res3ab_to_res1_flag):
    if pd.isna(s) or s == '':
        return s

    # Address cases where details have been added in Occupancy Type
    if "GOV2" in s: 
        return "GOV2"
    elif "EDU1" in s: 
        return "EDU1"
    elif "RES1" in s: 
        return "RES1"
    elif s in ['RES3A','RES3B','RES3AM','RES3BM']: 
        if res3ab_to_res1_flag: 
            return "RES1"
        else: 
            return "RES3"
    elif s in ['RES3C','RES3D','RES3E','RES3F','RES3CM','RES3DM','RES3EM','RES3FM']:
        return "RES3"
    
    # For non-GOV2 and non-EDU1 cases
    elif not s[-1].isdigit():  # Check if the last character is not a digit
        if not s[-2].isdigit():# Check if the second to last character is not a digit (case of RES3CM, for example)
            return s[:-2]  # Remove the last two characters
        else: 
            return s[:-1]  # Remove the last character
    return s  # Return the string unchanged if it ends with a number
########



###### VERSION OF MODULATE OCC THAT SIMPLIFIES INTO NSI OCCUPANCIES 
def simplify_occ(s):
    if pd.isna(s) or s == '':
        return s

    # Address cases where details have been added in Occupancy Type
    if "GOV2" in s: 
        return "GOV2"
    elif "EDU1" in s: 
        return "EDU1"
    
    # Handle mixed use cases
    elif s[-1]=='M':  
        
        return s[:-1]  # Remove the last character
    return s  # Return the string unchanged if it ends with a number
########




###### CREATED NEW VERSION OF MODULATE_WEIGHTS -- TAKEN FROM BRAILS++ CODE 
def modulate_weights(weights, structure_types, region, occ, year_class, height, allow_mh_only_for_res2, no_urm):

        if len(weights)==0:
            # do nothing
            return weights, structure_types

        
        # if not RES2, turn off mobile home
        if allow_mh_only_for_res2:
            if (not (occ=="RES2")) and ('MH' in structure_types):
                # find MH and remove it from the list
                MHidx = np.argmax(structure_types == 'MH')
                #MHidx = structure_types.index('MH')
                if weights[MHidx]>0:
                    structure_types = np.delete(structure_types, MHidx)
                    weights = np.delete(weights, MHidx)
                    weights = weights/np.sum(weights)

        # turn of urm
        if no_urm:
            if "URM" in structure_types:
                URMidx = np.argmax(structure_types == 'URM')
                #URMidx = structure_types.index('URM')
                if weights[URMidx]>0:
                    structure_types = np.delete(structure_types, URMidx)
                    weights = np.delete(weights, URMidx)
                    weights = weights/np.sum(weights)
        
        return weights, structure_types
########



############
def compute_hazus_replacement_cost(inv_mod, hazus_conversion):
    """"
    Compute Hazus replacement cost for each footprint 
    """

    # Create a temporary instance of the inventory for use
    data = inv_mod.copy()

    # # Narrow columns
    data = data[['PlanArea_Best','Stories_Best','OccupancyClass_Best','FootprintID','geometry']]

    # Artifically fill with 1 story for cost calculation
    data['Stories_Best'] = data['Stories_Best'].fillna(1)
    data['SqFt'] = data['PlanArea_Best'] * data['Stories_Best']

    # Module occupancy class to get only simple cases
    data['OccupancyClass_Simple'] = data['OccupancyClass_Best'].apply(simplify_occ)

    # Pull in value per square footage per Hazus 
    data = data.merge(hazus_conversion, left_on='OccupancyClass_Simple', right_on='OccupancyClass', how='left')

    # Fill replacement cost (ESTIMATED - NOT RECOMMENDED FOR REAL USE)
    data["ReplacementCost_Hazus"] = (data["SqFt"] * data["Value"])

    # Simplify
    cost_info = data[['FootprintID','ReplacementCost_Hazus']]

    # Append cost info to results
    inv_mod = inv_mod.merge(cost_info, on = 'FootprintID', how = 'left')

    return inv_mod
########



###### FUNCTION TAKEN FROM BRAILS++ CODE 
def add_features_to_asset(new_prop, strtype_key, structure_types, weights, n_pw, n_bldg_subset, global_asset_indices):
    if len(weights)==0 or sum(weights)==0:
        for count, index in enumerate(global_asset_indices):     
            #new_prop[index] = {strtype_key: "NOT IN HAZUS" } 
            new_prop[index] = {strtype_key: np.nan} 
        return new_prop
    
    if n_pw==0:
        # most likely struct
        struct_pick = [structure_types[np.argmax(weights)]]*n_bldg_subset            
    else:                
        # sample nbldg x n_pw  
        struct_pick = np.random.choice(structure_types, size=[n_bldg_subset, n_pw], replace=True, p=weights ).tolist()

    for count, index in enumerate(global_asset_indices):   

        # shrinks to a scalar value if same.
        val_vec = struct_pick[count]
        if not isinstance(val_vec, list):
            val = val_vec[0]
        elif len(set(val_vec)) == 1:
            val = val_vec[0]
        else:
            val = val_vec

        new_prop[index] = {strtype_key: val} # if #elem in list is 1, convert it to integer

        #new_prop[index] = {strtype_key: self.flatten_array(struct_pick[count])} # if #elem in list is 1, convert it to integer

    return new_prop
########


###### CREATED NEW VERSION OF BRAILLS++ CODE - MODIFIED INTO FUNCTION, ADDED ABILITY TO USE BUILDING TYPE IF FLAG SET AS TRUE
def infer_structure_type(bldg_properties_df, state, occ_key, nstory_key, year_key, bldgtype_key, strtype_key, n_pw, use_bldg_type, allow_mh_only_for_res2, no_urm, res3ab_to_res1_flag):


    #
    # establish structure types list per building type 
    #
    type_by_bldgtype = {
            'M': ['RM1','RM2','URM'], 
            'W': ['W1','W2'],
            'S': ['S1','S2','S3','S4','S5'], 
            'C': ['C1','C2','C3','PC1','PC2'],
            'H': ['MH']}


    #
    # get hazus rulesets 
    #
    states_to_region = hazrules.get_hazus_state_region_mapping()
    height_classes = hazrules.get_hazus_height_classes()
    year_classes = hazrules.get_hazus_year_classes()
    type_lists, type_weights = hazrules.get_hazus_occ_type_mapping()
    bldg_types = ['W','H','M','C','S', None]

    #
    # Add "state" and "region" columns
    #
    bldg_properties_df["state"]=state
    bldg_properties_df["region"]=states_to_region[state]['RegionGroup']
    print('CHECK: Regions considered in structure type assignment:', bldg_properties_df["region"].unique())

    #
    # Add "height_class" column 
    #

    bldg_properties_df["height_class"] = ""
    for height_class, story_list in height_classes.items():
        in_class_index = bldg_properties_df[nstory_key].isin(story_list)
        if sum(in_class_index)>0:
            bldg_properties_df.loc[in_class_index, 'height_class'] = height_class

    #
    # Add "year_class" column
    #

    bldg_properties_df["year_class"] = ""
    for year_class, year_list in year_classes.items():
        in_class_index = bldg_properties_df[year_key].isin(year_list)
        if sum(in_class_index)>0:
            bldg_properties_df.loc[in_class_index, 'year_class'] = year_class

    #
    # Add "bldg_type" column
    #
    bldg_properties_df['bldg_type'] = bldg_properties_df[bldgtype_key]
    bldg_properties_df['bldg_type'] = bldg_properties_df['bldg_type'].replace('',np.nan)

    #
    # Clean occupancy class and add as a new column
    #
    bldg_properties_df[f'{occ_key}_clean'] = bldg_properties_df[occ_key].apply(modulate_occ, args=(res3ab_to_res1_flag,))
    occ_key = f'{occ_key}_clean'

    #
    # Get all cases of interest
    #

    region_list = list(set(bldg_properties_df['region']))
    occ_list = list(set(bldg_properties_df[occ_key]))
    height_list = height_classes.keys()
    state_list = list(set(bldg_properties_df['state']))
    classes_in_inventory = list(product(region_list,occ_list,height_list))


    #
    # Run inference
    #

    new_prop = {}

    for region, occ, height in classes_in_inventory: # for all regions that appear at least once in inventory

        
        subset_inventory = bldg_properties_df[(bldg_properties_df['region']==region) & (bldg_properties_df[occ_key]==occ) & (bldg_properties_df['height_class']==height)]
        nbldg_subset = len(subset_inventory)


        if nbldg_subset==0:
            # no instance found
            continue
            
        if region=="West Coast":

            # year built is considered only in west coast
            
            for year_class in year_classes:
                
                subset_inventory2 = subset_inventory[(subset_inventory['year_class']==year_class)] # inventory with specific region, occ, height, year
                nbldg_subset2 = len(subset_inventory2)
                
                if nbldg_subset2==0:
                    # no instance found
                    continue

                if occ =="RES1":

                    for state in state_list:
                        subset_inventory3 = subset_inventory2[(subset_inventory2['state']==state)] # inventory with specific region, occ, height, year, state
                        nbldg_subset3 = len(subset_inventory3) 

                        if nbldg_subset3==0:
                            # no instance found
                            continue

                        # If flagged to use building type, create additional layer
                        if use_bldg_type: 
                            
                            for bldg_type in bldg_types:

                                # Get building weights 
                                weights = np.array(type_weights[region][occ][year_class][state])/100.
                                structure_types = np.array(type_lists[region]["RES1"])

                                # Modify weights if use_bldg_type flag and data available 
                                if bldg_type: # Filters out None case
                                        
                                    subset_inventory4 = subset_inventory3[(subset_inventory3['bldg_type']==bldg_type)] # inventory with specific region, occ, height, year, state, building type
                                    nbldg_subset4 = len(subset_inventory4)

                                    if nbldg_subset4==0: 
                                        continue
                        
                                    if len(weights): # If weights available for specified region, occ, height, year, state, building type

                                        indices = np.where(np.isin(structure_types, type_by_bldgtype[bldg_type]))[0]
                                        structure_types = structure_types[indices]
                                        weights = weights[indices]

                                        # Normalize if sum greater than 0 
                                        if sum(weights) > 0:
                                            weights = weights / sum(weights)

                                    ## ASSIGN INFORMATION 
                                    weights, structure_types = modulate_weights(weights, structure_types, region, occ, year_class, height, allow_mh_only_for_res2, no_urm)
                                    
                                    if len(weights)==0 or sum(weights) == 0:
                                        print(f"1 HAZUS does not provide structural type information for {region}-{occ}-{height}-{year_class}-{state}-{bldg_type}")
                                        print('Num Buildings in that Class:', nbldg_subset4)

                                    new_prop = add_features_to_asset(new_prop, strtype_key, structure_types, weights, n_pw, nbldg_subset4, global_asset_indices=subset_inventory4.index)
                                    
                                # Case with bldg_type == None
                                else: 

                                    subset_inventory4 = subset_inventory3[(subset_inventory3['bldg_type'].isna())] # inventory with specific region, occ, height, year, state, building type
                                    nbldg_subset4 = len(subset_inventory4)

                                    if nbldg_subset4==0: 
                                        continue
                                    
                                    # Modulate weights
                                    weights, structure_types = modulate_weights(weights, structure_types, region, occ, year_class, height, allow_mh_only_for_res2, no_urm)
                                    
                                    if len(weights)==0 or sum(weights) == 0:
                                        print(f"2 HAZUS does not provide structural type information for {region}-{occ}-{height}-{year_class}-{state}")
                                        print('Num Buildings in that Class:', nbldg_subset4)

                                    new_prop = add_features_to_asset(new_prop, strtype_key, structure_types, weights, n_pw, nbldg_subset4, global_asset_indices=subset_inventory4.index)
                            
                        else: # Don't use building types in data
                            if nbldg_subset3==0: 
                                continue
        
                            weights = np.array(type_weights[region][occ][year_class][state])/100.
                            structure_types = np.array(type_lists[region]["RES1"])

                            # Modulate weights
                            weights, structure_types = modulate_weights(weights, structure_types, region, occ, year_class, height, allow_mh_only_for_res2, no_urm)

                            if len(weights)==0 or sum(weights) == 0:
                                print(f"3 HAZUS does not provide structural type information for {region}-{occ}-{height}-{year_class}-{state}")
                                print('Num Buildings in that Class:', nbldg_subset4)

                            new_prop = add_features_to_asset(new_prop, strtype_key, structure_types, weights, n_pw, nbldg_subset3, global_asset_indices=subset_inventory3.index)
                        
                else: # Not RES1

                    # If flagged to use building type, create additional layer
                    if use_bldg_type: 
                        for bldg_type in bldg_types:

                            # Get building weights 
                            weights = np.array(type_weights[region][occ][height][year_class])/100.
                            structure_types = np.array(type_lists[region][height])

                            # Modify weights if use_bldg_type flag and data available 
                            if bldg_type: # Filters out None case

                                subset_inventory4 = subset_inventory2[(subset_inventory2['bldg_type']==bldg_type)] # inventory with specific region, occ, height, year, building type
                                nbldg_subset4 = len(subset_inventory4)

                                if nbldg_subset4==0:
                                    continue
                                    
                                if len(weights): # If weights available for specified region, occ, height, year, building type
                                    indices = np.where(np.isin(structure_types, type_by_bldgtype[bldg_type]))[0]
                                    structure_types = structure_types[indices]
                                    weights = weights[indices]
                            
                                    # Normalize if sum greater than 0 
                                    if sum(weights) > 0:
                                        weights = weights / sum(weights)

                                # ASSIGN INFORMATION 
                                weights, structure_types = modulate_weights(weights, structure_types, region, occ, year_class, height, allow_mh_only_for_res2, no_urm)
                                        
                                if len(weights)==0 or sum(weights) == 0:
                                    print(f"HAZUS does not provide structural type information for {region}-{occ}-{height}-{year_class}-{bldg_type}")
                                    print('Num Buildings in that Class:', nbldg_subset4)
                        
                                new_prop = add_features_to_asset(new_prop, strtype_key, structure_types, weights, n_pw, nbldg_subset4, global_asset_indices=subset_inventory4.index)
                            
                            else: # Case with bldg_type == None

                                subset_inventory4 = subset_inventory2[(subset_inventory2['bldg_type'].isna())] # inventory with specific region, occ, height, year, building type
                                nbldg_subset4 = len(subset_inventory4)

                                if nbldg_subset4==0:
                                    continue
                                    
                                # ASSIGN INFORMATION 
                                weights, structure_types = modulate_weights(weights, structure_types, region, occ, year_class, height, allow_mh_only_for_res2, no_urm)
                                    
                                if len(weights)==0 or sum(weights) == 0:
                                    print(f"HAZUS does not provide structural type information for {region}-{occ}-{height}-{year_class}-{bldg_type}")
                                    print('Num Buildings in that Class:', nbldg_subset4)

                                new_prop = add_features_to_asset(new_prop, strtype_key, structure_types, weights, n_pw, nbldg_subset4, global_asset_indices=subset_inventory4.index)


                    else: # Don't use building types in data
                        if nbldg_subset2==0:
                            continue

                        weights = np.array(type_weights[region][occ][height][year_class])/100.
                        structure_types = np.array(type_lists[region][height])

                        # Modulate weights
                        weights, structure_types = modulate_weights(weights, structure_types, region, occ, year_class, height, allow_mh_only_for_res2, no_urm)
                        
                        if len(weights)==0 or sum(weights) == 0:
                            print(f"HAZUS does not provide structural type information for {region}-{occ}-{height}-{year_class}")
                            print('Num Buildings in that Class:', nbldg_subset2)

                        new_prop = add_features_to_asset(new_prop, strtype_key, structure_types, weights, n_pw, nbldg_subset2, global_asset_indices=subset_inventory2.index)


        # NOT WEST COAST
        else:
            structureType_key = 'StructureType'

            if occ == "RES1":

                for state in state_list:
                    subset_inventory3 = subset_inventory[(subset_inventory["state"] == state)] 
                    nbldg_subset3 = len(subset_inventory)

                    weights = np.array(type_weights[region][occ][state]) / 100.0
                    structure_types = np.array(type_lists[region]["RES1"])

                    print(structure_types)
                    print(weights)

                    weights, structure_types = modulate_weights(weights, structure_types, region, occ, year_class, height, allow_mh_only_for_res2, no_urm)

                    print(weights)
                    if len(weights) == 0:
                        print(
                            f"HAZUS does not provide structural type information for {region}-{occ}-{height}-{year_class}-{state}. {structureType_key} will be missing in id={subset_inventory2.index.tolist()}"
                        )

                    new_prop = add_features_to_asset(
                        new_prop,
                        structureType_key,
                        structure_types,
                        weights,
                        n_pw,
                        nbldg_subset3,
                        global_asset_indices=subset_inventory3.index,
                    )
            else:

                weights = np.array(type_weights[region][occ][height]) / 100.0
                structure_types = np.array(type_lists[region][height])
                weights, structure_types = modulate_weights(weights, structure_types, region, occ, year_class, height, allow_mh_only_for_res2, no_urm)

                if len(weights) == 0:
                    print(
                        f"HAZUS does not provide structural type information for {region}-{occ}-{height}. {structureType_key} will be missing in id={subset_inventory.index.tolist()}"
                    )

                # add assets
                new_prop = add_features_to_asset(
                    new_prop,
                    structureType_key,
                    structure_types,
                    weights,
                    n_pw,
                    nbldg_subset,
                    global_asset_indices=subset_inventory.index,
                )


    # Append structure type to original building dataframe 
    for index, feature in new_prop.items():
        bldg_properties_df.loc[index, 'StructureType'] = feature['StructureType']

    # RETURN
    return bldg_properties_df
#############



#############
def find_height_class(df, structure_col, numberofstories_col, heightclass_col):
    """
    This function determines the height class of a given structure based on the structure type and number of stories
    heightclass_col (str) is the name of the new column created in the df
    """
    results = []

    # grab indices of dataframe
    structure_idx = df.columns.get_loc(structure_col)
    stories_idx = df.columns.get_loc(numberofstories_col)

    # Loop through each row by index
    for i in range(len(df)):
        structure = df.iloc[i, structure_idx]
        stories = df.iloc[i, stories_idx]

        # determine value for height class column based on values in StructureType and NumberOfStories
        if any(x in structure for x in ['W1', 'W2', 'S3', 'PC1', 'MH']):
            results.append('')
        elif 'RM1' in structure:
            results.append('Low-Rise' if stories <= 3 else 'Mid-Rise')
        elif 'URM' in structure:
            results.append('Low-Rise' if stories <= 2 else 'Mid-Rise')
        else:
            if stories <= 3:
                results.append('Low-Rise')
            elif stories <= 7:
                results.append('Mid-Rise')
            else:
                results.append('High-Rise')

    # result series becomes the height class column
    df[heightclass_col] = results
    return df
#############


#############
def find_design_level(df, structure_col, year_col, designlevel_col):
    """
    This function determines the design level of a given structure based on the structure type and year built.
    designlevel_col (str) is the name of the new column created in the df
    """
    results = []

    # grab indices of dataframe
    structure_idx = df.columns.get_loc(structure_col)
    year_idx = df.columns.get_loc(year_col)

    # Loop through each row by index
    for i in range(len(df)):
        structure = df.iloc[i, structure_idx]
        year = df.iloc[i, year_idx]

        # determine value for design level column based on values in StructureType and YearBuilt
        if 'W1' in structure:
            if year < 1975:
                results.append('Moderate-Code')
            else:
                results.append('High-Code')
        elif any(x in structure for x in ['S5', 'C3', 'URM']):
            if year < 1940:
                results.append('Pre-Code')
            else:
                results.append('Low-Code')
        else:
            if year < 1940:
                results.append('Pre-Code')
            elif year < 1975:
                results.append('Moderate-Code')
            else:
                results.append('High-Code')

    # result series becomes the design level column
    df[designlevel_col] = results
    return df
#############