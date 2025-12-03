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
import random


##########################
def json_to_gdf(file_path, crs_main):
    """
    This function reads a GeoJSON file, extracts the features, and converts them into a GeoDataFrame.
    Input: JSON file path to open
    Output: A GeoDataFrame object with the loaded geometry and attributes.
    """
    with open(file_path, 'r') as f:
        data = json.load(f)
        gdf_loaded = gpd.GeoDataFrame.from_features(data['features'])
        gdf_loaded.set_geometry(gdf_loaded['geometry'], inplace=True)
        gdf_loaded.set_crs(crs_main, inplace=True)
        gdf_loaded = gdf_loaded.map(lambda x: np.nan if x == 'None' else x)
    return gdf_loaded
##########################



##########################
def gdf_to_json(gdf, filepath):
    """
    This function takes a gdf file, puts it in a format that can be stored using a json, then writes a file
    Input: GeoDataFrame and specified file path to save. 
    Output: No output, just saves the file at the specified path. 
    """
    # Apply the function to all columns
    columns_to_edit = gdf.columns
    columns_to_edit = columns_to_edit.drop('geometry')
    for column in columns_to_edit:
        gdf[column] = gdf[column].apply(make_serializable)

    # json_data = gdf.to_json()
    # with open(filepath, 'w') as json_file:
    #     json_file.write(json_data)
    
    with open(filepath, 'w') as f:
        json.dump(json.loads(gdf.to_json()), f, indent=2)
    
    print('JSON File Saved')

# Helper function to make columns of gdf serializable for gdf_to_json
def make_serializable(value):
    if isinstance(value, (int, float, str)):
        return value
    elif isinstance(value, list):
        return [make_serializable(item) for item in value]
    else:
        return str(value)
##########################



##########################
def associate_scraped_data(address_merge, gdf_index, row, columns):
    """
    This function is to associate scraped tax data with an associated set of address points 
    address_merge: overall address dataframe with which tax data should be associated 
    gdf_index: index values which should be updated using scraped data
    row: row of scraped tax data to be paired with the address data
    columns: column headers to be transfered from scraped tax data to address data 
    """
    # Associate scraped data
    address_merge.loc[gdf_index, 'ScrapeFlag'] = 1
    for col in columns:
        if col != 'APN_PQ':
            colname = 'Scrape_'+ col
            address_merge.loc[gdf_index, colname] = row[col]
    # Return
    return address_merge
##########################



##########################
def associate_scraped_split(address_merge, gdf_index, row, columns, address_with_ftpt, use_height):
    """
    This function is to associate scraped tax data by splitting it between several different building footprints based on square footage
    This is used in cases where there are multiple address/footprint pairs within a single parcel
    address_merge: overall address dataframe with which tax data should be associated 
    gdf_index: index values which should be updated using scraped data
    row: row of scraped tax data to be paired with the address data
    columns: column headers to be transfered from scraped tax data to address data 
    address_with_ftpt: gdf containing information on which address point belongs within each footprint. Used to determine scaling factor
    """
    # Associate data with each footprint 
    address_merge = associate_scraped_data(address_merge, gdf_index, row, columns)

    # Adjust scaling on various rows when splitting data across footprints
    if use_height: 
        num = address_with_ftpt.loc[gdf_index,'FootprintArea'] * address_with_ftpt.loc[gdf_index,'FootprintHeight'] # Volumne of each buildint
        denom = sum(address_with_ftpt.loc[gdf_index,'FootprintArea'] * address_with_ftpt.loc[gdf_index,'FootprintHeight']) # Summed volumne of everything in parcel
        factor = num / denom 
    else: 
        factor = address_with_ftpt.loc[gdf_index,'FootprintArea'] / sum(address_with_ftpt.loc[gdf_index,'FootprintArea'])
    address_merge.loc[gdf_index, 'Scrape_Total_Value_Update'] = row['Total_Value'] * factor
    address_merge.loc[gdf_index, 'Scrape_Improvement_Value_Update'] = row['Improvement_Value'] * factor
    address_merge.loc[gdf_index, 'Scrape_Bldg_Area_Update'] = row['Bldg_Area'] * factor

    # Assign number of units - allow for 0 if original scrape has 0, but round to 1 in all other 0 cases 
    if row['Num_Units']== 0: 
        address_merge.loc[gdf_index, 'Scrape_Num_Units_Update'] = 0
    else: 
        address_merge.loc[gdf_index, 'Scrape_Num_Units_Update'] = (np.round(row['Num_Units'] * factor)).apply(lambda val: max(val, 1))

    # Assign number of buildings - allow for 0 if original scrape has 0, but round to 1 in all other 0 cases 
    if row['Num_Bldg']== 0: 
        address_merge.loc[gdf_index, 'Scrape_Num_Bldg_Update'] = 0
    else: 
        address_merge.loc[gdf_index, 'Scrape_Num_Bldg_Update'] = (np.round(row['Num_Bldg'] * factor)).apply(lambda val: max(val, 1))

    # Record infomration for tracking 
    address_merge.loc[gdf_index, 'DataUpdate'] = 'Split Parcel Data Between Footprints'
    address_merge.loc[gdf_index, 'Num_Addresses_Split'] = len(gdf_index)

    # Return  
    return address_merge
##########################



##########################
def associate_scraped_split_multiple_addresses(address_merge, gdf_index, row, columns, address_with_ftpt, ftpt_name, use_height):
    """
    This function is to associate scraped tax data by splitting it between several different building footprints based on square footage
    If differs from the associate_scraped_split function because it handles cases where there are several address points within a single footprint
    This is used in cases where there are multiple address/footprint pairs within a single parcel
    address_merge: overall address dataframe with which tax data should be associated 
    gdf_index: index values which should be updated using scraped data
    row: row of scraped tax data to be paired with the address data
    columns: column headers to be transfered from scraped tax data to address data 
    address_with_ftpt: gdf containing information on which address point belongs within each footprint. Used to determine scaling factor
    """

    # Find rows that have unique footprint and FeatureCode 
    retained = address_with_ftpt.loc[gdf_index].drop_duplicates(subset=[ftpt_name,'FeatureCode'], keep='first')

    # Make notes on number of points 
    address_with_ftpt['Num_Points'] = address_with_ftpt.groupby(ftpt_name)[ftpt_name].transform('count')
    address_merge.loc[address_with_ftpt.index, 'Num_Points'] = address_with_ftpt['Num_Points'].values

    # If some/all footprints have multiple feature codes: 
    if len(retained[retained[ftpt_name].duplicated()]) > 0: 

        # Update Feature Code and Address ID for footprints with more than one feature code 
        duplicated_footprints = retained[retained.duplicated(subset=ftpt_name, keep=False)]
        ftpt_groups = duplicated_footprints.groupby(ftpt_name)
        for ftpt_id, group in ftpt_groups: 
            for i in range(len(group)): 
                address_merge.at[group.index[i], 'FC_Updated'] = list(group['FeatureCode'])
                address_merge.at[group.index[i], 'AddressID_Updated'] = list(group['Address_ID'])

        # Keep closest points for cases with points outside of footprints
        if ftpt_name == 'ClosestFtpt_ID':
            retained = retained.loc[retained.groupby(ftpt_name)['DistanceToFtpt'].idxmin()]
        else: 
            retained = retained.drop_duplicates(subset=[ftpt_name], keep='first')


    # If only one footprint/point remaining 
    if len(retained) == 1: 
        address_merge = associate_scraped_data(address_merge, retained.index, row, columns)

    # If multiple remaining, associate data with one address within each footprint 
    else:
        address_merge = associate_scraped_split(address_merge, retained.index, row, columns, address_with_ftpt, use_height)

    # Make notes for dropped data 
    drop_idx = gdf_index.difference(retained.index)
    address_merge.loc[drop_idx, 'NoScrapedData'] = 'Duplicate Address within Parcel / Footprint'
    
    # Return  
    return address_merge, drop_idx
##########################



##########################
def map_closest_footprints(address_with_ftpt, footprints_orig, use_height):
    """
    This function is to map building height and building area information into the address_with_ftpt gdf to be used for tax data scaling later on. 
    address_with_ftpt: gdf containing information on which address point belongs within each footprint. Used to determine scaling factor
    footprints_orig: gdf used to pull reference information for each FootprintID
    use_height: Flag indicating if scaling should be done just by area of footrpint (use_height = False) vs by volume of building (use_height = True)
    """
    if use_height: 
        area_map = address_with_ftpt[['ClosestFtpt_ID']].merge(footprints_orig[['FootprintID', 'FootprintHeight', 'FootprintArea']], left_on='ClosestFtpt_ID', right_on='FootprintID', how='left')
        address_with_ftpt['FootprintArea'] = area_map['FootprintArea'].values
        address_with_ftpt['FootprintHeight'] = area_map['FootprintHeight'].values
    else: 
        area_map = address_with_ftpt[['ClosestFtpt_ID']].merge(footprints_orig[['FootprintID', 'FootprintArea']], left_on='ClosestFtpt_ID', right_on='FootprintID', how='left')
        address_with_ftpt['FootprintArea'] = area_map['FootprintArea'].values

    # Return
    return address_with_ftpt
##########################



##########################
def find_nearest(points_CB, polygon_adjacent):
    """
    This function finds the nearest Footprint ID (measured to the edge of the footprint) and the distance to that nearest footprint. 
    points_CB is updated and returned with data in the two relevant columns
    """
    # Initialize list to store nearest building info
    nearest_building_indices = []
    nearest_building_distance = []

    # Find the nearest building footprint for each filtered address
    for idx, address in points_CB.iterrows():
        min_dist = np.inf
        nearest_building_idx = None

        for building_idx, building in polygon_adjacent.iterrows():
            distance = address.geometry.distance(building.geometry)
            if distance < min_dist:
                min_dist = distance
                nearest_building_idx = building['FootprintID']

        nearest_building_indices.append(nearest_building_idx)
        nearest_building_distance.append(min_dist)

    # Check for length mismatch
    if len(nearest_building_indices) != len(points_CB):
        raise ValueError("Length mismatch between DataFrame and Index Values")

    # Assign the list to the DataFrame column
    points_CB.loc[:, 'ClosestFtpt_ID'] = nearest_building_indices
    points_CB.loc[:, 'DistanceToFtpt'] = nearest_building_distance

    return points_CB
##########################



##########################
def merge_addresses_into_footprint(ftpt_inv, list_columns, sum_columns, ftpt_cols, ftpt_id, group, local_flag, ftpt_id_column):
    """
    This function finds is intended to merge several address points that fall within the same footprint into a single footprint inventory row. 
    ftpt_inv: footprint-baesd inventory that is taken in. In the function, the original row with the current footprint is dropped, 
    and a new row is added in containing updated information. 
    list_columns: Columns that should not be summed while merging 
    sum_columns: Columns that are summed while merging 
    ftpt_cols: Columns in the original ftpt_inv that should be retained in the new row o fdata
    ftpt_id: Current footprint id being addressesd
    group: gdf of address points located within the same footprint
    local_flag: Flag that should be associated with the new rows being created 

    The function returns a new gdf row that should be appended to ftpt_inv, and a id value of a row that should be replaced (dropped) 
    from the original ftpt_inv
    """

    # Obtian first row to use as baseline  
    data = group.iloc[0].copy()

    # Find appropriate inventory index 
    idx = ftpt_inv[ftpt_inv[ftpt_id_column]==ftpt_id].index
    if len(idx) != 1:
        raise ValueError('Single Footprint Condition Broken')
    
    # Assign single value or list to new row based on variability within group for columns that are potentially uncertain/probailistic
    for col in list_columns:
        col_list = [item for sublist in group[col] for item in (sublist if isinstance(sublist, list) else [sublist])]
        if len(set(col_list)) == 1:
            data[col] = col_list[0]
        else: 
            data[col] = col_list

    # Assign summed values for columns where all data should be maintained for a single structure 
    for col in sum_columns:
        data[col] = sum(group[col])

    # Address number of units 
    # NOTE: This is somewhat specific to the way Num_Units is reported in Hayward. Modifications may be needed if this is being used in a another context 
    group_units = group['Scrape_Num_Units'].dropna().tolist()
    if len(group_units) > 0: 
        most_common_unit = max(set(group_units), key=group_units.count)
        if most_common_unit == 1: # Some cases have all units listed as 1, with each unit representing one apartment within a larger building
            curr_units = len(group) 
        else: # Sometimes, this data reports for all units within a building, which should not be double counted 
            curr_units = most_common_unit
        data['Scrape_Num_Units'] = curr_units
    else: 
        data['Scrape_Num_Units'] = np.nan

    # Set footprint information
    for col in ftpt_cols:
        data[col] = ftpt_inv.loc[idx,col].values[0]

    # Set Flag
    data['Local_Flag'] = local_flag

    # Set number of points 
    data['Num_Points'] = np.sum(group['Num_Points'])

    # Format data to be merged into footprint inventory 
    data = data.to_frame().T
    data_gdf = gpd.GeoDataFrame(data, geometry = 'geometry')
    data_gdf.crs = group.crs

    # Return row to append and index to drop 
    return data_gdf, idx
##########################


##########################
def merge_addresses_into_missing_footprint(list_columns, sum_columns, group):

    # Obtian first row to use as baseline  
    data = group.iloc[0].copy()

    # Assign single value or list to new row based on variability within group for columns that are potentially uncertain/probailistic
    for col in list_columns:
        col_list = [item for sublist in group[col] for item in (sublist if isinstance(sublist, list) else [sublist])]
        if len(set(col_list)) == 1:
            data[col] = col_list[0]
        else: 
            data[col] = col_list

    # Assign summed values for columns where all data should be maintained for a single structure 
    for col in sum_columns:
        data[col] = sum(group[col])

    # Address number of units 
    # NOTE: This is somewhat specific to the way Num_Units is reported in Hayward. Modifications may be needed if this is being used in a another context 
    group_units = group['Scrape_Num_Units'].dropna().tolist()
    if len(group_units) > 0: 
        most_common_unit = max(set(group_units), key=group_units.count)
        if most_common_unit == 1: # Some cases have all units listed as 1, with each unit representing one apartment within a larger building
            curr_units = len(group) 
        else: # Sometimes, this data reports for all units within a building, which should not be double counted 
            curr_units = most_common_unit
        data['Scrape_Num_Units'] = curr_units
    else: 
        data['Scrape_Num_Units'] = np.nan

    # Set number of points 
    data['Num_Points'] = np.sum(group['Num_Points'])

    # Assign geometry point as centroid of all points in group 
    centroid = group.unary_union.centroid
    data['geometry'] = centroid

    # Format data to be merged into footprint inventory 
    data = data.to_frame().T
    data_gdf = gpd.GeoDataFrame(data, geometry = 'geometry')
    data_gdf.crs = group.crs
    data_gdf = data_gdf.drop(columns=['Within_Limit', 'Nearby_AddressIDs', 'GroupID'])

    # Return row to append and index to drop 
    return data_gdf
# ##########################



# ##########################          
def map_values(val, occ_map):
    """
    This function maps occupancy class values using a map provided in occ_map. 
    In the context of Hayward, it used to map the parcel Scrape_Use_Description and the address FeatureCode to NSI equivalents 
    """
    # If the entry is a list, map each element and return list (if not all the same) or value (if all entires map to same NSI type)
    if isinstance(val, list): 
        nsi_list = [occ_map.get(item, item) for item in val]
        if len(set(nsi_list)) == 1: 
            return nsi_list[0]
        else: 
            return nsi_list
    # If the entry is a single value, map it directly
    else:
        return occ_map.get(val, val)
# ##########################



# ##########################   
def list_nearby_address_ids(gdf, distance=7):
    """
    This function is used for address points that do not have a footprint within their parcel. This function finds all other address points that are within
    a specified distance limit (distance) and then stores the associated Address_ID values 
    """
    # Create a new column to store whether each point is within 1 meter of another point
    gdf['Within_Limit'] = False
    
    # Create a new column to store a list of AddressID values within the given distance
    gdf['Nearby_AddressIDs'] = None
    
    # Iterate through each point and compare with all other points
    for i, point in gdf.iterrows():

        # Calculate the distances and identify other points that are within the given distance (excluding itself)
        distances = gdf.geometry.distance(point.geometry)
        nearby_points = gdf[(distances < distance) & (distances > 0)]
        
        if not nearby_points.empty:
            gdf.at[i, 'Within_Limit'] = True
            gdf.at[i, 'Nearby_AddressIDs'] = nearby_points['POINT_ID'].tolist()
    
    return gdf
##########################



##########################  
def find_groups(gdf):
    """
    This function is used for address points that do not have a footprint within their parcel. This based on the close AddressID list that is created by the list_nearby_address_ids function,
    this function creates GroupIDs that group points together that are likely within the same footprint. It creates a column called GroupID in the gdf 
    """
    # Create a new column to store group IDs (initialize with None)
    gdf['GroupID'] = None
    
    current_group_id = 0
    
    # Function to recursively find all connected rows
    def expand_group(start_row, group_id):
        
        # Initialize the group with the start row's AddressID
        group = set([start_row['POINT_ID']])
        
        # Keep track of rows that have been added to the group
        rows_to_process = [start_row]
        
        # Process rows iteratively to find all connected AddressIDs
        while rows_to_process:
            row = rows_to_process.pop()
            
            # Get the current AddressID and its Nearby_AddressIDs
            current_address_id = row['POINT_ID']
            nearby_ids = row['Nearby_AddressIDs'] if row['Nearby_AddressIDs'] else []
            
            # Loop through nearby AddressIDs and add them to the group if not already included
            for nearby_id in nearby_ids:
                if nearby_id not in group:
                    group.add(nearby_id)
                    
                    # Get the corresponding row for the nearby AddressID and add it to processing
                    nearby_row = gdf[gdf['POINT_ID'] == nearby_id].iloc[0]
                    rows_to_process.append(nearby_row)
        
        # Assign the group ID to all rows in the group
        for address_id in group:
            gdf.loc[gdf['POINT_ID'] == address_id, 'GroupID'] = group_id
    
    # Iterate through each row in the GeoDataFrame
    for idx, row in gdf.iterrows():
        if pd.isna(row['GroupID']):  # If the row has not been assigned a group
            # Expand a new group starting from this row
            expand_group(row, current_group_id)
            current_group_id += 1  # Increment group ID for the next group
    
    return gdf
##########################



##########################
def baseline_logic_occ_asignment(value):
    """
    This function defines rules by which to prioritize and select a single occupancy class category based on a list of possible occupancy classes. 
    """

    # Check for 'RES1' or 'RES3' in the list - if residential value in list, assumed to be mixed use 
    if any(item == 'RES3M' for item in value):
        output = 'RES3M'
    elif any(item == 'RES3' for item in value) and any('COM' in item or 'IND' in item or 'REL' in item or 'GOV' in item for item in value): # Allow for mixed use residential / commercial, industrial, religious, government
        output = 'RES3M'
    elif any(item == 'RES3' for item in value): # All other rows with an instance of RES3 are saved as RES3 
        output = 'RES3'
    # This is now cases with no RES3 or RES3M in list
    elif any(item == 'RES1' for item in value): # For any instance of RES1, designate as RES1 
        output = 'RES1'
    # Now check for vacacncy 
    elif any(('_VAC' in item) for item in value):
        non_vac = [item for item in value if pd.notna(item) and ('_VAC' not in item)]
        if non_vac:
            # Get the most common string of the entires that are not planned/vacant 
            output = pd.Series(non_vac).mode().iloc[0]
        else: 
            # These cases remaining are all vacant RES1 and RES3  lists - setting all entires to RES3_VAC 
            output = 'RES3_VAC'

    else:
        # Filter out instances of NOTBLDG and UNK 
        not_notbldg_unk = [item for item in value if pd.notna(item) and item != 'NOTBLDG' and item != 'UNK']

        # If there are entires other than NOTBLDG and UNK 
        if not_notbldg_unk: 

            # Filter out instances of generic 'COM' or 'IND' in order to select other values if present 
            non_na_specific = [item for item in not_notbldg_unk if pd.notna(item) and item != 'COM' and item != 'IND']
            non_na_general = [item for item in not_notbldg_unk if pd.notna(item) and (item == 'COM' or item == 'IND')]

            if non_na_specific:
                # Get the most common string of specific entires
                output = pd.Series(non_na_specific).mode().iloc[0]
            elif non_na_general:
                # Get the most common string for general entires
                output = pd.Series(non_na_general).mode().iloc[0]
            else:
                # Handle case where all values were NaN
                output = np.nan
            
        else: # If only entires in list are NOTBLDG and UNK, return most common string
            output = pd.Series(value).mode().iloc[0]
        
    return output
##########################



##########################
def modify_to_single_tax_occupancy(row):
    """
    Assign a single representative occupancy class from a list of possible occupancy classes from local data, prioritizing based on specified zone.
    Special logic is applied for low-density residential and industrial zones to reflect local land use rules.
    """


    value = row.iloc[0]  # First column
    zoning = row.iloc[1]  # Second column
    

    if isinstance(value, str):
        # If it's a single string, just return it
        if 'RES1' in value:
            output = 'RES1'
        else:
            output = value


    elif isinstance(value, list):

        # Address special cases - low density residential 
        if zoning == 'Low Density Residential': # If zoned low density residential, prioritize 1-4 unit homes
            """
            In Hayward, in low density residential zones, no multi-unit residential with more than four units is permitted. 
            """
            if any(item in ['RES1','RES3A','RES3B','RES3'] for item in value) : # If there is a RES1, RES3A, RES3B present, prioritize that 
                low_density_res = [item for item in value if pd.notna(item) and item in['RES1','RES3A','RES3B','RES3']]
                output = pd.Series(low_density_res).mode().iloc[0]
            else: 
                output = baseline_logic_occ_asignment(value)
                
        # Address special cases - industrial zoning 
        elif zoning in ['Air Terminal-Industrial Park','General Industrial','Light Industrial','Industrial Park']:
            """
            In Hayward, no residential uses are permitted in industrial zoned areas, except for caretakers quarters. Thus, multifamily housing is not prioritized in this logic. 
            Per zoning, there should not be cultural or religious facilities in these areas. Thus, These are only designated if they are the mode of all values in a list. 
            """
            if any(item in ['RES1','RES3A','RES3B'] for item in value) and any('COM' in item or 'IND' in item or 'GOV' in item for item in value): # If there is a mixed use RES1, RES3A, RES3B present, prioritize that 
                low_density_res = [item for item in value if pd.notna(item) and item in['RES1','RES3A','RES3B']]
                output = 'RES1M'
            elif any(item in ['RES1','RES3A','RES3B'] for item in value) : # If there is a RES1, RES3A, RES3B present, prioritize that 
                low_density_res = [item for item in value if pd.notna(item) and item in['RES1','RES3A','RES3B']]
                output = pd.Series(low_density_res).mode().iloc[0]
            else: 
                # Filter out instances of NOTBLDG, UNK, nad nan
                not_notbldg_unk = [item for item in value if pd.notna(item) and item != 'NOTBLDG' and item != 'UNK']

                # If there are entires other than NOTBLDG and UNK 
                if not_notbldg_unk: 
                    modes = pd.Series(not_notbldg_unk).mode()
                    if len(modes) == 1: # Output mode 
                        output = modes.iloc[0]
                    else: # If multiple modes of list, select mode with IND designation 
                        ind_modes = [item for item in modes if 'IND' in item]
                        if len(ind_modes) == 1: 
                            output = ind_modes[0]
                        elif len(ind_modes) > 1: 
                            output = pd.Series(ind_modes).mode().iloc[0]
                        else: 
                            output = modes.iloc[0]            
                else: # If only entires in list are NOTBLDG, UNK, and nan, return most common string
                    output = pd.Series(value).mode().iloc[0]

        # All other cases where zoning is not low density residential and is not industrial 
        else: 
            output = baseline_logic_occ_asignment(value)      

    else:
        # Handle any unexpected data types as NaN
        output = np.nan                              

    return output
##########################



##########################
def clean_supplemental_occ(inv_mod):
    """
    Clean and remove instances of supplemental occuapncy values that indicate that a building has likely not been built,
    if the data represents infrastructure, but likely not an actual structure, or if the data is not helpful in determining 
    the occupancy class of a building 
    """

    # Remove instances where tax and address data both say NOTBLDG
    inv_mod = inv_mod[~((inv_mod['Tax_UseDescription_Hazus_Single'] == 'NOTBLDG') & (inv_mod['Address_FeatureCode_Hazus_Single'] == 'NOTBLDG'))]

    # Remove cases of buildings that have likely not been constructed (parcel is planned or vacant, no tax year built present in data)
    inv_mod = inv_mod[~((inv_mod['Tax_UseDescription_Hazus_Single'].str.contains('_VAC')) & (inv_mod['Address_FeatureCode_Hazus_Single'].str.contains('_VAC')) & (inv_mod['YearBuilt_Single'].isna()))]

    # Remove cases of buildings that have parcel listed as vacant land and address listed as NOTBLDG or vice versa 
    inv_mod = inv_mod[~((inv_mod['Tax_UseDescription_Hazus_Single'].str.contains('_VAC')) & (inv_mod['Address_FeatureCode_Hazus_Single'] == 'NOTBLDG'))]
    inv_mod = inv_mod[~((inv_mod['Address_FeatureCode_Hazus_Single'].str.contains('_VAC')) & (inv_mod['Tax_UseDescription_Hazus_Single'] == 'NOTBLDG'))]

    # Fill nan with ''
    inv_mod['Tax_UseDescription_Hazus_Single'] = inv_mod['Tax_UseDescription_Hazus_Single'].fillna('')
    inv_mod['Address_FeatureCode_Hazus_Single'] = inv_mod['Address_FeatureCode_Hazus_Single'].fillna('')
    if 'NSI_OccupancyClass_Single' in inv_mod.columns:
        inv_mod['NSI_OccupancyClass_Single'] = inv_mod['NSI_OccupancyClass_Single'].fillna('')

    # Fill cases of UNK with ''
    inv_mod['Tax_UseDescription_Hazus_Single'] = inv_mod['Tax_UseDescription_Hazus_Single'].replace('UNK', '')
    inv_mod['Address_FeatureCode_Hazus_Single'] = inv_mod['Address_FeatureCode_Hazus_Single'].replace('UNK', '')

    # If tax and parcel disagree over if it is NOTBLDG, fill NOTBLDG with ''
    inv_mod['Tax_UseDescription_Hazus_Single'] = inv_mod['Tax_UseDescription_Hazus_Single'].replace('NOTBLDG', '')
    inv_mod['Address_FeatureCode_Hazus_Single'] = inv_mod['Address_FeatureCode_Hazus_Single'].replace('NOTBLDG', '')

    # If tax and parcel disagree over if it is vacant, fill VAC with ''
    inv_mod.loc[inv_mod['Tax_UseDescription_Hazus_Single'].str.contains('_VAC', na=False), 'Tax_UseDescription_Hazus_Single'] = ''
    inv_mod.loc[inv_mod['Address_FeatureCode_Hazus_Single'].str.contains('_VAC', na=False), 'Address_FeatureCode_Hazus_Single'] = ''

    return inv_mod
##########################



##########################
def assign_generic_tax_missing(inv_mod, category, possible_vals):
    """
    For cases with generic (IND, COM, RES) tax occupancy and missing NSI Data, 
    assign randomly from available occupancy classes
    """
    
    # Filter rows based on the given category
    if 'NSI_OccupancyClass_Single' in inv_mod.columns:
        generic_tax_missing_nsi = inv_mod[
            ((inv_mod['Address_FeatureCode_Hazus_Single'].isin(category)) & # Address occupancy is generic
            (inv_mod['Tax_UseDescription_Hazus_Single'].isin(category)))&  # Parcel occupancy is generic
            (inv_mod['NSI_OccupancyClass_Single'] == '')]     
    else: 
        generic_tax_missing_nsi = inv_mod[
        ((inv_mod['Address_FeatureCode_Hazus_Single'].isin(category)) & # Address occupancy is generic
        (inv_mod['Tax_UseDescription_Hazus_Single'].isin(category)))]  # Parcel occupancy is generic
    

    
    # Assign a random number within the specified bounds
    occupancy_generated = [f"{category[0]}{random.choice(possible_vals)}" 
                      for _ in range(len(generic_tax_missing_nsi))]

    # Update the specified columns
    inv_mod.loc[generic_tax_missing_nsi.index, 'OccupancyClass_Best'] = occupancy_generated
    
    inv_mod.loc[generic_tax_missing_nsi.index, 'OccupancyClass_Best_Source'] = 'Assigned_from_General_Tax'
    return inv_mod








##########################
def tag_ftpt_with_possible_apn(footprints,parcels, lower_bound, upper_bound):

        
    # Perform spatial join to add APN_PQ from parcels_clean to footprints
    footprints_with_apn = gpd.sjoin(footprints, parcels[['APN_PQ', 'geometry']], how="left", predicate="intersects")

    # Merge the parcel geometry to the footprints based on the spatial join
    parcels = parcels.rename(columns={'geometry': 'parcel_geometry'})
    footprints_with_apn = footprints_with_apn.merge(parcels[['APN_PQ', 'parcel_geometry']], on='APN_PQ')

    # Calculate the area of each footprint
    footprints_with_apn['footprint_area'] = footprints_with_apn.geometry.area

    # Calculate the intersection area between the footprint and parcel geometries
    footprints_with_apn['intersection'] = footprints_with_apn.geometry.intersection(footprints_with_apn['parcel_geometry'])
    footprints_with_apn['intersection_area'] = footprints_with_apn['intersection'].area

    # Calculate the percentage of the footprint that is within the parcel
    footprints_with_apn['percent_overlap'] = (footprints_with_apn['intersection_area'] / footprints_with_apn['footprint_area']) * 100

    # First, filter out rows where the overlap is less than a given % of the footprint wiht a given parcel 
    footprints_with_apn_filter = footprints_with_apn[footprints_with_apn['percent_overlap'] >= lower_bound]
    footprints_with_apn_filter = footprints_with_apn

    # Second, filter rows that have at least a given % of the footprint within a single parcel (drop other parcels associated with that house)
    ftpt_above = footprints_with_apn_filter[footprints_with_apn_filter['percent_overlap'] >= upper_bound]
    ftpt_ids_above = list(ftpt_above['FootprintID'].unique())
    ftpt_below = footprints_with_apn_filter[footprints_with_apn_filter['percent_overlap'] < upper_bound]
    ftpt_below_reduced = ftpt_below[~ftpt_below['FootprintID'].isin(ftpt_ids_above)]

    # Recombine dataframes 
    footprints_filtered = pd.concat([ftpt_above, ftpt_below_reduced], axis=0, ignore_index=True)

    # Reset geometry and drop columns 
    footprints_filtered = footprints_filtered.drop(columns=['intersection', 'footprint_area', 'intersection_area', 'percent_overlap', 'index_right','parcel_geometry'])
    footprints_filtered.set_geometry('geometry', inplace=True)

    # Remove duplicate rows 
    footprints_filtered = footprints_filtered.drop_duplicates(keep = 'first')

    return footprints_filtered