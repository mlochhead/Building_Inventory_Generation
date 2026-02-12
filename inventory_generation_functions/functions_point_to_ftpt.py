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
from shapely.geometry import Polygon, MultiPolygon
from collections import Counter
import random
import warnings
import requests
from shapely.geometry import shape, Point, box
import os





##########################
def estimate_ftpt_size_for_merge(footprints, estimate_stories):
    """
    This function estimates the number of stories given a specified footprint height. Footprint height should be in feet for the conversion
    work as intended. 

    Here, it is assumed that any structure shorter than 19 feet is one story. 19 feet is much taller than a typicaly story, but was assumed 
    as an upper bound due to the potential presence of cripple wall / crawlspaces under houses, other architectural features, etc. Beyond one
    story, a typicaly floor height of 10 feet was assumed, which is somewhat standard for residential construction. 

    This estimaiton could be improved by incorporating the number of stories directly if it is available. In addition, it could also be 
    improved by taking the occupancy class of the footprint into account. Grocery stores, for example, have much taller floors than a single 
    family home, for example. However, this was not done here because we did not want the point attribution process to depend on the 
    features of the point being considered, and we did not have occupancy or story information explicitly tied to the individual footprint. 

    Thus, likely number of stories and square footage is estiamted just based on footprint area and height. 
    """

    if estimate_stories:
        # Estimate story heights
        story_height1 = 19 # Estimated maximum for first story height
        story_height = 10  # Estimated average story height
        max_height = int(footprints['FootprintHeight'].max())   # Maximum building height in dataset

        # Generate bins 
        bins = [0] + [story_height1 + i * story_height for i in range((max_height) // story_height + 1)] # Divisions for building height 
        labels = list(range(1, len(bins)))  # Associated number of stories

        # Estimate stories
        footprints['EstimatedStories'] = pd.cut(footprints['FootprintHeight'], bins=bins, labels=labels, right=False)

        # Remove negative values of story height 
        footprints.loc[footprints['FootprintHeight'] < 0, 'FootprintHeight'] = np.nan

        # Estimate 1 story for all values missing hegiht (good estimate in Hayward, may require different logic elsewhere)
        footprints['EstimatedStories'] = footprints['EstimatedStories'].fillna(1)

        # Conver to float 
        footprints['EstimatedStories'] = footprints['EstimatedStories'].astype(int)

        # Estimate corresponding total square footage of building
        footprints['Total_SqFt'] = footprints['FootprintArea'] * footprints['EstimatedStories']

    else: # Total area assumed to be same as footprint area if height not available
        # Estimate corresponding total square footage of building
        footprints['Total_SqFt'] = footprints['FootprintArea']

    # Return 
    return footprints
##########################




########################### 
def map_to_units(occupancy_list):
    """
    Function to map residential occupancy classes to a number of units, sum units, then use the number of units to re-assign occupancy class
    This function is limited by using the mean of the range designated in NSI (would be much more beneficial to have the actual number of units)
    """
    
    # Create residential scheme 
    mapping_scheme_avg = {'RES1': 1,
                        'RES1-1SNB': 1,
                        'RES1-2SNB': 1,
                        'RES1-2SWB': 1,
                        'RES1-3SNB': 1,
                        'RES1-1SWB': 1,
                        'RES1-3SWB': 1,
                        'RES3A': 2,
                        'RES3B': np.mean([3,4]),
                        'RES3C': np.mean([5,9]),
                        'RES3D': np.mean([10,19]),
                        'RES3E': np.mean([20,50]),
                        'RES3F': 51}

    mapped_avg = [mapping_scheme_avg.get(value, 0) for value in occupancy_list] # If not a residential point, maps to 0 

    # Assign summary occupancy type for size comparisons 
    total_units = round(np.sum(mapped_avg))
    if (total_units) <= 1: 
        group_occ = occupancy_list[0]
    elif (total_units) == 2: 
        group_occ = "RES3A"
    elif (total_units >= 3) & (total_units <= 4):
        group_occ = "RES3B"
    elif (total_units > 4) & (total_units < 10):
        group_occ = "RES3C"
    elif (total_units >= 10) & (total_units < 20):
        group_occ = "RES3D"
    elif (total_units >= 20) & (total_units <= 50):
        group_occ = "RES3E"
    elif (total_units > 50):
        group_occ = "RES3F"
    
    # Return 
    return group_occ
##########################



########################### UPDATED JUL 1
def update_new_rows(nsi, all_new_rows, expected_length):
    """
    Update nsi data with row that has abosorbed other points and been updated accordingly. Drop previous row, and check that no data has been lost using expected_length
    """
    
    # Update NSI dataframe to reflect merged rows 
    new_rows_df = pd.concat(all_new_rows, ignore_index=True)

    # Remove duplicates based on POINT_ID and keep the one with the maximum NSI_NumPoints
    new_rows_df = new_rows_df.loc[new_rows_df.groupby('POINT_ID')['POINT_NumPoints'].idxmax()].reset_index(drop=True)
    
    # Drop and concat points
    nsi_with_rows_dropped = nsi[~nsi['POINT_ID'].isin(set(new_rows_df['POINT_ID']))]
    dtype_reference = nsi_with_rows_dropped if not nsi_with_rows_dropped.empty else new_rows_df
    new_aligned = new_rows_df.reindex(columns=dtype_reference.columns).astype(dtype_reference.dtypes.to_dict(), errors="ignore")
    nsi = pd.concat([nsi_with_rows_dropped, new_aligned], ignore_index=True)

    # Sanity check for length of dataframe 
    if len(nsi) != expected_length:
        raise ValueError(f'Information dropped')
    
    return nsi
##########################


##########################
def find_remaining(points, footprints, point_ftpt_col, point_merge_col):
    """
    This function finds the unpaired points and footprints in NSI data and foorptinr data, based on POINT_MergeFlag and POINT_FootprintID
    Inputs: points: NSI data
            footprints: footprint data
            point_ftpt_col: name of column in nsi data that paired footprint ID is stored
            point_merge_col: name of column in nsi data that MergeFlag is stored
    """
    remaining_points = points[points[point_merge_col]==0]
    remaining_ftpt = footprints[~footprints['FootprintID'].isin(points[point_ftpt_col])]

    print('Number of Points Remaining:',len(remaining_points))
    print('Number of Footprints Remaining:',len(remaining_ftpt))

    return remaining_points, remaining_ftpt
##########################


##########################
def merge_intersecting(points, footprints, crs_plot, plot):
    """
    FUNCTION WITH MERGEFLAG1: MERGE CASES WITH ONE POINT AND ONE FOOTPRINT

    Finds initial intersecting points where one footprint contains one point. 
    Performs spatial joins to identify points within footprints, filters for unique points within each footprint, 
    and merges relevant information into the points GeoDataFrame.

    Outputs:
    - points: GeoDataFrame with updated NSI data
    - m: Folium map showing points associated with multiple footprints if such conflicts are found, or just a string saying no overlap found 
    """

    # Find point that are intersecting with footprint 
    points_with_footprints = gpd.sjoin(points, footprints, how="inner", predicate='within')
    print('Points within Footprints:',len(points_with_footprints))

    # Filter out points that are unique within each footprint
    unique_points = points_with_footprints.groupby('FootprintID').filter(lambda x: len(x) == 1)
    print('Unique points within Footprints (one point per footprint):',len(unique_points))

    # See if any point was associated with multiple footprints 
    if unique_points.index.is_unique == False:
        print('ERROR: SINGLE POINT ASSOCIATED WITH MULTIPLE FOOTPRINTS - PLOTTING DUPLICATE IDS AND ASSOCIATED FOOTPRINTS')
        if plot:
            duplicate = unique_points[unique_points.duplicated(subset='POINT_ID', keep = False)]
            dup_ftpts = footprints[footprints['FootprintID'].isin(duplicate['FootprintID'])]

            # Create a base map
            dup_ftpts_plot = dup_ftpts.copy().to_crs(crs_plot)
            duplicate_plot = duplicate.copy().to_crs(crs_plot)

            m = folium.Map(location=[dup_ftpts_plot.geometry.iloc[0].centroid.y, dup_ftpts_plot.geometry.iloc[0].centroid.x], zoom_start=16)

            # Add footprints (polygons)
            folium.GeoJson(dup_ftpts_plot).add_to(m)

            for idx, row in duplicate_plot.iterrows():
                folium.CircleMarker(location=[row.geometry.y, row.geometry.x],
                                    radius=1,
                                    color='blue',
                                    fill=True,
                                    fill_color='blue',
                                    popup = row['FootprintID']).add_to(m)
        else:
            m = "Failed Check: Overlapping footprints found"
        return points, m 

    else: 
        m = "Passed Check: No overlapping footprints found"

    # Map FootprintID, Merge, and other columns directly
    points = points.merge(unique_points[['POINT_ID', 'FootprintID']], on='POINT_ID', how='left')
    points['POINT_FootprintID'] = points['FootprintID']
    points['ClosestFtpt_ID'] = points['FootprintID']
    points['POINT_MergeFlag'] = points['POINT_FootprintID'].map(lambda x: 0 if pd.isna(x) else 1)
    points['DistanceToFtpt'] = points['POINT_MergeFlag'].replace({1: 0, 0: None})

    # Drop footprint column 
    points = points.drop(columns=['FootprintID'])

    # Display and return 
    print('Data with Associated Footprints (should match row above):',points['POINT_FootprintID'].notna().sum())
    return points, m 
##########################



##########################
def check_post_merge_duplicates(nsi):
    """
    This function checks if any points are designated as both being merged into the same footprint. This should not occur given the current methodology. 
    """
    # Check for duplicate footprints in points that have already been merged
    nsi_paired = nsi[nsi['POINT_MergeFlag']!=0].copy()
    nsi_paired['POINT_FootprintID'] = nsi_paired['POINT_FootprintID'].astype(int)
    duplicates = nsi_paired[nsi_paired.duplicated(subset='POINT_FootprintID', keep=False)]
    if len(duplicates) == 0: 
        print('Passed Check: No duplicates found')
    else: 
        raise ValueError('Duplicates found: Please check process')
##########################


##########################
def recombine_dropped_data(point0, point1, nsi_length):
    """
    This function recombines updated data with and without point data. 
    """
    dtype_reference = point0 if not point0.empty else point1
    nsi1_aligned = point1.reindex(columns=dtype_reference.columns).astype(dtype_reference.dtypes.to_dict(), errors="ignore")

    nsi = pd.concat([point0, nsi1_aligned], ignore_index=True)
    if len(nsi) != nsi_length:
        raise ValueError('NSI Points Dropped')
    return nsi
##########################


########################### UPDATED JUL 1
def create_size_limit_dict(nsi, footprints): 
    """
    This function uses all points with MergeFlag = 1 to find mean and standard deviation footprint sizes for given occupancy types.
    This is used in the future to check if a footprint should be mared as "not full," meaning that the footprint is larger than expected 
    given the occupancy type 
    
    Outputs: the function retruns a limit value to compare against footprint size 
    """
    # Create dictionary 
    size_limit = {}

    # Find Flag1 Entires
    flag1 = nsi[nsi['POINT_MergeFlag']==1].copy()

    # Use only entires that were NOT updated by absorbing other points with NSI_BID
    flag1['POINT_DataUpdate_STR'] = flag1['POINT_DataUpdate'].apply(lambda x: ', '.join(x) if isinstance(x, list) else x)
    flag1 = flag1[~flag1['POINT_DataUpdate_STR'].str.contains('same NSI_BID')]

    # Create dictionary of size limit
    for occ_type in flag1['NSI_OccupancyClass'].unique():

        occ_assigned = flag1[flag1['NSI_OccupancyClass'] == occ_type]
        occ_footprints = footprints[footprints['FootprintID'].isin(occ_assigned['POINT_FootprintID'])]
        occ_size = occ_footprints['Total_SqFt']

        # Compute mean, std, and associated limits 
        occ_size_mean = np.mean(occ_size)
        occ_size_std = np.std(occ_size)

        # Assign limit as 2 standard deviations above the mean
        size_limit[occ_type] = occ_size_mean + 2 * occ_size_std

    return size_limit
##########################



########################### UPDATED JUL 1
def update_mergeflag99(nsi_fxn, footprints, mergeflag):
    """
    This function checks for larger than expected building footprints. It uses the above function (create_size_limit_dict) to create a dictionary, 
    then compares footprint size against the limit for a given occupancy type. If it is larger than expected, the POINT_MergeFlag is set to 99. 
    If there is a list of occupancy class values (due to merged points), the limit is set to whatever occupancy class limit is largest.

    Outputs: Entire updated nsi GeoDataFrame with rows marked as POINT_MergeFlag - 99
    """

    # AD: adding this to prevent SettingWithCopyWarning
    nsi_fxn = nsi_fxn.copy()

    size_limit_dict = create_size_limit_dict(nsi_fxn, footprints)

    # Convert data to list values for ease of use 
    nsi_fxn['OC_List'] = nsi_fxn['NSI_OC_Update'].apply(convert_to_list)

    # Find size corresponding to each occupancy class in list 
    nsi_fxn['size_vals'] = nsi_fxn['OC_List'].map(lambda lst: [size_limit_dict.get(item, np.nan) for item in lst] if isinstance(lst, list) else np.nan)

    # Find max size, the largest for the list of occupancy classes in list 
    nsi_fxn['max_size'] = nsi_fxn['size_vals'].map(lambda lst: max(lst) if isinstance(lst, list) and len(lst) > 0 else np.nan)

    # AD: Normalize merge key dtypes. POINT_FootprintID is stored as an object,
    # while FootprintID is int64. Pandas won’t merge on different dtypes, so it throws a ValueError.
    nsi_fxn["POINT_FootprintID"] = pd.to_numeric(
        nsi_fxn["POINT_FootprintID"], errors="coerce"
    ).astype("Int64")
    footprints["FootprintID"] = pd.to_numeric(
        footprints["FootprintID"], errors="coerce"
    ).astype("Int64")

    #  Join footprint sizes directly into nsi
    nsi_fxn = nsi_fxn.merge(footprints[['FootprintID', 'Total_SqFt']],how='left',left_on='POINT_FootprintID',right_on='FootprintID')

    # Compare and set flag 
    condition = (nsi_fxn['POINT_MergeFlag'] == mergeflag) & (nsi_fxn['Total_SqFt'] > nsi_fxn['max_size'] )
    nsi_fxn['POINT_MergeFlag'] = nsi_fxn['POINT_MergeFlag'].where(~condition, 99)

    # Drop extra columns 
    nsi_fxn = nsi_fxn.drop(columns = ['Total_SqFt','FootprintID','OC_List','size_vals','max_size'])

    return nsi_fxn
##########################



# ##########################
def check_size_limit_from_dict(footprint_size, data, size_limit_dict):
    """
    Check if the given footprint size exceeds the precomputed size limit.
    """

    # Get list of occupancy classes associated with footprint 
    oc = data['NSI_OC_Update']
    oc_list = oc if isinstance(oc, list) else [oc]

    # Map each occupancy to get corresponding size limit
    size_vals = [size_limit_dict.get(item, np.nan) for item in oc_list]

    # Take max size
    max_size = max(size_vals) if size_vals else np.nan

    # Return flag if footprint is larger than max size 
    return int(footprint_size > max_size)
# ##########################


##########################
def drop_ids(nsi, ids_to_drop, note):
    drop_idx = nsi[nsi['POINT_ID'].isin(ids_to_drop)].index
    nsi.loc[drop_idx, 'POINT_DropFlag'] = 1
    nsi.loc[drop_idx, 'POINT_DropNote'] = note
    return nsi
##########################


##########################
def extract_holes(footprint_gdf):
    """
    This is a supporting function only used by address_overlapping_points. Function creates a gdf that contains polygons for all spaces enclosed by a single footprint
    (i.e. a courtyard space). This gdf is used to find points that are lying within these holes, meaning that they are not within the footpring geometry, but are
    enclosed fully by a single footprint. 
    """
    holes_data = []
    for idx, row in footprint_gdf.iterrows():
        footprint = row['geometry']
        
        # Create a dictionary of all row attributes
        row_attributes = row.drop('geometry').to_dict()
        
        if isinstance(footprint, Polygon):  # If the footprint is a single polygon
            if footprint.interiors:
                for hole in footprint.interiors:
                    hole_data = row_attributes.copy()
                    hole_data['geometry'] = Polygon(hole)
                    holes_data.append(hole_data)
                    
        elif isinstance(footprint, MultiPolygon):  # If the footprint is a multipolygon
            for poly in footprint.geoms:
                if poly.interiors:
                    for hole in poly.interiors:
                        hole_data = row_attributes.copy()
                        hole_data['geometry'] = Polygon(hole)
                        holes_data.append(hole_data)
    
    # Create a GeoDataFrame for the holes
    if len(holes_data) > 0: 
        holes_gdf = gpd.GeoDataFrame(holes_data, crs=footprint_gdf.crs)
        
    else: # No holes, create empty gdf
        holes_gdf = gpd.GeoDataFrame(columns=['geometry'], geometry='geometry', crs=footprint_gdf.crs)
    
    return holes_gdf

##########################




##########################
def merge_occ_type(group, manually_assigned_occupancy, print_odd_occupancy_pairings, crs_plot):

    """
    This funciton determines and merges the occupancy type for a group of points assigned to the same footprint.
    If multiple RES1 or RES3 points are assigned to a footprint, they are combined into a single RES3 point with units
    based summing the mean number of units in each individual occupancy class being combined. Occupancy classes other 
    than RES1 and RES3 are unchanged. 

    If print_odd_occupancy_pairings == True, occupancy class pairings determined to be odd/unlikely are printed out with 
    coordinates to check manually in google maps. This function only prints these out, it does not modify them. If no
    action is taken based on printed information, all occupancy classes are retained, and it is assigned a mixed use occupancy. 

    Function output is group_occ. If there is only one occupancy class in the group (one row or multiple rows with same occupancy),
    group_occ is a single value. If there are multiple occupancy classes in the group, group_occ is a list of all values, including duplicates.
    """

    # Find number of unique occupancy types 
    occ_class_list = [item for sublist in group['NSI_OC_Update'] for item in (sublist if isinstance(sublist, list) else [sublist])]
    occ_class_series = pd.Series(occ_class_list)

    # First, check manually assigned occupancy class
    if int(group.iloc[0]['POINT_FootprintID']) in list(manually_assigned_occupancy['FootprintID']): 
        group_occ = manually_assigned_occupancy[manually_assigned_occupancy['FootprintID'] == group.iloc[0]['POINT_FootprintID']].iloc[0]['POINT_OccupancyClass']
    
    # Not in manually assigned occupancy 
    else:

        #### CODE TO PRINT OUT ODD OCCUPANCY PAIRINGS AND THEIR COORDINATES ####
        
        if print_odd_occupancy_pairings:

            ## EDUCATIONAL AND INDUSTRIAL POINTS 
            if (occ_class_series.str.contains('EDU').any()) and (occ_class_series.str.contains('IND').any()):
                df_to_print = group.copy().to_crs(crs_plot)[['POINT_FootprintID','POINT_ID','NSI_OC_Update','NSI_Population_Day', 'NSI_Population_Night','geometry']]
                df_to_print['coords'] = df_to_print['geometry'].apply(lambda point: (point.y, point.x))
                df_to_print = df_to_print.drop(columns = ['geometry'])
                print('WARNING: UNEXPECTED OCCUPANCY COMBINATION - Check Occupancy Type Manually and Assign or Drop (if no action taken, will be kept as mixed use)')
                # AD: display will not work in non-notebook environment. not a good idea to include display in base functions
                # display(df_to_print)
            
            # RESIDENTIAL AND INDUSTRIAL POINTS (IND6 EXCLUDED -- COMMONLY HAS SAME BID AS  RES1)
            if (occ_class_series.str.contains('RES1|RES2|RES3').any()) and (occ_class_series.str.contains('IND1|IND2|IND3|IND4|IND5').any()):
                df_to_print = group.copy().to_crs(crs_plot)[['POINT_FootprintID','POINT_ID','NSI_OC_Update','NSI_Population_Day', 'NSI_Population_Night','geometry']]
                df_to_print['coords'] = df_to_print['geometry'].apply(lambda point: (point.y, point.x))
                df_to_print = df_to_print.drop(columns = ['geometry'])
                print('WARNING: UNEXPECTED OCCUPANCY COMBINATION - Check Occupancy Type Manually and Assign or Drop (if no action taken, will be kept as mixed use)')
                # AD: display will not work in non-notebook environment. not a good idea to include display in base functions
                # display(df_to_print)

            # RESIDENTIAL AND GOVERNMENT POINTS 
            if (occ_class_series.str.contains('RES1|RES2|RES3').any()) and (occ_class_series.str.contains('GOV1').any()):
                df_to_print = group.copy().to_crs(crs_plot)[['POINT_FootprintID','POINT_ID','NSI_OC_Update','NSI_Population_Day', 'NSI_Population_Night','geometry']]
                df_to_print['coords'] = df_to_print['geometry'].apply(lambda point: (point.y, point.x))
                df_to_print = df_to_print.drop(columns = ['geometry'])
                print('WARNING: UNEXPECTED OCCUPANCY COMBINATION - Check Occupancy Type Manually and Assign or Drop (if no action taken, will be kept as mixed use)')
                # AD: display will not work in non-notebook environment. not a good idea to include display in base functions
                # display(df_to_print)



        #### CODE TO FIND AND ASSIGN APPROPRIATE OCCUPANCY DATA ####

        # Handle all cases with only RES1 (Single Family) or RES3 (Multi-Family) occupnacy designations in group 
        if occ_class_series.str.contains('RES1|RES3').all():
            # Combine residential rows and assign occupancy based on mean units (this will be updated later using census data)
            group_occ = map_to_units(occ_class_list)
        

        # If there is only one occupancy class within the group, do not modify 
        elif len(set(occ_class_list)) == 1: 
            group_occ = group['NSI_OC_Update'].iloc[0]


        # Mised use residential and other buildings
        elif occ_class_series.str.contains('RES1|RES3').any():

            # Find residential and nonresidential occupancies 
            res_occupancies = occ_class_series[occ_class_series.str.contains('RES1|RES3')].to_list()
            nonres_occupancies = occ_class_series[~occ_class_series.str.contains('RES1|RES3')].to_list()
            # Assign res occupancy based on mean units and re-combine mixed occupancy types 
            if len(res_occupancies)>0:
                res_occ = map_to_units(res_occupancies)
                group_occ = [res_occ] + nonres_occupancies
            else:
                group_occ = nonres_occupancies

        # No RES1 or RES3 modification needed 
        else:
            group_occ = occ_class_list


    return group_occ
##########################


##########################
def merge_into_group(footprints_indexed, group, merge_flag, list_columns, sum_columns, assigned_ids, manually_assigned_occupancy, size_limit_dict, use_size_limit, use_nsi_occupancy_merge, print_odd_occupancy_pairings, crs_plot): # MTL Added two flags
    """
    Merges a group of overlapping points into a single record, updating occupancy type, residential units, 
    and other attributes. It also checks for size limits and flags the merged record if applicable.
    This function has the ability to deal with points that have already been grouped (i.e. this function can handle invidual points, existing groups of points, or both).

    Outputs:
    - data_gdf: A GeoDataFrame containing the merged group data with updated attributes and centroid geometry.
    - ids_absorbed: List of POINT_IDs to be removed after merging because points were absorbed into another row 

    Process:
    1. Merges occupancy types for the group based on the footprint and manually assigned occupancy rules.
    2. Checks if the footprint exceeds size limits and assigns the appropriate merge flag.
    3. Assigns values or lists to each column as specified  
    4. Calculates the centroid of the group's geometry and prepares the data for merging back into the NSI.
    """

    # Obtian first row for some simplified calculations 
    data = group.iloc[0].copy()

    # Record IDs of rows that have been absorbed into others 
    additional_rows = group.iloc[1:].copy()
    ids_absorbed = [item for sublist in additional_rows['POINT_ID'] for item in (sublist if isinstance(sublist, list) else [sublist])]

    if use_nsi_occupancy_merge:
        # Merge occupancy type 
        group_occ = merge_occ_type(group, manually_assigned_occupancy, print_odd_occupancy_pairings, crs_plot)

        # Set occupancy information  
        data['NSI_OC_Update'] = group_occ

    size_flag = 0
    if use_size_limit:
        # Check how footprint size compares to size limit for specified occupancy to set MergeFlag = 99 flag 
        footprint_id = int(data['POINT_FootprintID'])
        if footprint_id not in assigned_ids: 
            footprint_size = footprints_indexed.at[footprint_id, 'Total_SqFt']
            size_flag = check_size_limit_from_dict(footprint_size, data, size_limit_dict)


    # Assign single value or list to new row based on variability within group for columns that are potentially uncertain/probailistic
    for col in list_columns:
        all_data_list = [item for sublist in group[col] for item in (sublist if isinstance(sublist, list) else [sublist])]
        if len(set(all_data_list)) == 1:
            data[col] = group.iloc[0][col]
        else: 
            data[col] = all_data_list
        
    # Assign summed values for columns where all data should be maintained for a single structure 
    for col in sum_columns:
        all_data_list = [item for sublist in group[col] for item in (sublist if isinstance(sublist, list) else [sublist])]
        data[col] = float(sum(all_data_list))

    # Collect all unique data updates associated with group, then append the new update tag 
    unique_data_updates = list({item for sublist in group['POINT_DataUpdate'] for item in (sublist if isinstance(sublist, list) else [sublist]) if item != ''})
    unique_data_updates.append('Absorbed data from NSI point(s) within same footprint')
    data['POINT_DataUpdate'] = unique_data_updates

    # Get in format appropriate for merge with NSI 
    data = data.to_frame().T

    # Set Merge Flag 
    if size_flag == 1: 
        data['POINT_MergeFlag'] = 99
    else: 
        data['POINT_MergeFlag'] = merge_flag

    # Convert to GeoDataFrame
    data_gdf = gpd.GeoDataFrame(data, geometry = 'geometry')
    data_gdf.crs = group.crs

    return data_gdf, ids_absorbed
##########################




##########################
def address_overlapping_points(nsi_fxn, footprints, list_columns, sum_columns, manually_assigned_occupancy, use_size_limit, use_nsi_occupancy_merge, print_odd_occupancy_pairings, crs_plot): # MTL use_size_limit
    """
    FUNCTION WITH MERGEFLAG2: MERGE CASES WITH MULTIPLE POINTS AND ONE FOOTPRINT
    The function evaluates and merges similar points, updates the NSI data, and handles occupancy assignments for overlapping points. 
    This function address cases with multiple points within a single footprint geometry. it also address points that are not explicitly within a footprint, 
    but are fully enclosed on all sides by the footprint. 

    Output:
    - nsi: Updated GeoDataFrame with resolved overlapping points and footprint assignments.
    """

    # Find remaining points and extract those that are within footprint geometries (these will be merged with footprint information in this step)
    remaining_points, remaining_ftpt = find_remaining(nsi_fxn, footprints,'POINT_FootprintID','POINT_MergeFlag')
    remaining_points_with_footprints = gpd.sjoin(remaining_points, remaining_ftpt[['geometry','FootprintID']], how="inner", predicate='within')

    # Find holes enclosed by building footprints 
    holes_gdf = extract_holes(remaining_ftpt)

    if len(holes_gdf) > 0: 
        # If holes present, find associated points
        points_in_holes = gpd.sjoin(remaining_points, holes_gdf[['geometry','FootprintID']], how="inner", predicate="within")

        # Combine points within footprints and points in holes enclosed by footprints
        remaining_points_with_footprints = pd.concat([remaining_points_with_footprints, points_in_holes], ignore_index=True)

    # Assign POINT_FootprintID
    remaining_points_with_footprints['POINT_FootprintID'] = remaining_points_with_footprints['FootprintID']
    remaining_points_with_footprints = remaining_points_with_footprints.drop(columns = ['index_right','FootprintID'])

    # Get groups of points assigned to same footprint 
    nonunique_groups = remaining_points_with_footprints.groupby('POINT_FootprintID')
    print('Number of Points within Footprint Polygons:', len(remaining_points_with_footprints))
    print('Number of Footprints with Multiple Points (Looping Through These Now):', len(nonunique_groups))


    ### PREPROCESS DATA TO PREPARE FOR MERGE ###

    # Create manually assiged ID list
    assigned_ids = set(manually_assigned_occupancy['FootprintID'])

    # Compute size limit dictionary
    size_limit_dict = {}
    if use_size_limit: 
        size_limit_dict = create_size_limit_dict(nsi_fxn, footprints)

    # Index footprints 
    footprints_indexed = footprints.set_index('FootprintID')

    # Create checkpoints to track progress
    counter = 0 
    progress_checkpoints = {round(len(nonunique_groups) * i / 10) for i in range(1, 11)} 

    # Create storage 
    all_new_rows = []
    all_ids_absorbed = set()


     ### MERGE INTO FOOTPRINTS ###

    for footprint_id, group in nonunique_groups:

        # Process Group
        new_row, ids_absorbed  = merge_into_group(footprints_indexed, group, 2, list_columns, sum_columns, assigned_ids, manually_assigned_occupancy, size_limit_dict, use_size_limit, use_nsi_occupancy_merge, print_odd_occupancy_pairings, crs_plot)

        # Set notes for updated row 
        new_row['DistanceToFtpt'] = 0 

        # Save information for future manipulation 
        all_new_rows.append(new_row)
        all_ids_absorbed.update(ids_absorbed)

        # Print progress 
        if counter in progress_checkpoints:
            percent = round((counter / len(nonunique_groups)) * 100)
            print(f"{percent}% complete")
        counter += 1
    print("100% complete")


    ### UPDATE NSI TO REFLECT MERGE ###
    expected_length = len(nsi_fxn)

    # Mark data that has been dropped 
    nsi_fxn = drop_ids(nsi_fxn, all_ids_absorbed, 'Data merged with another NSI point within same footprint')

    # Update NSI data to reflect merged rows
    if all_new_rows:
        nsi_fxn = update_new_rows(nsi_fxn, all_new_rows, expected_length)

    # Return 
    return nsi_fxn 
##########################





##########################
def find_nearest(points_CB, polygon_adjacent):
    """
    This function finds the nearest Footprint ID (measured to the edge of the footprint) and the distance to that nearest footprint.
    points_CB is updated and returned with data in the two relevant columns.
    """
    # Use a vectorized approach to find the nearest building footprint
    def find_closest_building(address):

        # Compute distances to all buildings
        distances = polygon_adjacent.geometry.distance(address.geometry)

        if len(polygon_adjacent):
            # Find the minimum distance and corresponding FootprintID
            min_dist_idx = distances.idxmin()
            return polygon_adjacent.loc[min_dist_idx, 'FootprintID'], distances.min()
        
        else: # If there are no footprints available 
            return None, np.inf

    # Apply the function to each row of points_CB
    points_CB[['ClosestFtpt_ID', 'DistanceToFtpt']] = points_CB.apply(lambda row: pd.Series(find_closest_building(row)), axis=1)

    return points_CB
##########################



# ########################## USED
def find_index(gdf, nsi_id_to_match):
    """
    Find index in NSI dataframe with multiple NSI points per row (vectorized version)
    """
    mask = gdf['POINT_ID'].apply(lambda x: isinstance(x, int) and x == nsi_id_to_match)
    matching_indices = gdf[mask].index
    return matching_indices[0] if not matching_indices.empty else None
# ##########################



##########################
def pair_empty_ftpt_distance(nsi_fxn, footprints_indexed, unpaired_remaining_points, unpaired_remaining_ftpt, current_points, current_polygons, distance_limit, assigned_ids, size_limit_dict, use_size_limit, merge_flag):
    """
    Pairs unassigned points with unoccupied building footprints within a specified distance limit. This function iterates through 
    remaining unpaired points, finds the closest footprint within a given distance limit, and updates the NSI dataset 
    with occupancy and footprint information.

    Process:
    1. Finds the nearest footprint for each point.
    2. If the point is within the distance limit, it is paired with the closest footprint and occupancy is updated.
    3. Occupancy types are checked for size constraints, and the merge flag is adjusted accordingly.
    4. Paired points and footprints are removed from the remaining pool.
    5. The loop continues until all points are paired or the closest point is beyond the distance limit.

    Outputs:
    - nsi: Updated GeoDataFrame with footprint assignments and occupancy details.
    - unpaired_remaining_points: Remaining unpaired points after the pairing process.
    - unpaired_remaining_ftpt: Remaining unpaired footprints after the pairing process.
    - current_points: Updated list of points to be processed.
    - current_polygons: Updated list of footprints to be processed.
    """
    # While there is at least one point remaining in the census block and loop has not been manually stopped
    # This while loop pairs points with footprints that have no points yet 
    conitnue_flag = True

    # Ensure all POINT_IDs are integers for lookup process
    nsi_fxn['POINT_ID'] = nsi_fxn['POINT_ID'].astype(int) 

    while (len(current_points) > 0) & (len(current_polygons) > 0) & (conitnue_flag): 

        # Find nearest building footprint ID and associated distance 
        current_points = find_nearest(current_points.copy(), current_polygons)

        # Find point with minimum distance to a building footprint 
        closest_point_idx = current_points['DistanceToFtpt'].idxmin()
        closest_point = current_points.loc[[closest_point_idx]]

        if closest_point['DistanceToFtpt'].iloc[0] < distance_limit: 

            # Check footprint flag for updated occupancy type 
            size_limit_flag = 0
            if use_size_limit: 
                closest_id = closest_point['ClosestFtpt_ID'].iloc[0]
                if closest_id not in assigned_ids: 
                    footprint_size = footprints_indexed.at[closest_id, 'Total_SqFt']
                    size_limit_flag = check_size_limit_from_dict(footprint_size, closest_point.iloc[0], size_limit_dict) 

            # Find corresponding index in df and set footprint and NumPoints values 
            index_of_match = find_index(nsi_fxn, int(closest_point['POINT_ID'].iloc[0]))
            nsi_fxn.loc[index_of_match, ['POINT_FootprintID']] = closest_point['ClosestFtpt_ID'].iloc[0]
            nsi_fxn.loc[index_of_match, ['DistanceToFtpt']] = closest_point['DistanceToFtpt'].iloc[0]

            # Reset POINT_MergeFlag if there is still space in the building footprint 
            if size_limit_flag == 1:
                    nsi_fxn.loc[index_of_match, ['POINT_MergeFlag']] = 99

            # If size flag was not raised (paired and no space in building footprint)
            elif size_limit_flag == 0: 
                nsi_fxn.loc[index_of_match, ['POINT_MergeFlag']] = merge_flag
    

            # Drop closest points from current_points and remaining points 
            current_points = current_points[current_points['POINT_ID'] != closest_point['POINT_ID'].iloc[0]]
            unpaired_remaining_points = unpaired_remaining_points[unpaired_remaining_points['POINT_ID'] != closest_point['POINT_ID'].iloc[0]]

            # Drop associated footprint from current_polygons and remaining polygons
            current_polygons = current_polygons[current_polygons['FootprintID'] != closest_point['ClosestFtpt_ID'].iloc[0]]
            unpaired_remaining_ftpt = unpaired_remaining_ftpt[unpaired_remaining_ftpt['FootprintID'] != closest_point['ClosestFtpt_ID'].iloc[0]]
 
        # If the closest points is more than distance_limit away from the nearest building footprint, end the first merge loop (high confidence)
        else: 
            conitnue_flag = False 
            
    return nsi_fxn, unpaired_remaining_points, unpaired_remaining_ftpt, current_points, current_polygons
##########################



##########################
def pair_partial_ftpt_distance(nsi_fxn, footprints, footprints_indexed, unpaired_remaining_points, current_points, distance_limit, assigned_ids, bounding_id_name, manually_assigned_occupancy, size_limit_dict, use_size_limit, use_nsi_occupancy_merge, print_odd_occupancy_pairings, adjacent_bgs, CB_ID, merge_flag, list_columns, sum_columns, use_surrounding_bgs, crs_plot): # CHANGED GEOID MTL 
    """
    This function works similarly to pair_empty_ftpt_distance. However, in this case, the function is evaluating all footprints that have been designated as "not full," by having 
    their POINT_MergeFlag set as 99.
    """

    # While loop pairing with empty footprints is completed. Now check footprints that still have space remaining 
    # Find polygons that are larger than expected, given occupancy type 
    not_full_nsi = nsi_fxn[nsi_fxn['POINT_MergeFlag']==99]
    not_full_ftpt = footprints[footprints['FootprintID'].isin(not_full_nsi['POINT_FootprintID'])]
    umpaired_not_full_ftpt = not_full_ftpt.copy()

    # Set polygons based on surrounding_blocks
    if use_surrounding_bgs:
        not_full_polygon_adjacent = umpaired_not_full_ftpt[umpaired_not_full_ftpt[bounding_id_name].isin(adjacent_bgs[str(bounding_id_name) + '_left'].unique()) | (umpaired_not_full_ftpt[bounding_id_name] == CB_ID)]
    else:
        not_full_polygon_adjacent = umpaired_not_full_ftpt[umpaired_not_full_ftpt[bounding_id_name] == CB_ID]
    current_polygons_not_full = not_full_polygon_adjacent.copy()      

    # Create storage 
    all_new_rows = []
    all_ids_absorbed = set()

    # Loop continues while there are still unmatched points, polygons with available capacity designated as "not full"
    conitnue_flag = True
    while (len(current_points) > 0) & (len(current_polygons_not_full) > 0) & (conitnue_flag): 

        # Find nearest building footprint ID and associated distance 
        current_points = find_nearest(current_points.copy(), current_polygons_not_full)

        # Find points within distance_limit of a polygon 
        close_points = current_points[current_points['DistanceToFtpt'] < distance_limit]

        if len(close_points) > 0: # One or more remaining points are within distance limit of a footprint that still has space available 

            # Group nearby points by the footprint they are closest to
            nonunique_groups = close_points.groupby('ClosestFtpt_ID')

            # Loop through each group of points associated with the same footprint
            for footprint_id, group in nonunique_groups:

                # Retrieve existing NSI entries already assigned to this footprint
                curr_nsi = nsi_fxn[nsi_fxn['POINT_FootprintID'] == footprint_id].copy()
                curr_nsi['ClosestFtpt_ID'] = curr_nsi['POINT_FootprintID']

                # Label POINT_FootprintID
                group['POINT_FootprintID'] = group['ClosestFtpt_ID'] 

                # Align columns to ensure consistent structure for concatenation, using datatypes from curr_nsi
                dtype_reference = curr_nsi if not curr_nsi.empty else group
                group_aligned = group.reindex(columns=dtype_reference.columns).astype(dtype_reference.dtypes.to_dict(), errors="ignore")

                # Merge points newly attributed to footprint and existing NSI data already attributed to same footprint 
                combined_gdf = pd.concat([curr_nsi, group_aligned], ignore_index=True)

                # Merge to create new row 
                new_row, ids_absorbed = merge_into_group(footprints_indexed, combined_gdf, merge_flag, list_columns, sum_columns, assigned_ids, manually_assigned_occupancy, size_limit_dict, use_size_limit, use_nsi_occupancy_merge, print_odd_occupancy_pairings, crs_plot)

                #  Save information for future manipulation 
                all_new_rows.append(new_row)
                all_ids_absorbed.update(ids_absorbed)

                # Drop points from current_points and unpaired_remaining_points
                group_ids = list(group['POINT_ID'].values)
                current_points = current_points[~current_points['POINT_ID'].isin(group_ids)]
                unpaired_remaining_points = unpaired_remaining_points[~unpaired_remaining_points['POINT_ID'].isin(group_ids)]

                
                # Drop footprint from current 
                current_polygons_not_full = current_polygons_not_full[current_polygons_not_full['FootprintID'] != int(new_row.iloc[0]['POINT_FootprintID'])]
        
        else: # No remaining points are within set distance of a footprint that still has space available 
            conitnue_flag = False
    

    ### UPDATE NSI TO REFLECT MERGE 

    expected_length = len(nsi_fxn)

    # Mark data that has been dropped 
    nsi_fxn = drop_ids(nsi_fxn, all_ids_absorbed, 'Data merged with another NSI point within same footprint')

    # Update NSI data to reflect merged rows
    if all_new_rows:
        nsi_fxn = update_new_rows(nsi_fxn, all_new_rows, expected_length)
    
    # Return information
    return nsi_fxn, unpaired_remaining_points, current_points
##########################


##########################
def pair_any_ftpt_distance(nsi_fxn, footprints, footprints_indexed, unpaired_remaining_points, current_points, distance_limit, assigned_ids, bounding_id_name, manually_assigned_occupancy, size_limit_dict, use_size_limit, use_nsi_occupancy_merge, print_odd_occupancy_pairings, adjacent_bgs, BG_ID, merge_flag, list_columns, sum_columns, use_surrounding_bgs, crs_plot):
    """
    This function works similarly to pair_empty_ftpt_distance. However, in this case, the function is evaluating all footprints, including empty footprints, footprints with POINT_MergeFlag = 99, 
     and footprints that have been paired.
    """
    # Find all footprints within boudning geometry and adjacent 
    # Set polygons based on surrounding_blocks
    if use_surrounding_bgs:
        all_ftpt_cb = footprints[footprints[bounding_id_name].isin(adjacent_bgs[str(bounding_id_name) + '_left'].unique()) | (footprints[bounding_id_name] == BG_ID)]
    else:
        all_ftpt_cb = footprints[footprints[bounding_id_name] == BG_ID]
    
    # Create storage 
    all_new_rows = []
    all_ids_absorbed = set()

    # Find nearest building footprint ID and associated distance 
    current_points = find_nearest(current_points.copy(), all_ftpt_cb)

    # Find points within distance_limit of a polygon 
    close_points = current_points[current_points['DistanceToFtpt'] < distance_limit]

    if len(close_points) > 0: # One or more remaining points are within distance limit of a footprint 

        # Group nearby points by the footprint they are closest to
        nonunique_groups = close_points.groupby('ClosestFtpt_ID')

        # Loop through each group of points associated with the same footprint
        for footprint_id, group in nonunique_groups:

            # Retrieve existing NSI entries already assigned to this footprint
            curr_nsi = nsi_fxn[nsi_fxn['POINT_FootprintID'] == footprint_id].copy()
            curr_nsi['ClosestFtpt_ID'] = curr_nsi['POINT_FootprintID']

           # Label POINT_FootprintID
            group['POINT_FootprintID'] = group['ClosestFtpt_ID'] 

            # Align columns to ensure consistent structure for concatenation, using datatypes from curr_nsi
            dtype_reference = curr_nsi if not curr_nsi.empty else group
            group_aligned = group.reindex(columns=dtype_reference.columns).astype(dtype_reference.dtypes.to_dict(), errors="ignore")

            # Merge points newly attributed to footprint and existing NSI data already attributed to same footprint 
            combined_gdf = pd.concat([curr_nsi, group_aligned], ignore_index=True)

            # Merge to create new row 
            new_row, ids_absorbed = merge_into_group(footprints_indexed, combined_gdf, merge_flag, list_columns, sum_columns, assigned_ids, manually_assigned_occupancy, size_limit_dict, use_size_limit, use_nsi_occupancy_merge, print_odd_occupancy_pairings, crs_plot)

            # Save information for future manipulation 
            all_new_rows.append(new_row)
            all_ids_absorbed.update(ids_absorbed)

    ### UPDATE NSI TO REFLECT MERGE 

    expected_length = len(nsi_fxn)

    # Mark data that has been dropped 
    nsi_fxn = drop_ids(nsi_fxn, all_ids_absorbed, 'Data merged with another point within same footprint')

    # Update NSI data to reflect merged rows
    if all_new_rows:
        nsi_fxn = update_new_rows(nsi_fxn, all_new_rows, expected_length)
    
    # Return information
    return nsi_fxn, unpaired_remaining_points, current_points
##########################


##########################

def distance_limit_merge(bounding_id_list, nsi0, footprints, bounding_id_name, manually_assigned_occupancy, list_columns, sum_columns, bounding_geometry, crs_plot, distance_limit, use_surrounding_bgs, prioritize_empty_footprints, prioritize_partial_footprints, use_full_footprints, merge_flag, use_size_limit, use_nsi_occupancy_merge, print_odd_occupancy_pairings): ## MTL ADDED TWO FLAGS 
    """
    This is an overall driver funtion that handles merging data into footprints given a certain distance limit. 
    Inputs: 
    CB_list: list of census blocks that should be considered 
    nsi0: gdf of nsi data considered for merging 
    footprints: gdf of footprints 
    manually_assigned_occupancy: Dictionary with manually assigned occupancy given footprint ID 
    list_columns: Columns that should be mainted as separate options when points are merged into a single footprint (i.e. foundation type)
    sum_columns: Columns that should be summed when combined into a single footprint (i.e. structure value)
    city_blocks: Relevant Census Block geometries 
    distance_limit: Distance (in meters) that is the maximum points should be moved into footprints 
    use_surrounding_blocks: True/False flag indicating if footprints in adjacent census blocks should be considered in the merge 
    use_partial_footprints: True/False flag indicating if footprints that are partially full (MergeFlag = 99) should be cosnidered in the merge 
    use_full_footprints: True/False flag indicating if footprints that are full should be cosnidered in the merge 
    merge_flag: Flag that should be used when data is merged in this step, usually '3' folloed by distance limit (i.e. 310 for 10m limit)
    """

    # Find remaining points and polygons 
    remaining_points, remaining_ftpt = find_remaining(nsi0, footprints, 'POINT_FootprintID','POINT_MergeFlag')
    unpaired_remaining_points = remaining_points.copy()
    unpaired_remaining_ftpt = remaining_ftpt.copy()



    ### PREPROCESS DATA TO PREPARE FOR MERGE ###

    # Create manually assiged ID list
    assigned_ids = set(manually_assigned_occupancy['FootprintID'])

    # Compute size limit dictionary 
    size_limit_dict = {}
    if use_size_limit:
        size_limit_dict = create_size_limit_dict(nsi0, footprints)

    # Index footprints 
    footprints_indexed = footprints.set_index('FootprintID')


    # Create checkpoints to track progress
    counter = 0 
    progress_checkpoints = {round(len(bounding_id_list) * i / 10) for i in range(1, 11)} 
    print(f'Processing {len(bounding_id_list)} Bounding Geometries')

    # Loop through Census Blocks 
    for i in range(len(bounding_id_list)):
        
        # Set Census Block 
        BG_ID = bounding_id_list[i]

        # Find CBs that are adjacent to current CB 
        current_bg = bounding_geometry[bounding_geometry[bounding_id_name] == BG_ID]
        adjacent_bgs = gpd.sjoin(bounding_geometry, current_bg, predicate='touches')

        # Filter the polygons and points that are in the given CB
        points_bg = unpaired_remaining_points[unpaired_remaining_points[bounding_id_name] == BG_ID]
        if use_surrounding_bgs:
            polygon_adjacent = unpaired_remaining_ftpt[unpaired_remaining_ftpt[bounding_id_name].isin(adjacent_bgs[str(str(bounding_id_name)+'_left')].unique()) | (unpaired_remaining_ftpt[bounding_id_name] == BG_ID)]
        else:
            polygon_adjacent = unpaired_remaining_ftpt[unpaired_remaining_ftpt[bounding_id_name] == BG_ID]

        # Filter for cases where there is at least one point within the CB
        if len(points_bg):

            # Initialize current points and polygons
            current_points = points_bg.copy()
            current_polygons = polygon_adjacent.copy()

            # Merge points with any unpaired footprints within the distance limit  
            if prioritize_empty_footprints:
                nsi0, unpaired_remaining_points, unpaired_remaining_ftpt, current_points, current_polygons = pair_empty_ftpt_distance(nsi0.copy(), 
                                                                                                                                      footprints_indexed, 
                                                                                                                                      unpaired_remaining_points, 
                                                                                                                                      unpaired_remaining_ftpt, 
                                                                                                                                      current_points, 
                                                                                                                                      current_polygons, 
                                                                                                                                      distance_limit, 
                                                                                                                                      assigned_ids, 
                                                                                                                                      size_limit_dict, 
                                                                                                                                      use_size_limit, 
                                                                                                                                      merge_flag)

            # Merge points with footprints that are larger than their associated occupancy class within the distance limit  
            if len(current_points) and prioritize_partial_footprints:
                nsi0, unpaired_remaining_points, current_points = pair_partial_ftpt_distance(nsi0.copy(), 
                                                                                             footprints, 
                                                                                             footprints_indexed, 
                                                                                             unpaired_remaining_points, 
                                                                                             current_points, 
                                                                                             distance_limit, 
                                                                                             assigned_ids, 
                                                                                             bounding_id_name, 
                                                                                             manually_assigned_occupancy, 
                                                                                             size_limit_dict, 
                                                                                             use_size_limit, 
                                                                                             use_nsi_occupancy_merge, 
                                                                                             print_odd_occupancy_pairings,
                                                                                             adjacent_bgs, 
                                                                                             BG_ID, 
                                                                                             merge_flag, 
                                                                                             list_columns, 
                                                                                             sum_columns, 
                                                                                             use_surrounding_bgs, 
                                                                                             crs_plot)

            # Merge points with any footprint within distance limit 
            if len(current_points) and use_full_footprints:
                nsi0, unpaired_remaining_points, current_points = pair_any_ftpt_distance(nsi0.copy(), 
                                                                                         footprints, 
                                                                                         footprints_indexed, 
                                                                                         unpaired_remaining_points, 
                                                                                         current_points, 
                                                                                         distance_limit, 
                                                                                         assigned_ids, 
                                                                                         bounding_id_name, 
                                                                                         manually_assigned_occupancy, 
                                                                                         size_limit_dict, 
                                                                                         use_size_limit, 
                                                                                         use_nsi_occupancy_merge, 
                                                                                         print_odd_occupancy_pairings, 
                                                                                         adjacent_bgs, 
                                                                                         BG_ID,
                                                                                         merge_flag, 
                                                                                         list_columns, 
                                                                                         sum_columns, 
                                                                                         use_surrounding_bgs, 
                                                                                         crs_plot)

        # Print progress checkpoints 
        if counter in progress_checkpoints:
            percent = round((counter / len(bounding_id_list)) * 100)
            print(f"{percent}% complete")
        counter += 1
    print("100% complete")
        
    return nsi0

##########################




#########################
def convert_to_list(value):
    """
    Function is used to covert all entires, regardless of type or length, into list objects
    """
    # Check if the value is a string
    if isinstance(value, str):
        return [value]
    elif isinstance(value, list):
        return value
    else: # Handle cases where the value might be of other types
        return [value]
#########################
