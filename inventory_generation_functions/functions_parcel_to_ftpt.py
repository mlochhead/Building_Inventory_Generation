import geopandas as gpd
import pandas as pd
import numpy as np




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
###########################



###########################
def parcel_to_footprint(parcels, points, footprints, use_height): 

    ### MERGE ADDRESS POINTS WITH SCRAPED DATA ###

    #  Filter for footprints that do/do not contain address points 
    footprints_with_addresses = footprints[footprints['FootprintID'].isin(points['FootprintID'].to_list())]
    footprints_no_addresses = footprints[~footprints['FootprintID'].isin(points['FootprintID'].to_list())]

    # Dictionary
    new_ftpts_with_parcel = {}

    # Create checkpoints to track progress
    counter = 0 
    progress_checkpoints = {round(len(parcels) * i / 10) for i in range(1, 11)} 

    unique_index = 0

    # Record dropped parcels 
    dropped_parcels = []

    # Loop through parcels to populate dataframe 
    print(len(parcels), 'parcels total (looping through these now)')
    for index, row in parcels.iterrows():

        # Get footprints within parcel 
        parcel_ftpt_address = footprints_with_addresses[footprints_with_addresses['APN_PQ']==row['APN_PQ']]
        parcel_ftpt_no_address = footprints_no_addresses[footprints_no_addresses['APN_PQ']==row['APN_PQ']]

        # ONE FOOTPRINT WITH ADDRESS DATA
        if len(parcel_ftpt_address) == 1: 

            # Associate all parcel data with given footprint 
            row['FootprintID'] = int(parcel_ftpt_address['FootprintID'].values[0])
            new_ftpts_with_parcel[unique_index] = row
            unique_index += 1
        



        # MULTIPLE FOOTPRINTS WITH ADDRESS DATA 
        elif len(parcel_ftpt_address) > 1:
                
            # Associate all parcel data with set of footprints 

            # Set all relevant footprint IDs to have parcel data associated with them
            for index, ftpt in parcel_ftpt_address.iterrows():
                cur_row = row.copy()
                cur_row['FootprintID'] = int(ftpt['FootprintID'])
                

                # Adjust scaling on various rows when splitting data across footprints
                if use_height: 
                    num = ftpt['FootprintArea'] * ftpt['FootprintHeight'] # Volumne of each building
                    denom = sum(parcel_ftpt_address['FootprintArea'] * parcel_ftpt_address['FootprintHeight']) # Summed volumne of everything in parcel
                    factor = num / denom 
                else: 
                    factor = ftpt['FootprintArea'] / sum(parcel_ftpt_address['FootprintArea'])

                # Assign scaling for area and value
                cur_row['Total_Value'] = cur_row['Total_Value'] * factor
                cur_row['Improvement_Value'] = cur_row['Improvement_Value'] * factor
                cur_row['Bldg_Area'] = cur_row['Bldg_Area'] * factor

                # Assign number of units - allow for 0 if original scrape has 0, but round to 1 in all other 0 cases 
                if cur_row['Num_Units'] > 0: 
                    cur_row['Num_Units'] = max(np.round(cur_row['Num_Units'] * factor),1)

                # Assign number of buildings - allow for 0 if original scrape has 0, but round to 1 in all other 0 cases 
                if cur_row['Num_Bldg'] > 0: 
                    cur_row['Num_Bldg'] = max(np.round(cur_row['Num_Bldg'] * factor),1)

                # Assign data to footprint 
                new_ftpts_with_parcel[unique_index] = cur_row.copy()
                unique_index += 1




        # ONE FOOTPRINT, BUT NO ADDRESS DATA 
        elif (len(parcel_ftpt_address) == 0) and len(parcel_ftpt_no_address) == 1 :

            # Associate all parcel data with given footprint 
            row['FootprintID'] = int(parcel_ftpt_no_address['FootprintID'].values[0])
            new_ftpts_with_parcel[unique_index] = row
            unique_index += 1





        # MULTIPLE FOOTPRINTS, BUT NO ADDRESS DATA 
        elif (len(parcel_ftpt_address) == 0) and len(parcel_ftpt_no_address) > 1 :

            # Associate all parcel data with set of footprints 

            # Set all relevant footprint IDs to have parcel data associated with them
            for index, ftpt in parcel_ftpt_no_address.iterrows():
                cur_row = row.copy()
                cur_row['FootprintID'] = int(ftpt['FootprintID'])

                # Adjust scaling on various rows when splitting data across footprints
                if use_height: 
                    num = ftpt['FootprintArea'] * ftpt['FootprintHeight'] # Volumne of each building
                    denom = sum(parcel_ftpt_no_address['FootprintArea'] * parcel_ftpt_no_address['FootprintHeight']) # Summed volumne of everything in parcel
                    factor = num / denom 
                else: 
                    factor = ftpt['FootprintArea'] / sum(parcel_ftpt_no_address['FootprintArea'])

                # Assign scaling for area and value
                cur_row['Total_Value'] = cur_row['Total_Value'] * factor
                cur_row['Improvement_Value'] = cur_row['Improvement_Value'] * factor
                cur_row['Bldg_Area'] = cur_row['Bldg_Area'] * factor

                # Assign number of units - allow for 0 if original scrape has 0, but round to 1 in all other 0 cases 
                if cur_row['Num_Units'] > 0: 
                    cur_row['Num_Units'] = max(np.round(cur_row['Num_Units'] * factor),1)

                # Assign number of buildings - allow for 0 if original scrape has 0, but round to 1 in all other 0 cases 
                if cur_row['Num_Bldg'] > 0: 
                    cur_row['Num_Bldg'] = max(np.round(cur_row['Num_Bldg'] * factor),1)

                # Assign data to footprint 
                new_ftpts_with_parcel[unique_index] = cur_row.copy()
                unique_index += 1




        # NO FOOTPRINTS
        elif (len(parcel_ftpt_address) == 0) and (len(parcel_ftpt_no_address) == 0):

            ## Find address points in parcel 
            points_with_apn = points[points['APN_PQ']==row['APN_PQ']]

            ## If there are points in the parcel, keep parcel data 
            if len(points_with_apn) > 0: 

                # Assign data witb missing footpirnt
                row['FootprintID'] = np.nan 
                new_ftpts_with_parcel[unique_index] = row
                unique_index += 1

            ## If there are no points in the parcel, drop parcel 
            else: 
                dropped_parcels.append(row['APN_PQ'])

        
        else:
            raise ValueError('Error: Code should not reach this point')


        # Print progress 
        if counter in progress_checkpoints:
            percent = round((counter / len(parcels)) * 100)
            print(f"{percent}% complete")
        counter += 1


    ## Convert attributed parcel data to a geodataframe
    parcels_attributed = pd.DataFrame.from_dict(new_ftpts_with_parcel, orient="index")
    parcels_attributed = gpd.GeoDataFrame(parcels_attributed, geometry = 'geometry')
    parcels_attributed.crs = parcels.crs

    print('Attribution Complete')

    # Return
    return parcels_attributed
###########################



###########################
def combine_data_in_footprint(group, sum_columns,list_columns): 

    """
    This function is called from the merge_parcels_in_single_footprint function to combine multiple parcels attributed to a single footprint into a single row.
    """

    # Obtian first row for some simplified calculations 
    data = group.iloc[0].copy()

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

    # Convert to GeoDataFrame
    data = data.to_frame().T
    data_gdf = gpd.GeoDataFrame(data, geometry = 'geometry')
    data_gdf.crs = group.crs

    return data_gdf
############################



############################
def merge_parcels_in_single_footprint(parcels_attributed, sum_columns, list_columns):
    """
    This function combines data from multiple parcels assigned to the same footprint (i.e. building with multiple condos).
    """

    # Filter for parcels assigned to footprints 
    parcels_attributed['Num_Parcels'] = 1
    parcels_attributed_ftpt = parcels_attributed[parcels_attributed['FootprintID'].notna()].copy()
    parcels_attributed_no_ftpt = parcels_attributed[parcels_attributed['FootprintID'].isna()].copy()

    # Group by footprint 
    groups = parcels_attributed_ftpt.groupby('FootprintID')

    # Create storage 
    all_new_rows = []

    # Create checkpoints to track progress
    counter = 0 
    progress_checkpoints = {round(len(groups) * i / 10) for i in range(1, 11)} 

    for footprint_id, group in groups:

        # Process Group
        new_row  = combine_data_in_footprint(group, sum_columns,list_columns)

        # Save information
        all_new_rows.append(new_row)


        # Print progress 
        if counter in progress_checkpoints:
            percent = round((counter / len(groups)) * 100)
            print(f"{percent}% complete")
        counter += 1

    # Convert to df 
    footprints_with_parcel_data = pd.concat(all_new_rows, ignore_index=True)


    ## Recbombine parcels with and without footprint data 
    combined_gdf = pd.concat([footprints_with_parcel_data, parcels_attributed_no_ftpt], ignore_index=True)

    ## Return
    return combined_gdf
###########################




###########################
def combine_address_and_parcel(points, parcels, footprints_original):
    """
    This function combines address point and parcel data for cases with and without building footprints. For cases with no footprints, parcel data is distributed among address points within the parcel. 
    """

    ## Sort data based on whether it is attributed to a footprint or not 
    points_with_ftpt = points[points['FootprintID'].notna()]
    parcels_with_ftpt = parcels[parcels['FootprintID'].notna()]
    points_no_ftpt = points[points['FootprintID'].isna()]
    parcels_no_ftpt = parcels[parcels['FootprintID'].isna()]



    ### LINK PARCEL AND ADDRESS DATA FOR CASES WITH ATTRIBUTED FOOTPRINTS 

    points_with_ftpt = points_with_ftpt.drop(columns = ['geometry', 'APN_PQ'])
    parcels_with_ftpt = parcels_with_ftpt.drop(columns = ['geometry'])

    print(len(points_with_ftpt), 'points with footprints')
    print(len(parcels_with_ftpt), 'parcels with footprints')
    ftpt_with_parcels_address = points_with_ftpt.merge(parcels_with_ftpt, on = 'FootprintID', how = 'outer')

    # Merge data with footprint geometry 
    ftpt_with_parcels_address = ftpt_with_parcels_address.merge(footprints_original[['FootprintID','FootprintArea','FootprintHeight','geometry']], on = 'FootprintID', how = 'left')
    ftpt_with_parcels_address = gpd.GeoDataFrame(ftpt_with_parcels_address, geometry = 'geometry')
    ftpt_with_parcels_address.crs = footprints_original.crs

    # Modify footprint inventory geometry to be the centroid of each footprint 
    ftpt_with_parcels_address = ftpt_with_parcels_address.rename(columns={'geometry': 'ftpt_geometry'})
    ftpt_with_parcels_address['geometry'] = ftpt_with_parcels_address['ftpt_geometry'].centroid
    ftpt_with_parcels_address.set_geometry('geometry')
    ftpt_with_parcels_address['Footprint_Flag'] = 1





    ### LINK PARCEL AND ADDRESS DATA FOR CASES WITH NO FOOTPRINTS 

    ## CREATE FOOTPRINTS FOR ADDRESS POINTS 
    new_points_with_parcel = {}
    unique_index = 0

    # Split parcel data across address points 
    # Loop through parcels to populate dataframe 
    parcels_no_ftpt = parcels_no_ftpt.drop(columns=['geometry'])
    print(len(parcels_no_ftpt), 'parcels with no footprint (looping through these now)')
    for index, row in parcels_no_ftpt.iterrows():

        # Get points within parcel 
        points_in_parcel = points_no_ftpt[points_no_ftpt['APN_PQ']==row['APN_PQ']].copy()

        ## ONE ADDRESS POINT IN PARCEL 
        if len(points_in_parcel) == 1: 

            # Attribute parcel informaiton to address point 
            cur_row = points_in_parcel.copy()
            parcel_attrs = row.drop('APN_PQ')
            for col, val in parcel_attrs.items():
                cur_row[col] = val

            # Save information 
            new_points_with_parcel[unique_index] = cur_row.iloc[0]
            unique_index += 1


        ## MULTIPLE ADDRESS POINTS IN PARCEL 
        elif len(points_in_parcel) > 1: 

            # Attribute parcel informaiton to all address points
            cur_rows = points_in_parcel.copy()
            parcel_attrs = row.drop('APN_PQ')
            for col, val in parcel_attrs.items():
                cur_rows[col] = val
            
            # Divide relevant columns between parcels 
            factor = 1 / len(points_in_parcel)
            cur_rows['Total_Value'] = cur_rows['Total_Value'] * factor
            cur_rows['Improvement_Value'] = cur_rows['Improvement_Value'] * factor
            cur_rows['Bldg_Area'] = cur_rows['Bldg_Area'] * factor
            
            # Save information 
            for i in range(len(cur_rows)):
                new_points_with_parcel[unique_index] = cur_rows.iloc[i]
                unique_index += 1


    # Convert to gdf
    points_with_parcel_data = pd.DataFrame.from_dict(new_points_with_parcel, orient="index")
    points_with_parcel_data = gpd.GeoDataFrame(points_with_parcel_data, geometry = 'geometry')
    points_with_parcel_data.crs = parcels.crs


    ## Drop points that have no building footprint and no year built - most of these seem to be planning development parcels based on spot checks in google maps
    points_with_parcel_data = points_with_parcel_data[points_with_parcel_data['Year_Built'].notna()]


    return ftpt_with_parcels_address, points_with_parcel_data
###########################




###########################   
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




###########################
def create_missing_footprint_groups(points_with_parcel_data, sum_columns, list_columns): 
    
    """
    This function groups address points with parcel data but no footprints into likely footprints based on distance between points, and combines data within each group.
    """
    # Group address points based on likely being in same building (based on 7m distance limit, which was selected through Hayward trial and error)
    points_with_parcel_data = list_nearby_address_ids(points_with_parcel_data)
    points_with_parcel_data = find_groups(points_with_parcel_data)

    # Use GroupID (likely footprint) to combine data
    groups = points_with_parcel_data.groupby('GroupID')
    all_new_rows = []
    for group_id, group in groups:
        new_row  = combine_data_in_footprint(group, sum_columns,list_columns)
        all_new_rows.append(new_row)

    # Convert to df 
    missing_footprint_groups = pd.concat(all_new_rows, ignore_index=True)
    missing_footprint_groups = missing_footprint_groups.drop(columns=['Within_Limit','Nearby_AddressIDs','GroupID'])

    ## Create false footprint IDs for cases with missing footprints 
    missing_footprint_groups['FootprintID'] = range(len(missing_footprint_groups))
    missing_footprint_groups['FootprintID'] = missing_footprint_groups['FootprintID'] + 100000
    missing_footprint_groups['Footprint_Flag'] = 0

    return missing_footprint_groups
##########################   
