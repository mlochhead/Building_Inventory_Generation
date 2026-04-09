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
from collections import Counter
import random
import requests

import hazus_rulesets as hazrules






#########################
def modify_to_single_val(value):
    """
    Function to modify list entires created in the tax / nsi inventory development process to be single values. 
    This function explicitly handles cases where information is NOT paired in a way where correlations should be maintained. 
    """

     # If value is not a list, return it as is
    if not isinstance(value, list):
        if pd.notna(value) and value != 'Missing':
            return value
        else: 
            return float('nan')
    
    # Remove NaN values from the list
    clean_list = [v for v in value if pd.notna(v) and v != 'Missing' and v != 0 and v != '0']

    
    # If the list is empty after removing NaNs, return NaN
    if not clean_list:
        return float('nan')
    
    # Count occurrences of each value
    count = Counter(clean_list)
    
    # Find the most common value(s)
    most_common_values = count.most_common()
    max_count = most_common_values[0][1]
    
    # Get all values that have the max count (to handle ties)
    candidates = [val for val, freq in most_common_values if freq == max_count]
    
    # If there's a tie, randomly sample from candidates
    if len(candidates) > 1:
        return random.choice(candidates)
    else:
        return candidates[0]
#########################





#########################
def modify_to_single_val_paired(row):
    """
    Processes two columns in a row to select a value from the first column and the corresponding value from the second column.
    
    Parameters:
    row (pd.Series): A pandas row containing two columns (e.g., col1 and col2).
    
    Returns:
    tuple: Selected value from col1 and corresponding value from col2.
    """
    col1 = row.iloc[0]  # First column
    col2 = row.iloc[1]  # Second column

    # Process col1 to get the most common value
    if not isinstance(col1, list):
        if pd.notna(col1) and col1 != 'Missing':
            selected_col1 = col1
        else:
            return (float('nan'), float('nan'))
    else:
        clean_list1 = [v for v in col1 if pd.notna(v) and v != 'Missing' and v != 'None']
        if not clean_list1:
            return (float('nan'), float('nan'))
        
        count1 = Counter(clean_list1)
        most_common_values1 = count1.most_common()
        max_count1 = most_common_values1[0][1]
        candidates1 = [val for val, freq in most_common_values1 if freq == max_count1]
        
        selected_col1 = random.choice(candidates1) if len(candidates1) > 1 else candidates1[0]

    # Process col2 to get the value corresponding to the selected_col1
    if not isinstance(col2, list):
        return (selected_col1, col2 if pd.notna(col2) and col2 != 'Missing' else float('nan'))
    else:
        if not isinstance(col1, list):  # col1 is not a list, so no corresponding index
            return (selected_col1, float('nan'))
        
        try:
            idx = col1.index(selected_col1)  # Find the index of the selected value in col1
            if idx < len(col2):  # Check if index is within bounds for col2
                selected_col2 = col2[idx]
                return (selected_col1, selected_col2 if pd.notna(selected_col2) and selected_col2 != 'Missing' else float('nan'))
            else:
                return (selected_col1, float('nan'))
        except ValueError:  # If the selected_col1 is not in col1 (edge case)
            return (selected_col1, float('nan'))
#########################





######################### 
def modify_to_single_nsi_occupancy(value): 

    """
    This function takes a list of values of NSI occupancy classes associated with a single building footprint, and returns two values. 

    First, it returns the single value associated with the occupancy in a deterministic version of the inventory (output). This is determined by using the mode
    of the values, and may be designated as mixed use if there is a combination of RES1/RES2/RES3 and non-residential occupancy classes.

    Second, it returns the mixed_output, which is returned as an empty string, unless output is mixed use, in which case it represents the non-residential occupancy class
    casuing the mixed use designation.
    """
    
    # Set random seed
    np.random.seed(1)

    # Preset mixed output as empty 
    mixed_output = ''

    # String
    if isinstance(value, str):
        if 'RES1' in value: # Simplify RES1 occupancies 
            output = 'RES1'
        else:
            output = value

    # Empty list
    elif isinstance(value, list) and len(value) == 0:
        output = np.nan
    
    # List with len 1
    elif isinstance(value, list) and len(value) == 1:
        if 'RES1' in value[0]:
            output = 'RES1'
        else:
            output = value[0]
    
    # If it is a list of potential occupancy classes, use logic to decide 
    elif isinstance(value, list):

        # If occupancy only contains GOV2 and EDU2, it is likely a campus police station - specify GOV2 if it is present 
        if all("EDU2" in entry or "GOV2" in entry for entry in value):
            gov = [item for item in value if 'GOV2' in item]
            edu = [item for item in value if 'EDU2' in item]
            if len(gov):
                output = np.random.choice(pd.Series(gov).mode())
            else: 
                output = np.random.choice(pd.Series(edu).mode())
        
        # If occupancy only contains RES5 and EDU2, it is likely a campus dorm - specify RES5 if it is presernt 
        elif all("EDU2" in entry or "RES5" in entry for entry in value):
            res = [item for item in value if 'RES5' in item]
            edu = [item for item in value if 'EDU2' in item]
            if len(res):
                output = np.random.choice(pd.Series(res).mode())
            else: 
                output = np.random.choice(pd.Series(edu).mode())
        
            
        else: # Most cases with multiple occupancies fall under this cateogry 

            # Separate out appropriate categories to prioritize
            res123ab = [item for item in value if 'RES1' in item or 'RES2' in item or 'RES3A' in item or 'RES3B' in item]
            res3cf = [item for item in value if 'RES3C' in item or 'RES3D' in item or 'RES3E' in item or 'RES3F' in item]
            res456 = [item for item in value if 'RES4' in item or 'RES5' in item or 'RES6' in item]
            edu_gov2 = [item for item in value if 'EDU' in item or 'GOV2' in item]
            other = [item for item in value if item not in res123ab and item not in res3cf and item not in res456 and item not in edu_gov2]


            
            ## PRIORITIZE EDUCATIONAL AND GOV2 FACILITIES BECAUSE THEY ARE FROM HIFLD 
            if len(edu_gov2): 
                output = np.random.choice(pd.Series(edu_gov2).mode())
            
                # If there are mixed occupancies, prioritize other residential occs as mixed use occupancy if they are present 
                if len(res3cf): 
                    mixed_output = np.random.choice(pd.Series(res3cf).mode())
                elif len(res456): 
                    mixed_output = np.random.choice(pd.Series(res456).mode())
                elif len(res123ab):
                    mixed_output = np.random.choice(pd.Series(res123ab).mode())
                elif len(other):
                    mixed_output = np.random.choice(pd.Series(other).mode())
            
            ## PRIORITIZE LARGE RESIDENTIAL OCCUPANCIES 
            # If there is a large RES3 data point, label as mixed use multiunit residential
            elif len(res3cf):
                output = np.random.choice(pd.Series(res3cf).mode())

                # If there are mixed occupancies, other residential occs (RES4, RES5, RES6) as mixed use occupancy if they are present 
                if len(res456): 
                    mixed_output = np.random.choice(pd.Series(res456).mode())
                    output = output + 'M'
                elif len(other):
                    mixed_output = np.random.choice(pd.Series(other).mode())
                    output = output + 'M'

            
            ## PRIORITIZE ADDITIONAL RESIDENTIAL OCCUPANCIES
            # These are prioritized over smaller residential buildings (RES1, RES2, RES3A, RES3B) becuase structural system is likely driven by mixed use residential type 
            # These are also not very common in Hayward, so they do not make a very large impact 
            elif len(res456): 
                output = np.random.choice(pd.Series(res456).mode())
            
                # If there are mixed occupancies, prioritize other residential occs as mixed use occupancy if they are present 
                if len(res123ab): 
                    mixed_output = np.random.choice(pd.Series(res123ab).mode())
                    output = output + 'M'
                elif len(other):
                    mixed_output = np.random.choice(pd.Series(other).mode())
                    output = output + 'M'
            

            ## PRIORITIZE SMALL RESIDENTIAL BUILDINGS 
            # If there are no large RES3 points, no EDU/GOV2 points, and no RES4/5/6 points, prioritize RES1, RES2, RES3A, and RES3B 
            elif len(res123ab): 
                output = np.random.choice(pd.Series(res123ab).mode())

                # If there are mixed occupancies, specify here
                if len(other): 
                    mixed_output = np.random.choice(pd.Series(other).mode())
                    output = output + 'M'
            
            ## OTHER OCCUPANCY CLASSES 
            # Only nonres occupancies. Specify the mode of the list of values from NSI 
            elif len(other):
                output = np.random.choice(pd.Series(other).mode())
                

            # Code should never get here. Put in warning if this occurs
            else: 
                output = np.nan
                print('WARNING: Occupancy Class Specified as NaN')
    
    # List NaN for other data types
    else: 
        output = np.nan

    return (output, mixed_output)
#########################




##########################
def baseline_logic_occ_asignment(value):
    """
    This function defines rules by which to prioritize and select a single occupancy class category based on a list of possible occupancy classes. 
    It is called by modify_to_single_tax_occupancy, which is specific to Hayward data. 
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
    NOTE: THIS FUNCTION IS SPECIFIC TO HAYWARD ZONING DATA
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
    This function is specific to Hayward data. 
    Clean and remove instances of supplemental occuapncy values that indicate that a building has likely not been built,
    if the data represents infrastructure, but likely not an actual structure, or if the data is not helpful in determining 
    the occupancy class of a building 
    """

    # Remove instances where tax and address data both say NOTBLDG
    inv_mod = inv_mod[~((inv_mod['Parcel_UseDescription_Hazus_Single'] == 'NOTBLDG') & (inv_mod['Address_FeatureCode_Hazus_Single'] == 'NOTBLDG'))]

    # Remove cases of buildings that have likely not been constructed (parcel is planned or vacant, no tax year built present in data)
    inv_mod = inv_mod[~((inv_mod['Parcel_UseDescription_Hazus_Single'].str.contains('_VAC')) & (inv_mod['Address_FeatureCode_Hazus_Single'].str.contains('_VAC')) & (inv_mod['Parcel_YearBuilt_Single'].isna()))]

    # Remove cases of buildings that have parcel listed as vacant land and address listed as NOTBLDG or vice versa 
    inv_mod = inv_mod[~((inv_mod['Parcel_UseDescription_Hazus_Single'].str.contains('_VAC')) & (inv_mod['Address_FeatureCode_Hazus_Single'] == 'NOTBLDG'))]
    inv_mod = inv_mod[~((inv_mod['Address_FeatureCode_Hazus_Single'].str.contains('_VAC')) & (inv_mod['Parcel_UseDescription_Hazus_Single'] == 'NOTBLDG'))]

    # Fill nan with ''
    inv_mod['Parcel_UseDescription_Hazus_Single'] = inv_mod['Parcel_UseDescription_Hazus_Single'].fillna('')
    inv_mod['Address_FeatureCode_Hazus_Single'] = inv_mod['Address_FeatureCode_Hazus_Single'].fillna('')
    if 'NSI_OccupancyClass_Single' in inv_mod.columns:
        inv_mod['NSI_OccupancyClass_Single'] = inv_mod['NSI_OccupancyClass_Single'].fillna('')

    # Fill cases of UNK with ''
    inv_mod['Parcel_UseDescription_Hazus_Single'] = inv_mod['Parcel_UseDescription_Hazus_Single'].replace('UNK', '')
    inv_mod['Address_FeatureCode_Hazus_Single'] = inv_mod['Address_FeatureCode_Hazus_Single'].replace('UNK', '')

    # If tax and parcel disagree over if it is NOTBLDG, fill NOTBLDG with ''
    inv_mod['Parcel_UseDescription_Hazus_Single'] = inv_mod['Parcel_UseDescription_Hazus_Single'].replace('NOTBLDG', '')
    inv_mod['Address_FeatureCode_Hazus_Single'] = inv_mod['Address_FeatureCode_Hazus_Single'].replace('NOTBLDG', '')

    # If tax and parcel disagree over if it is vacant, fill VAC with ''
    inv_mod.loc[inv_mod['Parcel_UseDescription_Hazus_Single'].str.contains('_VAC', na=False), 'Parcel_UseDescription_Hazus_Single'] = ''
    inv_mod.loc[inv_mod['Address_FeatureCode_Hazus_Single'].str.contains('_VAC', na=False), 'Address_FeatureCode_Hazus_Single'] = ''

    return inv_mod
##########################






##########################
def assign_generic_tax_missing(inv_mod, category, possible_vals):
    """
    This function is specific to Hayward data. 
    For cases with generic (IND, COM, RES) tax occupancy and missing NSI Data, 
    assign randomly from available occupancy classes
    """
    
    # Filter rows based on the given category
    if 'NSI_OccupancyClass_Single' in inv_mod.columns:
        generic_tax_missing_nsi = inv_mod[
            ((inv_mod['Address_FeatureCode_Hazus_Single'].isin(category)) & # Address occupancy is generic
            (inv_mod['Parcel_UseDescription_Hazus_Single'].isin(category)))&  # Parcel occupancy is generic
            (inv_mod['NSI_OccupancyClass_Single'] == '')]     
    else: 
        generic_tax_missing_nsi = inv_mod[
        ((inv_mod['Address_FeatureCode_Hazus_Single'].isin(category)) & # Address occupancy is generic
        (inv_mod['Parcel_UseDescription_Hazus_Single'].isin(category)))]  # Parcel occupancy is generic
    

    
    # Assign a random number within the specified bounds
    occupancy_generated = [f"{category[0]}{random.choice(possible_vals)}" 
                      for _ in range(len(generic_tax_missing_nsi))]

    # Update the specified columns
    inv_mod.loc[generic_tax_missing_nsi.index, 'OccupancyClass_Best'] = occupancy_generated
    
    inv_mod.loc[generic_tax_missing_nsi.index, 'OccupancyClass_Best_Source'] = 'Assigned_from_General_Tax'
    return inv_mod
##########################








#########################
# Find the nearest polygon for each point
def outside_ftpt_nearest_cb(point, polygons):
    """
    This function finds the nearest census block for every point that is outside of the 
    2020 census blocks. The remainder of the anlaysis (other than the census unit reassignemnt) is
    conducted with the 2010 census, so points may be outside of the 2020 census blocks.
    """

    # Compute the distance from the point to each polygon
    distances = polygons.distance(point)

    # Find the index of the polygon with the minimum distance
    nearest_index = distances.idxmin()

    # Return nearest census block
    return polygons.loc[nearest_index]['GEOID20']
#########################



#########################
def download_census_data(census_api_key, hayward_blocks20, state_fips, county_fips):
    """
    DOWNLOAD RELEVANT 2020 CENSUS DATA USING API AND PYTHON CENSUS PACKAGE
    """

    base_url = "https://api.census.gov/data/2020/dec/pl"

    params = {
        "get": "NAME,P1_001N,H1_001N,H1_002N",  # Population, Total Units, Occupied Units
        "for": "block:*",
        "in": f"state:{state_fips} county:{county_fips}",
        "key": census_api_key
    }

    # Make the API request
    response = requests.get(base_url, params=params)

    # Process response
    if response.status_code == 200:
        data = response.json()
        cbs20 = pd.DataFrame(data[1:], columns=data[0])  # Convert JSON to DataFrame
    else:
        print("Error:", response.status_code, response.text)

    # Rename columns 
    cbs20 = cbs20.rename(columns={'P1_001N': 'POP', 
                            'H1_001N': 'UNITS', 
                            'H1_002N': 'OCCUPIED'})
    cbs20 = cbs20.drop(columns = ['NAME'])

    # Convert data types 
    cbs20['POP'] = cbs20['POP'].astype(int)
    cbs20['UNITS'] = cbs20['UNITS'].astype(int)
    cbs20['OCCUPIED'] = cbs20['OCCUPIED'].astype(int)

    # Assemble 15 digit CB code 
    cbs20['cb_code'] = cbs20[['state', 'county', 'tract', 'block']].apply(lambda row: ''.join(row.values.astype(str)), axis=1)

    # Filter for only Hayward
    cbs20 = cbs20[cbs20['cb_code'].isin(hayward_blocks20['GEOID20'].unique())]

    # Export census data
    cbs20.to_csv(f'Input_Data/Census/2020_Census_Units_{state_fips}{county_fips}.csv')

    return cbs20
#########################




#########################
def check_occupancy_class(value, types):
    """
    This function is to extract all rows containing for specific occupancy class types from the footprint NSI inventory. 
    """
    if isinstance(value, list):
        # If it's a list, check if any of the options are in the list
        return any(option in value for option in types)
    elif isinstance(value, str):
        # If it's a string, check if it's in the options list
        return value in types
    return False
#########################





#########################
def assign_units_from_censusblock(inventory, inventory_columm, cbs):
    """
    This function uses total number of units from the 2020 census for a given census block to re-assign the number of 
    units in RES3B through RES3F structures based on the NSI night population. 
    
    RES1, RES3A, and RES2 number of units are assumed to be 1, 2, and 1 respectively, and are not modified based on census data. 
    RES3A (2 units) serves as a lower bound of what RES3B through RES3F can be assigned. 
    Units_CensusEstimate is the estimated number of units 

    Flag_ModifiedByCensus is 1 or 0, depending on if the census units have been estimated (1) or not (0)
    Note_ModifiedByCensus is a note explaining the changes made to the units based on census data
    """

    # Get list of Census Blocks - Only intersted in CBs that have NSI Data in them 
    CB_list = list(inventory[inventory_columm].unique())

    # Loop through Census Blocks present in the NSI Inventory 
    for i in range(len(CB_list)):
        
        CB_ID = CB_list[i]

        # Find NSI data for current CB 
        points_nsi = inventory[inventory[inventory_columm] == CB_ID].copy()
        pop_nsi = points_nsi['NSI_Population_Night'].sum()
        res1 = points_nsi[points_nsi['NSI_OccupancyClass_Single'].str.contains('RES1')].copy()
        res2 = points_nsi[points_nsi['NSI_OccupancyClass_Single'].str.contains('RES2')].copy()
        res3a = points_nsi[points_nsi['NSI_OccupancyClass_Single'].str.contains('RES3A')].copy()
        res3 = points_nsi[points_nsi['NSI_OccupancyClass_Single'].str.contains('RES3B|RES3C|RES3D|RES3E|RES3F')].copy()

        # Set population based on NSI for current CB
        pop_res1_nsi = 0
        pop_res2_nsi = 0
        pop_res3a_nsi = 0
        pop_res3_nsi = 0
        if len(res1) > 0: 
            pop_res1_nsi = res1['NSI_Population_Night'].sum()
        if len(res2) > 0: 
            pop_res2_nsi = res2['NSI_Population_Night'].sum()
        if len(res3a) > 0:
            pop_res3a_nsi = res3a['NSI_Population_Night'].sum()
        if len(res3) > 0:
            pop_res3_nsi = res3['NSI_Population_Night'].sum()

        # Find Census data for current CB 
        pop_census = float(cbs[cbs['cb_code']==CB_ID]['POP'].values[0])
        num_occ_census = float(cbs[cbs['cb_code']==CB_ID]['OCCUPIED'].values[0])
        num_units_census = float(cbs[cbs['cb_code']==CB_ID]['UNITS'].values[0])

        # Compute Values 
        remaining_census_units = num_units_census - len(res1)  - len(res2) - 2*len(res3a)
        remaining_census_pop = pop_census - pop_res1_nsi - pop_res2_nsi - pop_res3a_nsi


        # If there are no residential points larger than a RES3A, don't make any changes
        if len(res3) == 0: 
            inventory.loc[points_nsi.index, 'Units_CensusEstimate'] = np.nan #inventory.loc[points_nsi.index, 'NSI_MinResUnits']
            inventory.loc[points_nsi.index, 'Flag_ModifiedByCensus'] = 0
            inventory.loc[points_nsi.index, 'Note_ModifiedByCensus'] = 'Nothing above RES3A, so no changes made'
        
        # If there are residential points larger than a RES3A, but RES1/RES2/RES3A have already used all Census units, change all RES3 buidlings to RES3A
        elif (len(res3) > 0) and (remaining_census_units <= 0): 
            inventory.loc[points_nsi.index, 'Units_CensusEstimate'] = np.nan # inventory.loc[points_nsi.index, 'NSI_MinResUnits']
            inventory.loc[points_nsi.index, 'Flag_ModifiedByCensus'] = 0
            inventory.loc[points_nsi.index, 'Note_ModifiedByCensus'] = 'NO MOD 1'
            inventory.loc[res3.index, 'Units_CensusEstimate'] = 2
            inventory.loc[res3.index, 'Flag_ModifiedByCensus'] = 1
            inventory.loc[res3.index, 'Note_ModifiedByCensus'] = 'All RES3 converted to RES3A because RES1/RES2/RES3A exceeding unit limit'

        else: # There are residential points larger than RES3A and remaining census units 

            # RES3 population is 0 - set units as mean of max and min 
            if pop_res3_nsi == 0: 

                # Reset census estimate units based on updates for RES3 
                inventory.loc[points_nsi.index, 'Units_CensusEstimate'] = np.nan #inventory.loc[points_nsi.index, 'NSI_MinResUnits']
                inventory.loc[points_nsi.index, 'Flag_ModifiedByCensus'] = 0
                inventory.loc[points_nsi.index, 'Note_ModifiedByCensus'] = 'NO MOD 2'
                inventory.loc[res3.index, 'Units_CensusEstimate'] = round((res3['NSI_MinResUnits'] + res3['NSI_MaxResUnits']) / 2)
                inventory.loc[res3.index, 'Flag_ModifiedByCensus'] = 1
                inventory.loc[res3.index, 'Note_ModifiedByCensus'] = 'RES3 population is 0 - set units as mean of min/max'
            
            # RES3 population is not 0 - Scale units based on nighttime population for residenital structures larger than RES3A
            else:
                census_units_per_nsi_person = remaining_census_units / pop_res3_nsi
                res3['ScaledUnits'] = round(res3['NSI_Population_Night'] * census_units_per_nsi_person)

                # Set lower bound as converting structures to 2 units (RES3A)
                res3['ScaledUnits'] = res3['ScaledUnits'].apply(lambda x: max(x, 2))

                # Reset census estimate units based on updates for RES3 
                inventory.loc[points_nsi.index, 'Units_CensusEstimate'] = np.nan #inventory.loc[points_nsi.index, 'NSI_MinResUnits']
                inventory.loc[points_nsi.index, 'Flag_ModifiedByCensus'] = 0
                inventory.loc[points_nsi.index, 'Note_ModifiedByCensus'] = 'NO MOD 3'
                inventory.loc[res3.index, 'Units_CensusEstimate'] = res3['ScaledUnits']
                inventory.loc[res3.index, 'Flag_ModifiedByCensus'] = 1
                inventory.loc[res3.index, 'Note_ModifiedByCensus'] = 'Units scaled from RES3 night population'

    # Ensure Units_CensusEstimate is na for any point not modified in this process
    inventory.loc[inventory.index.isin(inventory[inventory['Flag_ModifiedByCensus']==0].index), 'Units_CensusEstimate'] = np.nan

    # Check for points that were modified to have zero units - this indicates an error 
    mod_to_0 = inventory.loc[inventory.index.isin(inventory[inventory['Flag_ModifiedByCensus']==1].index) & (inventory['Units_CensusEstimate'] == 0)]
    if len(mod_to_0) > 0: 
        raise ValueError('Error: Points Modified to 0 Units by Census')

    # Check for points that have unit ranges but not reassigned - this indicates an error 
    res_types = ['RES1-1SNB', 'RES1-1SWB', 'RES1-2SNB', 'RES1-2SWB', 'RES1-3SNB', 'RES1-3SWB', 'RES1-SLNB', 'RES1-SLWB', 'RES1', 'RES2',' RES3A', 'RES3B', 'RES3C', 'RES3D','RES3E', 'RES3F']
    res = inventory[inventory['NSI_OccupancyClass_Single'].apply(check_occupancy_class, args=(res_types,))]
    na_units = res[(res['National_Flag']!=0) & (res['Flag_ModifiedByCensus']==0)]
    na_units_notequal = na_units[na_units['NSI_MinResUnits']!=na_units['NSI_MaxResUnits']]
    if len(na_units_notequal) > 0: 
        raise ValueError('Error: Points with Unit Ranges Not Assigned using Census')

    return inventory
#########################




##########################
def recombine_dropped_data(point0, point1, nsi_length):
    """
    This function recombines updated data with and without point data. (Also in functions_point_to_ftpt). 
    """
    dtype_reference = point0 if not point0.empty else point1
    nsi1_aligned = point1.reindex(columns=dtype_reference.columns).astype(dtype_reference.dtypes.to_dict(), errors="ignore")

    nsi = pd.concat([point0, nsi1_aligned], ignore_index=True)
    if len(nsi) != nsi_length:
        raise ValueError('NSI Points Dropped')
    return nsi
##########################






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




##########################
def reset_very_high_stories_by_units(inv_mod, stories_limit):
    """
    This function finds rows that have very high number of stories (user defined as a threshold value of stories_limit). 
    For those rows, the number of stories is reset based on the number of units and the area of the building footprint 
    This function aims to address the large overestimation of number of stories that appears in some NSI footprints. This is a 
    first attempt at addressing these points, and a more sophisticated method should be developed. 
    """

    ## BASED ON INFORMATIONFROM RENTCAFE, THE AVERAGE APARTMENT SIZE IN CALIFORNIA IS 854 SQ FT (https://www.rentcafe.com/average-rent-market-trends/us/ca/)
    ## THE EFFICIENCY RATIO (NET LIVEABLE AREA / FOOTPRINT AREA) IS ASSUMED TO BE 0.8

    stories_data = inv_mod.copy()

    # Compute estimated number of stories based on number of units 
    stories_data['Total_Area_from_Units'] = (stories_data['Units_Best'] * 854) / 0.8
    stories_data['Stories_from_Units'] = round(stories_data['Total_Area_from_Units']/stories_data['FootprintArea'])

    mask = (stories_data['NSI_NumberOfStories_Single'] > stories_limit)
    stories_data.loc[mask, 'NSI_NumberOfStories_Single'] = stories_data.loc[mask, 'Stories_from_Units']

    return stories_data['NSI_NumberOfStories_Single']
##########################






##########################
def reset_very_high_stories_to_mean(inv_mod, stories_limit):
    """
    This function finds rows that have very high number of stories (user defined as a threshold value of stories_limit). 
    For those rows, the number of stories is reset as the mean for the specified occupancy class. 
    This function aims to address the large overestimation of number of stories that appears in some NSI footprints. This is a 
    first attempt at addressing these points, and a more sophisticated method should be developed. 
    """

    # Create copy of data to modify 
    stories_data = inv_mod.copy()

    # Create dictionary of mean number of stories per occupancy class 
    stories_dict = {}

    # Create dictionary of size limit
    for occ_type in stories_data['OccupancyClass_Best'].unique():

        # Get rows with given occupancy 
        occ_assigned = stories_data[stories_data['OccupancyClass_Best'] == occ_type]

        # Get number of stories and compute mean 
        occ_stories = occ_assigned['NSI_NumberOfStories_Single']
        if occ_stories.dropna().empty:
            mean_stories = np.nan
        else:
            mean_stories = round(np.mean(occ_stories))

        # Record mean value to fill in the gaps 
        stories_dict[occ_type] = mean_stories


    # Using the above dictionary, reset stories that exceed the user-specified limit 
    mask = (stories_data['NSI_NumberOfStories_Single'] > stories_limit)
    stories_data.loc[mask, 'NSI_NumberOfStories_Single'] = stories_data.loc[mask, 'NSI_OccupancyClass_Single'].map(stories_dict)

    return stories_data['NSI_NumberOfStories_Single']
##########################









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
def compute_hazus_replacement_cost(inv_mod, hazus_conversion,include_scaling_for_contents):
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

    # Scale from structure value to replacement cost (per Hazus manual) 
    if include_scaling_for_contents: 
        inv_mod['ReplacementCost_Hazus'] = inv_mod['ReplacementCost_Hazus'] * 1.5 

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


#############
def extract_bldg_type(value):
    """
    This function updates the building type of each footprint to be consistent with the assigned structure type. 
    This is expecially important when structure type is decoupled from the original building type, such as the case of the National Synthesis Workflow
    """
    if value in ['RM1','RM2','URM']: 
        return 'M'
    elif value in ['W1','W2']: 
        return 'W'
    elif value in ['S1','S2','S3','S4','S5']:
        return 'S'
    elif value in  ['C1','C2','C3','PC1','PC2']:
        return 'C'
    elif value in ['MH']:
        return 'H'
    else: 
        return 'ERROR'
#############