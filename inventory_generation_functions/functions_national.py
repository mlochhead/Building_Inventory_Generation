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
def download_nsi(city, crs_plot):
    """
    Download NSI from API for specified city polygon 
    """
    # Specify Bounding Box for NSI Download
    def geometry_to_bbox_string(geom):
        minx, miny, maxx, maxy = geom.bounds
        return f"bbox={minx},{maxy},{maxx},{maxy},{maxx},{miny},{minx},{miny},{minx},{maxy}"
    bounding_box = city.geometry.apply(geometry_to_bbox_string)[0]

    # Headers for API
    headers = {'User-Agent': 'Mozilla/5.0'}

    # Retrieve NSI Data
    url = ("https://nsi.sec.usace.army.mil/nsiapi/structures?" + bounding_box)
    response = requests.get(url, headers=headers, timeout=30)

    # Check if the request was successful
    if response.status_code == 200:
        data = response.json()
    else: 
        print(f'Error retrieiving NSI data for county')

    # Convert the data to a gdf
    df = pd.DataFrame(data['features'])
    geometries = df['geometry'].apply(shape)
    properties_df = pd.json_normalize(df['properties'])
    gdf = gpd.GeoDataFrame(properties_df, geometry=geometries, crs="EPSG:4326")
    gdf = gdf.to_crs(crs_plot).copy()

    if len(gdf) == 0: 
        print('NO STRUCTURES FOR SPECIFIED AREA')

    # Export
    os.makedirs(f"./Input_Data/National/", exist_ok=True)
    gdf.to_file(f"./Input_Data/National/nsi_raw.geojson", driver="GeoJSON")
    print('NSI Data Exported to ./Input_Data/Nationl/nsi_raw.geojson')
##########################


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



##########################
def rename_nsi_data(centroids_NSI):
    """
    This function cleans and renames columns in the NSI (National Structure Inventory) data to more meaningful names.
    
    Input:
    - centroids_NSI: A GeoDataFrame containing NSI data with various building and population attributes.

    Steps:
    1. Drop irrelevant columns from the NSI data.
    2. Rename several columns to more descriptive names representing building and population characteristics.
    3. Sum day and night populations (over and under 65) to create total population columns for both time periods.
    4. Calculate total replacement cost by summing structure and content values.
    5. Drop the original columns after renaming and calculating necessary fields.
    
    Returns:
    - centroids_NSI: A cleaned and updated GeoDataFrame containing the necessary fields for analysis.
    """
    # Drop unnecessary columns from NSI data
    centroids_NSI.drop(['st_damcat', 'ftprntid', 
                        'firmzone', 'x', 'y', 
                        'ground_elv', 'ground_elv_m', 'val_vehic', 
                        'o65disable', 'u65disable', 'students'], 
                        axis=1, inplace=True)

    # Rename columns to more meaningful names
    centroids_NSI['NSI_FoundationType'] = centroids_NSI['found_type']
    centroids_NSI['NSI_FoundationHeight'] = centroids_NSI['found_ht']
    centroids_NSI['NSI_BuildingType'] = centroids_NSI['bldgtype']
    centroids_NSI['NSI_MedYearBuilt'] = centroids_NSI['med_yr_blt']
    centroids_NSI['NSI_fdid'] = centroids_NSI['fd_id']
    centroids_NSI['CensusBlock'] = centroids_NSI['cbfips']
    centroids_NSI['NSI_OccupancyClass'] = centroids_NSI['occtype']
    centroids_NSI['NSI_NumberOfStories'] = centroids_NSI['num_story']
    centroids_NSI['NSI_OrigSource'] = centroids_NSI['source']
    centroids_NSI['NSI_OrigFtptSource'] = centroids_NSI['ftprntsrc']
    centroids_NSI['NSI_BID'] = centroids_NSI['bid']
    centroids_NSI['NSI_TotalAreaSqFt'] = (centroids_NSI['sqft'])

    # Rename and calculate population data for night and day times
    centroids_NSI['NSI_PopOver65_Night'] = centroids_NSI['pop2amo65']
    centroids_NSI['NSI_PopUnder65_Night'] = centroids_NSI['pop2amu65']
    centroids_NSI['NSI_Population_Night'] = centroids_NSI[['pop2amu65', 'pop2amo65']].sum(axis=1)

    centroids_NSI['NSI_PopOver65_Day'] = centroids_NSI['pop2pmo65']
    centroids_NSI['NSI_PopUnder65_Day'] = centroids_NSI['pop2pmu65']
    centroids_NSI['NSI_Population_Day'] = centroids_NSI[['pop2pmu65', 'pop2pmo65']].sum(axis=1)

    # Rename value-related columns and calculate replacement cost
    centroids_NSI['NSI_ContentValue'] = centroids_NSI['val_cont']
    centroids_NSI['NSI_StructureValue'] = centroids_NSI['val_struct']
    centroids_NSI['NSI_ReplacementCost'] = centroids_NSI[['val_struct', 'val_cont']].sum(axis=1)

    # Drop the original columns that were renamed or recalculated
    centroids_NSI = centroids_NSI.drop(columns=['pop2amu65', 'pop2amo65', 'pop2pmu65', 'pop2pmo65', 
                                                'val_cont', 'val_struct', 'cbfips', 'occtype', 
                                                'num_story', 'found_type', 'found_ht', 
                                                'bldgtype', 'med_yr_blt', 'fd_id', 'sqft', 
                                                'source', 'ftprntsrc', 'bid'])

    # Return cleaned NSI data
    return centroids_NSI
##########################


##########################
def assign_point_block_and_track(nsi, city_blocks, city_tracts): 
    """
    This function assigns Census Block and Tract information to NSI data via spatial joins with city-specific blocks and tracts. 
    It resolves conflicts between pre-existing NSI block assignments and the spatial join to ensure accuracy.

    Inputs:
    - nsi: GeoDataFrame with NSI data, including pre-existing Census Block assignments.
    - city_blocks: GeoDataFrame for city-specific Census blocks.
    - city_tracts: GeoDataFrame for city-specific Census tracts.

    Process:
    1. Spatially joins NSI data with city Census blocks and checks for conflicts.
    2. Resolves conflicts by filling missing blocks or fixing mismatches.
    3. Removes centroids outside the study area.
    4. Joins NSI data with city Census tracts.

    Output:
    - nsi: Cleaned GeoDataFrame with accurate Census Block and Tract assignments.
    """
    # Merge NSI data with City-Specific Census Blocks 
    raw = nsi.sjoin(city_blocks[['GEOID10','geometry']], how='left')

    # Post-Process Results 
    # NSI data already has census block infomration assigned, so the following code compares this 
    # information with the spatially joined census blocks

    # Select points where the CB doesn't match between NSI and spatial join
    # Note that every other case matches and those are not modified
    A = raw[~(raw['GEOID10'] == raw['CensusBlock'])]
    print('Points where Census Block Does Not Match between NSI and Spatial Join (Including Outside Study Area):',len(A)) 

    # Now select the ones in A where the NSI CensusBlock is an empty string
    # And update the CensusBlock in those with the one based on spatial join
    B = A[(A['CensusBlock']=='')]
    B_complement = A[~(A['CensusBlock']=='')]
    raw.loc[B.index, 'CensusBlock'] = raw.loc[B.index, 'GEOID10']
    print('Points Missing CB in NSI Data  (Filled Using Spatial Join):',len(B))

    # Now move on with the ones in A where the NSI CensusBlock was not an empty string
    # And check if any of those NSI CensusBlocks are within the list of considered census blocks
    # For these, we assume that the location in NSI is wrong and the CensusBlock information is right
    C = B_complement[B_complement['CensusBlock'].isin(city_blocks['GEOID10'].values)]
    print('Conflicting Points within CBs Considered in Study (Assigned via Spatial Join):',len(C))

    # What remains is a set with NSI Census blocks that are not among the ones we consider in the study
    # Check that these are indeed not within the space considered in the study
    C_complement = B_complement[~B_complement['CensusBlock'].isin(city_blocks['GEOID10'].values)]
    D = C_complement[~pd.isna(C_complement['GEOID10'])]

    if D.shape[0] > 0:
        print(f'WARNING: Some NSI Census block conflicts were not resolved -- {len(D)} points dropped')
        
    # Remove the centroids that are not within the space considered in the study
    raw = raw.drop(C_complement.index, axis=0)

    # if there was no error above, we are done with this check and we can remove the spatially joined columns
    nsi = raw.drop(['GEOID10'], axis=1)

    # Assign to census tracts
    nsi = nsi.drop(columns = ['index_right'])
    nsi_copy = nsi.copy()
    nsi_copy = nsi_copy.sjoin(city_tracts, how='left')
    nsi.loc[:, 'CensusTract'] = nsi_copy['GEOID10'].values

    # Return 
    return nsi
##########################


##########################
def format_and_locate_edu1(public, private, city_tracts, city_blocks):
    """
    This function processes public and private school data to assign Census Block and Tract information within a city.
    It formats the school data to match the NSI format and combines public and private schools into a single dataset.

    Inputs:
    - public: GeoDataFrame of public school data, downloaded from HIFLD.
    - private: GeoDataFrame of private school data, downloaded from HIFLD.
    - city_tracts: GeoDataFrame of city Census tracts.
    - city_blocks: GeoDataFrame of city Census blocks.

    Outputs:
    - public_city: GeoDataFrame of public schools within the city, including Census Block and Tract information.
    - private_city: GeoDataFrame of private schools within the city, including Census Block and Tract information.
    - school_import: Combined GeoDataFrame of all schools formatted to match NSI.
    """
    # Sort for only schools within city boundaries 
    public_city = public.sjoin(city_blocks[['GEOID10','geometry']], how='inner')
    private_city = private.sjoin(city_blocks[['GEOID10','geometry']], how='inner')

    # Assign Census Block 
    public_city = public_city.drop(columns = ['index_right'])
    public_city = public_city.rename(columns={'GEOID10':'CensusBlock'})
    private_city = private_city.drop(columns = ['index_right'])
    private_city = private_city.rename(columns={'GEOID10':'CensusBlock'})

    # Assign Census Tract 
    public_city_copy = public_city.copy().sjoin(city_tracts, how='left')
    public_city.loc[:, 'CensusTract'] = public_city_copy['GEOID10'].values
    private_city_copy = private_city.copy().sjoin(city_tracts, how='left')
    private_city.loc[:, 'CensusTract'] = private_city_copy['GEOID10'].values

    # Drop extraneous columns
    public_city = public_city.drop(columns=['OBJECTID','ADDRESS','CITY','STATE','ZIP','ZIP4','TELEPHONE','WEBSITE',
                                                'ST_GRADE','END_GRADE','DISTRICTID','SHELTER_ID','COUNTY','COUNTYFIPS',
                                                'COUNTRY','LATITUDE','LONGITUDE','SOURCE','SOURCE_DATE','VAL_METHOD','VAL_DATE',
                                                'TYPE','STATUS','NAICS_CODE','NAICS_DESC','ENROLLMENT','FT_TEACHERS'])
    private_city = private_city.drop(columns=['OBJECTID','ADDRESS','CITY','STATE','ZIP','ZIP4','TELEPHONE','WEBSITE',
                                                'ST_GRADE','END_GRADE','SHELTER_ID','COUNTY','COUNTYFIPS',
                                                'COUNTRY','LATITUDE','LONGITUDE','SOURCE','SOURCE_DATE','VAL_METHOD','VAL_DATE',
                                                'TYPE','STATUS','NAICS_CODE','NAICS_DESC','ENROLLMENT','FT_TEACHERS'])
    # Assign Occupancy Type 
    public_city['PUBLIC'] = 'EDU1-PUB'
    private_city['PUBLIC'] = 'EDU1-PRIV'
    

    # Concatenate school dataframes 
    school_import = pd.concat([public_city, private_city])

    # Format school_impot to match NSI format 
    school_import['NSI_PopOver65_Night'] = 0
    school_import['NSI_PopUnder65_Night'] = 0
    school_import['NSI_Population_Night'] = 0
    school_import['NSI_PopOver65_Day'] = 0
    school_import['NSI_PopUnder65_Day'] = school_import['POPULATION']
    school_import['NSI_Population_Day'] = school_import['POPULATION']
    school_import['NSI_OccupancyClass'] = school_import['PUBLIC']
    school_import['NSI_DropFlag'] = 0
    school_import['NSI_Source'] = 'HIFLD'
    school_import = school_import.drop(columns = ['NCESID','POPULATION','LEVEL_','PUBLIC'])

    # Remove negateive population values for all rows
    school_import['NSI_PopUnder65_Day'] = school_import['NSI_PopUnder65_Day'].mask(school_import['NSI_PopUnder65_Day'] < 0, np.nan)
    school_import['NSI_PopOver65_Day'] = school_import['NSI_PopOver65_Day'].mask(school_import['NSI_PopOver65_Day'] < 0, np.nan)
    school_import['NSI_PopUnder65_Night'] = school_import['NSI_PopUnder65_Night'].mask(school_import['NSI_PopUnder65_Night'] < 0, np.nan)
    school_import['NSI_PopUnder65_Night'] = school_import['NSI_PopUnder65_Night'].mask(school_import['NSI_PopUnder65_Night'] < 0, np.nan)
    school_import['NSI_Population_Night'] = school_import['NSI_Population_Night'].mask(school_import['NSI_Population_Night'] < 0, np.nan)
    school_import['NSI_Population_Day'] = school_import['NSI_Population_Day'].mask(school_import['NSI_Population_Day'] < 0, np.nan)

    
    # Print number of schools
    print('Public Schools:', len(public_city))
    print('Private Schools:', len(private_city))

    # Reset index
    school_import = school_import.reset_index(drop=True)

    return public_city, private_city, school_import
##########################



##########################
def synthesize_edu1_and_HIFLD(nsi, school_import, crs_plot, plot_flag, drop_unpaired_nsi_edu1, drop_gov1_near_edu1):
    """
    This function integrates NSI EDU1 points with HIFLD school data. It identifies NSI points with occupancy class 'EDU1' and spatially joins them with HIFLD school
    data within a 50-meter radius. Where overlap exists, it prioritizes HIFLD daytime population data and NSI
    nighttime and over-65 data (which are set to 0 for HIFLD points). Matched points are merged into the HIFLD dataset, and unmatched NSI EDU1 points
    are flagged for removal. It also removes GOV1 points within 50 meters of any imported EDU1 point to avoid duplication.

    If `plot_flag` is True, a map is generated showing original EDU1 points (black), HIFLD school points (blue), and matched/merged points (red).
    """

    # Create NSI_MedYearBuilt column if not already present
    if 'NSI_MedYearBuilt' not in school_import.columns: 
        school_import['NSI_MedYearBuilt'] = np.nan
        
    # Separate NSI EDU1 points 
    edu1 = nsi[nsi['NSI_OccupancyClass'] == 'EDU1']

    # Drop duplicated columns 
    edu1 = edu1.drop(columns=['CensusBlock','CensusTract','NSI_OccupancyClass','NSI_DropFlag','NSI_Source'])

    ## Find NSI EDU points that correspond to HIFLD EDU points 
    nsi_edu1_near_hifld = gpd.sjoin_nearest(
        edu1,
        school_import,
        how="inner",
        max_distance=50, 
        distance_col="distance",
        lsuffix="nsi",
        rsuffix="hifld")
    
    # Keep HIFLD pairings with closest NSI EDU1 point
    nsi_edu1_near_hifld = nsi_edu1_near_hifld.loc[nsi_edu1_near_hifld.groupby('index_hifld')['distance'].idxmin()]

    # Drop duplicates (happens in case of distance being the same for multiple points)
    nsi_edu1_near_hifld = nsi_edu1_near_hifld.drop_duplicates(subset='index_hifld', keep='first')
    
    # Prioritize HIFLD Year Built if specified; otherwise, list NSI year built 
    nsi_edu1_near_hifld['NSI_MedYearBuilt'] = nsi_edu1_near_hifld['NSI_MedYearBuilt_hifld'].fillna(nsi_edu1_near_hifld['NSI_MedYearBuilt_nsi'])
    
    # Prioritize HIFLD Population Data if specified; otherwise, list NSI population for NSI_Population_Day and NSI_PopUnder65_Day
    nsi_edu1_near_hifld['NSI_Population_Day'] = nsi_edu1_near_hifld['NSI_Population_Day_hifld'].fillna(nsi_edu1_near_hifld['NSI_Population_Day_nsi'])
    nsi_edu1_near_hifld['NSI_PopUnder65_Day'] = nsi_edu1_near_hifld['NSI_PopUnder65_Day_hifld'].fillna(nsi_edu1_near_hifld['NSI_PopUnder65_Day_nsi'])

    # For nighttime and over 65 day population, prioritize NSI (HIFLD processing sets all of these to zero)
    for col in ['NSI_PopUnder65_Night','NSI_PopOver65_Night','NSI_Population_Night','NSI_PopOver65_Day']:
        nsi_edu1_near_hifld[col] = nsi_edu1_near_hifld[col + '_nsi'].fillna(nsi_edu1_near_hifld[col + '_hifld'])

    # Drop additional columns 
    nsi_edu1_near_hifld = nsi_edu1_near_hifld.drop(columns=['NSI_PopUnder65_Night_nsi',
                                                'NSI_PopUnder65_Night_hifld',
                                                'NSI_PopOver65_Night_nsi',
                                                'NSI_PopOver65_Night_hifld',
                                                'NSI_Population_Night_nsi',
                                                'NSI_Population_Night_hifld',
                                                'NSI_PopOver65_Day_nsi',
                                                'NSI_PopOver65_Day_hifld',
                                                'NSI_PopUnder65_Day_nsi',
                                                'NSI_PopUnder65_Day_hifld',
                                                'NSI_Population_Day_nsi',
                                                'NSI_Population_Day_hifld',
                                                'NSI_MedYearBuilt_nsi',
                                                'NSI_MedYearBuilt_hifld'
                                                ])


    ## Assemble all HIFLD Data (data with and without NSI agumentation) for merge 
    nsi_edu1_near_hifld['NSI_DataUpdate']='HIFLD_AND_NSI_EDU'
    remaining_school_import = school_import[~school_import.index.isin(nsi_edu1_near_hifld['index_hifld'])].copy()
    remaining_school_import['NSI_DataUpdate']='Only_HIFLD_EDU'
    school_import_w_nsi = pd.concat([remaining_school_import,nsi_edu1_near_hifld])
    if len(school_import_w_nsi) != len(school_import):
        raise ValueError('HIFLD School Data Dropped')


    # Mark EDU1 points to be dropped from NSI (data from close points already associated with HIFLD points)
    edu1_absorbed = nsi[
        (nsi['NSI_fdid'].isin(nsi_edu1_near_hifld['NSI_fdid'].unique())) & # Not paired with HIFLD data
        (nsi['NSI_OccupancyClass'] == 'EDU1')].index # Occupancy class is EDU1
    nsi.loc[edu1_absorbed,'NSI_DropFlag']=1
    nsi.loc[edu1_absorbed,'NSI_DropNote']='NSI EDU1 Points <50m from HIFLD Absorbed by HIFLD'

    if drop_unpaired_nsi_edu1:
        edu1_to_drop = nsi[
            (~nsi['NSI_fdid'].isin(nsi_edu1_near_hifld['NSI_fdid'].unique())) & # Not paired with HIFLD data
            (nsi['NSI_OccupancyClass'] == 'EDU1')].index # Occupancy class is EDU1
        nsi.loc[edu1_to_drop,'NSI_DropFlag']=1
        nsi.loc[edu1_to_drop,'NSI_DropNote']='NSI EDU1 Points >50m from HIFLD Dropped'


    # Merge new HIFLD points into NSI Dataframe 
    nsi = pd.concat([nsi,school_import_w_nsi])
    nsi['NSI_NumPoints'] = 1

    # Flag all GOV1 Points within 50m of a newly imported EDU1 point 
    if drop_gov1_near_edu1:
        gov1_near_edu1 = find_gov1_near_hifld(nsi.copy(), 'EDU1-PUB',50)
        nsi.loc[gov1_near_edu1.index,'NSI_DropFlag']=1
        nsi.loc[gov1_near_edu1.index,'NSI_DropNote']='GOV1 Point within 50m of HIFLD School'

    # Drop additional columns 
    nsi = nsi.drop(columns=['distance','index_hifld'])

    # Plot if requested 
    if plot_flag: 

        # Create a base map
        m = folium.Map(location=[edu1.copy().to_crs(crs_plot).geometry.iloc[0].y, edu1.copy().to_crs(crs_plot).geometry.iloc[0].x],zoom_start=12)
        
        # Add remaining points     
        for idx, row in edu1.copy().to_crs(crs_plot).iterrows():
            folium.CircleMarker(location=[row.geometry.y, row.geometry.x], 
                                radius=3, 
                                color='black', 
                                fill=True, 
                                fill_color='black').add_to(m)
            
        for idx, row in school_import.copy().to_crs(crs_plot).iterrows():
            folium.CircleMarker(location=[row.geometry.y, row.geometry.x], 
                                radius=3, 
                                color='blue', 
                                fill=True, 
                                fill_color='blue').add_to(m)

        for idx, row in nsi_edu1_near_hifld.copy().to_crs(crs_plot).iterrows():
            folium.CircleMarker(location=[row.geometry.y, row.geometry.x], 
                                radius=7, 
                                color='red', 
                                fill=True, 
                                fill_color='red').add_to(m)


        
    else:
        m = 'No Plot Requested'

    # Return
    return nsi, m
##########################


##########################
def find_gov1_near_hifld(nsi, hifld_occ, buffer):
    """
    Locates GOV1 points within specified distance of a specfied occupancy class
    Input:
    - nsi: GeoDataFrame with NSI data.
    Output:
    - nsi: GeoDataFrame with GOV1 near specified occupancy class (to be dropped from NSI)
    """
    gov1_points = nsi[nsi['NSI_OccupancyClass'] == 'GOV1']
    hifld_points = nsi[nsi['NSI_OccupancyClass'] == hifld_occ]

    # Create a buffer around EDU1-PUB points with a radius of 50 meters
    hifld_points_buffer = hifld_points.copy()
    hifld_points_buffer['geometry'] = hifld_points_buffer.geometry.buffer(buffer)

    # Perform a spatial join to find GOV1 points within 100 meters of EDU1-PUB
    gov1_near_edu1_pub = gpd.sjoin(gov1_points, hifld_points_buffer, how='inner', predicate='within')

    # Return
    return gov1_near_edu1_pub
##########################


##########################
def locate_edu2(univ, univ_pts, city_tracts, city_blocks):
    """
    This function processes university campus polygons and points, assigning Census Block and Tract information within city boundaries.

    Inputs:
    - univ: GeoDataFrame of university campus polygons, downloaded from HIFLD.
    - univ_pts: GeoDataFrame of university points, downloaded from HIFLD.
    - city_tracts: GeoDataFrame of city Census tracts.
    - city_blocks: GeoDataFrame of city Census blocks.

    Output:
    - univ_city: GeoDataFrame of university polygons with assigned Census Block and Tract information.
    - univ_pts_city: GeoDataFrame of university points with assigned Census Block and Tract information.
    """
    # Sort for only schools within city boundaries 
    univ_city = univ.sjoin(city_blocks[['GEOID10','geometry']], how='inner')
    univ_pts_city = univ_pts.sjoin(city_blocks[['GEOID10','geometry']], how='inner')

    # Drop duplicate polygons (based on geometry, total enrollement, and source date being duplicated)
    univ_city = univ_city.drop_duplicates(subset=["geometry", "TOT_ENROLL","SOURCEDATE"])

    print('Campus Polygons:',len(univ_city))
    print('College/Univeristy Points:',len(univ_pts_city))

    # Drop and rename columns for Census Block
    univ_city = univ_city.drop(columns = ['index_right'])
    univ_city = univ_city.rename(columns={'GEOID10':'CensusBlock'})
    univ_pts_city = univ_pts_city.drop(columns = ['index_right'])
    univ_pts_city = univ_pts_city.rename(columns={'GEOID10':'CensusBlock'})

    # Assign tract based on Centroid
    univ_city_centroids = univ_city.copy()
    univ_city_centroids.geometry = univ_city.geometry.centroid
    univ_city_centroids = univ_city_centroids.sjoin(city_tracts, how='left')
    univ_city.loc[:, 'CensusTract'] = univ_city_centroids['GEOID10'].values

    # Assign tract for points 
    univ_pts_city_copy = univ_pts_city.copy().sjoin(city_tracts, how='left')
    univ_pts_city.loc[:, 'CensusTract'] = univ_pts_city_copy['GEOID10'].values

    # Return
    return univ_city, univ_pts_city
##########################


##########################
def prepare_pts_without_campuses(univ_pts_city, univ_city, nsi):
    """
    This function identifies university points without associated campus polygons, prepares their data, and merges them into the NSI dataset.

    Inputs:
    - univ_pts_city: GeoDataFrame of university points within the city, downloaded from HIFLD.
    - univ_city: GeoDataFrame of university polygons within the city, downloaded from HIFLD.
    - nsi: GeoDataFrame of the NSI data.

    Output:
    - edu2_no_poly: GeoDataFrame with additional university points that do not have associated polygons to be merged into NSI
    """
    # Find school points that do not have associated polygons 
    merged = univ_pts_city.sjoin(univ_city[['UNIQUEID','POPULATION','TOT_ENROLL','TOT_EMP','geometry']], how='left', predicate='intersects')
    points_not_intersecting = merged[merged['index_right'].isna()]

    # Drop Columns 
    points_not_intersecting = points_not_intersecting.drop(columns = ['OBJECTID','IPEDSID', 'NAME', 'ADDRESS', 'CITY', 'STATE', 'ZIP',
                                                        'ZIP4', 'TELEPHONE', 'TYPE', 'STATUS', 'COUNTY','LEVEL_',
                                                        'COUNTYFIPS', 'COUNTRY', 'LATITUDE', 'LONGITUDE', 'NAICS_CODE',
                                                        'NAICS_DESC', 'SOURCE', 'SOURCEDATE', 'VAL_METHOD', 'VAL_DATE',
                                                        'WEBSITE', 'STFIPS', 'COFIPS', 'SECTOR','HI_OFFER', 'DEG_GRANT', 'LOCALE',
                                                        'CLOSE_DATE', 'MERGE_ID', 'ALIAS', 'SIZE_SET', 'INST_SIZE', 'PT_ENROLL',
                                                        'FT_ENROLL', 'TOT_ENROLL_left', 'HOUSING', 'DORM_CAP', 'TOT_EMP_left',
                                                        'SHELTER_ID','index_right', 'UNIQUEID', 'POPULATION_right', 'TOT_ENROLL_right','TOT_EMP_right'])

    # Prepare those points to be added to NSI Data 
    edu2_no_poly = points_not_intersecting.copy()
    edu2_no_poly = edu2_no_poly.drop(columns = 'POPULATION_left')

    # Create population data based on rules from HAZUS 6.0 inventory manual 
    edu2_no_poly['NSI_PopOver65_Day'] = round(points_not_intersecting['POPULATION_left']*0.005)
    edu2_no_poly['NSI_PopUnder65_Day'] = round(points_not_intersecting['POPULATION_left']*0.995)
    edu2_no_poly['NSI_Population_Day'] = edu2_no_poly['NSI_PopOver65_Day'] + edu2_no_poly['NSI_PopUnder65_Day']
    edu2_no_poly['NSI_PopUnder65_Night'] = round(edu2_no_poly['NSI_PopUnder65_Day'] * 0.005)
    edu2_no_poly['NSI_PopOver65_Night'] = round(edu2_no_poly['NSI_PopOver65_Day'] * 0.005)
    edu2_no_poly['NSI_Population_Night'] = edu2_no_poly['NSI_PopOver65_Night'] + edu2_no_poly['NSI_PopUnder65_Night'] 

    # Add remaining fields for NSI 
    edu2_no_poly['NSI_OccupancyClass'] = 'EDU2'
    edu2_no_poly['NSI_DropFlag'] = 0
    edu2_no_poly['NSI_Source'] = 'HIFLD'

    # Return 
    return edu2_no_poly
#########################


##########################
def prepare_pts_without_gov1(univ_pts_city, univ_city, nsi):
    """
    This function merges HIFLD school points that contain campus polygons (but have no GOV1 points in the polygons) into the NSI inventory.

    Inputs:
        - univ_pts_city (GeoDataFrame): School point locations within a city.
        - univ_city (GeoDataFrame): School building polygons within a city.
        - nsi (GeoDataFrame

    Process:
        - Identifies school polygons without GOV1 points.
        - Finds school points within those polygons.
        - Prepares and assigns population data based on HAZUS rules (Inventory 6.0).
        - Adds school points to the NSI inventory as 'EDU2'.

    Returns:
        - edu2_no_poly: GeoDataFrame with HIFLD points with campus polygons that do not contain any GOV1 points inside the polygon to be merged into NSI
    """
    # Find school polygons that do not have associated GOV1 points  
    gov1 = nsi[nsi['NSI_OccupancyClass']=='GOV1']
    polygons_with_nsi = gpd.sjoin(univ_city, gov1, how='inner', predicate='contains')
    polygons_without_nsi = univ_city[~univ_city.index.isin(polygons_with_nsi.index)]

    # Find imported school points in polygons that don't have associated NSI points
    school_points_without_nsi = univ_pts_city.sjoin(polygons_without_nsi[['UNIQUEID','POPULATION','TOT_ENROLL','TOT_EMP','geometry']], how='inner', predicate='intersects')

    # Drop Columns 
    school_points_without_nsi = school_points_without_nsi.drop(columns = ['OBJECTID','IPEDSID', 'NAME', 'ADDRESS', 'CITY', 'STATE', 'ZIP',
                                                        'ZIP4', 'TELEPHONE', 'TYPE', 'STATUS', 'COUNTY','LEVEL_',
                                                        'COUNTYFIPS', 'COUNTRY', 'LATITUDE', 'LONGITUDE', 'NAICS_CODE',
                                                        'NAICS_DESC', 'SOURCE', 'SOURCEDATE', 'VAL_METHOD', 'VAL_DATE',
                                                        'WEBSITE', 'STFIPS', 'COFIPS', 'SECTOR','HI_OFFER', 'DEG_GRANT', 'LOCALE',
                                                        'CLOSE_DATE', 'MERGE_ID', 'ALIAS', 'SIZE_SET', 'INST_SIZE', 'PT_ENROLL',
                                                        'FT_ENROLL', 'TOT_ENROLL_left', 'HOUSING', 'DORM_CAP', 'TOT_EMP_left',
                                                        'SHELTER_ID','index_right', 'UNIQUEID', 'POPULATION_right', 'TOT_ENROLL_right','TOT_EMP_right'])

    # Prepare those points to be added to NSI Data 
    edu2_no_poly = school_points_without_nsi.copy()
    edu2_no_poly = edu2_no_poly.drop(columns = 'POPULATION_left')

    # Create population data based on rules from HAZUS 6.0 inventory manual 
    edu2_no_poly['NSI_PopOver65_Day'] = round(school_points_without_nsi['POPULATION_left']*0.005)
    edu2_no_poly['NSI_PopUnder65_Day'] = round(school_points_without_nsi['POPULATION_left']*0.995)
    edu2_no_poly['NSI_Population_Day'] = edu2_no_poly['NSI_PopOver65_Day'] + edu2_no_poly['NSI_PopUnder65_Day']
    edu2_no_poly['NSI_PopUnder65_Night'] = round(edu2_no_poly['NSI_PopUnder65_Day'] * 0.005)
    edu2_no_poly['NSI_PopOver65_Night'] = round(edu2_no_poly['NSI_PopOver65_Day'] * 0.005)
    edu2_no_poly['NSI_Population_Night'] = edu2_no_poly['NSI_PopOver65_Night'] + edu2_no_poly['NSI_PopUnder65_Night'] 

    # Add remaining fields for NSI
    edu2_no_poly['NSI_OccupancyClass'] = 'EDU2'
    edu2_no_poly['NSI_DropFlag'] = 0
    edu2_no_poly['NSI_Source'] = 'HIFLD'

    # Return
    return edu2_no_poly 
##########################


##########################
def merge_pts_with_campuses(univ_city, nsi, scale_edu2_pop):
    """
    Converts GOV1 points within campus polygons into the EDU2 occupancy class and scales population.

    Inputs:
        - univ_city (GeoDataFrame): School building polygons within a city.
        - nsi (GeoDataFrame): NSI inventory data.

    Process:
        - Identifies GOV1 points within campus boundaries and reassigns them to EDU2.
        - Associates each NSI point with its respective campus polygon.
        - Scales the population of EDU2 points based on campus population targets.
        - Resets other population characteristics (day/night populations) for EDU2 points.

    Returns:
        - nsi (GeoDataFrame): Updated NSI inventory with scaled population and reclassified GOV1 points as EDU2.
    """
    # Find NSI GOV1 points within boundaries of campuses  
    within_polygons = nsi.within(univ_city.unary_union)
    gov1_in_polygons = (nsi['NSI_OccupancyClass'] == 'GOV1') & within_polygons

    # Update Occupancy Class and source for those points
    nsi.loc[gov1_in_polygons, 'NSI_OccupancyClass'] = 'EDU2'
    nsi.loc[gov1_in_polygons, 'NSI_DataUpdate'] = 'GOV1 Point within Campus Polyon Convereted to EDU2'
    nsi.loc[gov1_in_polygons, 'NSI_Source'] = 'HIFLD'

    # Scale university population to match HIFLD-specified enrollement: 
    if scale_edu2_pop: 
        # Internal function to find campus polygon associated with each row of NSI data
        def find_polygon_id(row, polygons):
            matches = polygons[polygons.contains(row.geometry)]
            if not matches.empty:
                return matches.index[0]
            else:
                return None

        nsi['polygon_id'] = nsi.apply(find_polygon_id, polygons=univ_city, axis=1)

        # Group by polygon and sum the current population of EDU2 points
        current_population_by_polygon = nsi.groupby('polygon_id')['NSI_Population_Day'].sum()

        # Merge the target population from polygons with the current population
        population_scaling_factors = univ_city['POPULATION'] / current_population_by_polygon

        # Ensure that the population scaling factors are properly aligned and handle NaN values
        scaling_factors_mapped = nsi['polygon_id'].map(population_scaling_factors).fillna(1)

        # Scale the population of each EDU2 point
        nsi.loc[:, 'NSI_Population_Day'] *= scaling_factors_mapped

        # Reset Other Population Characteristics for EDU2 points 
        nsi = nsi.reset_index(drop=True)
        edu2_indices = nsi[nsi['NSI_OccupancyClass'] == 'EDU2'].index
        nsi.loc[edu2_indices, 'NSI_PopUnder65_Day'] = round(nsi.loc[edu2_indices, 'NSI_Population_Day']*0.995)
        nsi.loc[edu2_indices, 'NSI_PopOver65_Day'] = round(nsi.loc[edu2_indices, 'NSI_Population_Day']*0.005)
        nsi.loc[edu2_indices, 'NSI_PopUnder65_Night'] = round(nsi.loc[edu2_indices, 'NSI_PopUnder65_Day']*0.005)
        nsi.loc[edu2_indices, 'NSI_PopOver65_Night'] = round(nsi.loc[edu2_indices, 'NSI_PopOver65_Day']*0.005)
        nsi.loc[edu2_indices, 'NSI_Population_Night'] = nsi.loc[edu2_indices, 'NSI_PopUnder65_Night'] + nsi.loc[edu2_indices, 'NSI_PopOver65_Night']
        nsi['NSI_Population_Day'] = nsi['NSI_Population_Day'].round()

        # Drop additional column 
        nsi = nsi.drop(columns=['polygon_id'])

    # Return
    return nsi
##########################


##########################
def assign_census_hifld(gdf, city_blocks, city_tracts):
    """
    Assigns Census Block and Tract information to a GeoDataFrame using spatial joins with city blocks and tracts.
    This is used for fire, emergency, and police dataframes, imported from HIFLD. 
    """
    # Spatial join with city boundaries
    gdf = gdf.sjoin(city_blocks[['GEOID10', 'geometry']], how='inner')
    
    # Drop and rename columns for Census Block
    gdf = gdf.drop(columns=['index_right'])
    gdf = gdf.rename(columns={'GEOID10': 'CensusBlock'})
    
    # Assign Census Tract
    gdf_copy = gdf.copy().sjoin(city_tracts, how='left')
    gdf['CensusTract'] = gdf_copy['GEOID10'].values
    
    return gdf
##########################



##########################
def synthesize_gov2_and_HIFLD(nsi, new_gov2, crs_plot, plot_flag, drop_unpaired_nsi_gov2, drop_gov1_near_gov2):
    """
    This function integrates NSI EDU1 points with HIFLD government data. It identifies NSI points with occupancy class 'GOV2' and spatially joins them with HIFLD
    data within a 50-meter radius. Only one NSI point per HIFLD points is paired and vice versa (based on closest point). Matched points are merged into the HIFLD dataset, 
    and unmatched NSI GOV2 points are flagged for removal. It also removes GOV1 points within 10 meters of any imported GOV2 point to avoid duplication.

    If `plot_flag` is True, a map is generated showing original GOV2 points and HIFLD points 
    """
    # Separate out NSI GOV2 points and remove occupancy information (will be replaced with HIFLD)
    gov2 = nsi[nsi['NSI_OccupancyClass']=='GOV2']
    gov2 = gov2.drop(columns=['NSI_OccupancyClass','NSI_DropFlag','NSI_Source'])

    # Reset index for merge
    new_gov2 = new_gov2.reset_index(drop=True)

    ## Find NSI GOV2 points that correspond to HIFLD EDU points 
    nsi_gov2_near_hifld = gpd.sjoin_nearest(
        gov2,
        new_gov2[['NSI_OccupancyClass','geometry']],
        how="inner",
        max_distance=50, 
        distance_col="distance",
        lsuffix="nsi",
        rsuffix="hifld")

    # Keep only closest points 
    nsi_gov2_near_hifld = nsi_gov2_near_hifld.loc[nsi_gov2_near_hifld.groupby('index_hifld')['distance'].idxmin()]

    # Drop duplicates (happens in case of distance being the same for multiple points)
    nsi_gov2_near_hifld = nsi_gov2_near_hifld.drop_duplicates(subset='index_hifld', keep='first')

    ## Assemble all HIFLD Data (data with and without NSI agumentation) for merge 
    nsi_gov2_near_hifld['NSI_DataUpdate']='HIFLD_AND_NSI_GOV2'
    remaining_gov2_import = new_gov2[~new_gov2.index.isin(nsi_gov2_near_hifld['index_hifld'])].copy()
    remaining_gov2_import['NSI_DataUpdate']='Only_HIFLD_GOV2'
    gov2_import_w_nsi = pd.concat([remaining_gov2_import,nsi_gov2_near_hifld])
    gov2_import_w_nsi['NSI_DropFlag'] = 0
    gov2_import_w_nsi['NSI_Source'] = 'HIFLD'
    if len(gov2_import_w_nsi) != len(new_gov2):
        raise ValueError('HIFLD GOV2 Data Dropped')

    # Mark GOV2 points to be dropped from NSI (data from close points already associated with HIFLD points)
    gov2_absorbed = nsi[
        (nsi['NSI_fdid'].isin(nsi_gov2_near_hifld['NSI_fdid'].unique())) & # Not paired with HIFLD data
        (nsi['NSI_OccupancyClass'] == 'GOV2')].index # Occupancy class is EDU1
    nsi.loc[gov2_absorbed,'NSI_DropFlag']=1
    nsi.loc[gov2_absorbed,'NSI_DropNote']='NSI GOV2 Points <50m from HIFLD Absorbed by HIFLD'

    if drop_unpaired_nsi_gov2: 
        gov2_to_drop = nsi[
            (~nsi['NSI_fdid'].isin(nsi_gov2_near_hifld['NSI_fdid'].unique())) & # Not paired with HIFLD data
            (nsi['NSI_OccupancyClass'] == 'GOV2')].index # Occupancy class is EDU1
        nsi.loc[gov2_to_drop,'NSI_DropFlag']=1
        nsi.loc[gov2_to_drop,'NSI_DropNote']='NSI GOV2 Points >50m from HIFLD or Duplicated Dropped'


    # Merge new HIFLD points into NSI Dataframe 
    nsi = pd.concat([nsi,gov2_import_w_nsi])
    nsi['NSI_NumPoints'] = 1

    # Flag all GOV1 Points within 10m of a newly imported GOV2 point 
    if drop_gov1_near_gov2: 
        gov1_near_gov2 = find_gov1_near_hifld(nsi.copy(), 'GOV2',10)
        nsi.loc[gov1_near_gov2.index,'NSI_DropFlag']=1
        nsi.loc[gov1_near_gov2.index,'NSI_DropNote']='GOV1 Point within 10m of HIFLD GOV2'

    # Drop additional columns 
    nsi = nsi.drop(columns=['distance','index_hifld'])


    if plot_flag: 

        ### UNCOMMENT CODE TO PLOT INTERACTIVE MAP OF HIFLD POINTS AGAINST GOV2 NSI POINTS 
        # Create a base map
        m = folium.Map(location=[gov2.copy().to_crs(crs_plot).geometry.iloc[0].y, gov2.copy().to_crs(crs_plot).geometry.iloc[0].x],zoom_start=12)
        
        # Add remaining points     

        for idx, row in gov2.copy().to_crs(crs_plot).iterrows():
            folium.CircleMarker(location=[row.geometry.y, row.geometry.x], 
                                radius=3, 
                                color='black', 
                                fill=True, 
                                fill_color='black').add_to(m)
            
        for idx, row in new_gov2[new_gov2['NSI_OccupancyClass']=='GOV2-FIRE'].copy().to_crs(crs_plot).iterrows():
            folium.CircleMarker(location=[row.geometry.y, row.geometry.x], 
                                radius=3, 
                                color='red', 
                                fill=True, 
                                fill_color='red').add_to(m)
        
        for idx, row in new_gov2[new_gov2['NSI_OccupancyClass']=='GOV2-POLICE'].copy().to_crs(crs_plot).iterrows():
            folium.CircleMarker(location=[row.geometry.y, row.geometry.x], 
                                radius=3, 
                                color='blue', 
                                fill=True, 
                                fill_color='blue').add_to(m)
            
        for idx, row in new_gov2[new_gov2['NSI_OccupancyClass']=='GOV2-OPERATIONS'].copy().to_crs(crs_plot).iterrows():
            folium.CircleMarker(location=[row.geometry.y, row.geometry.x], 
                                radius=3, 
                                color='green', 
                                fill=True, 
                                fill_color='green').add_to(m)
        
        for idx, row in nsi_gov2_near_hifld.copy().to_crs(crs_plot).iterrows():
            folium.CircleMarker(location=[row.geometry.y, row.geometry.x], 
                                radius=7, 
                                color='black', 
                                fill=True, 
                                fill_color='black').add_to(m)
            
    else:
        m = 'No Plot Requested'

    # Return
    return nsi, m
##########################



##########################
def add_nsi_tracking_columns(nsi, filter_under_450):
    """
    This function creates new columns to be used to track information in footprint merge process. It also changes data types for missing information within certain rows 
    """

    # Create Tracking Column in for Footprint Merge
    nsi['NSI_ID'] = range(len(nsi)) # This is an ID number that is used throughout the script to refer to each row
    nsi['NSI_FootprintID'] = pd.Series([pd.NA] * len(nsi), dtype='Int64') #None # This is the FootprintID that will be paired witht the point data throughout
    nsi['DistanceToFtpt'] = None 
    nsi['ClosestFtpt_ID'] = None
    nsi['NSI_ID_List'] = nsi['NSI_ID'] # This tracks the ID numbers associated with that row 
    nsi['NSI_NumPoints'] = 1 # This tracks the number of points consolidated into the single row 
    nsi['NSI_MergeFlag'] = 0 # This tracks at what stage the point and footprint are merged

    # Rename missing data with appropriate information
    nsi['NSI_FoundationType'] = nsi['NSI_FoundationType'].replace({None: 'Missing'})
    nsi['NSI_FoundationHeight'] = nsi['NSI_FoundationHeight'].replace({np.nan: 'Missing'})
    nsi['NSI_BuildingType'] = nsi['NSI_BuildingType'].replace({None: 'Missing'})
    nsi['NSI_MedYearBuilt'] = nsi['NSI_MedYearBuilt'].replace({np.nan: 'Missing'})
    nsi['NSI_ContentValue'] = nsi['NSI_ContentValue'].replace({np.nan: 0})
    nsi['NSI_StructureValue'] = nsi['NSI_StructureValue'].replace({np.nan: 0})
    nsi['NSI_ReplacementCost'] = nsi['NSI_ReplacementCost'].replace({np.nan: 0})
    nsi['NSI_NumberOfStories'] = nsi['NSI_NumberOfStories'].replace({np.nan: 'Missing'})
    nsi['NSI_TotalAreaSqFt'] = nsi['NSI_TotalAreaSqFt'].replace({np.nan: 0})
    nsi['NSI_OrigSource'] = nsi['NSI_OrigSource'].replace({np.nan: 'Missing'})

    nsi['NSI_OrigSource'] = nsi['NSI_OrigSource'].replace({np.nan: 'Missing'})
    nsi['NSI_OrigFtptSource'] = nsi['NSI_OrigFtptSource'].replace({np.nan: 'Missing'})
    nsi['NSI_BID'] = nsi['NSI_BID'].replace({np.nan: 'Missing'})

    nsi['NSI_DropNote'] = nsi['NSI_DropNote'].replace({None : ""})
    nsi['NSI_DataUpdate'] = nsi['NSI_DataUpdate'].replace({None : ""})
    nsi['NSI_OC_Update'] = nsi['NSI_OccupancyClass'] 
    nsi = nsi.drop(columns=['NSI_fdid'])

    # Filter out NSI points smaller than 450 square feet 
    if filter_under_450:
        nsi = nsi[nsi['NSI_TotalAreaSqFt']>=450]

    return nsi 
##########################


########################## UPDATED JUL 1
def compute_min_mix_units(nsi):
    """
    Compute minimum and maximum number of units based on defined ranges for NSI typologies 
    """
    
    mapping_scheme_min = {'RES1': 1, 'RES3A': 2, 'RES3B': 3, 'RES3C': 5, 'RES3D': 10, 'RES3E': 20, 'RES3F': 51}
    mapping_scheme_max = {'RES1': 1, 'RES3A': 2, 'RES3B': 4, 'RES3C': 9, 'RES3D': 19, 'RES3E': 50, 'RES3F': 51}

    # Adjust occupancy class for 'RES1' case
    original_occupancy = nsi['NSI_OccupancyClass'].copy()
    nsi['NSI_OccupancyClass'] = nsi['NSI_OccupancyClass'].str.replace(r'.*RES1.*', 'RES1', regex=True)

    # Apply the mapping
    nsi['NSI_MinResUnits'] = nsi['NSI_OccupancyClass'].map(mapping_scheme_min).fillna(0)
    nsi['NSI_MaxResUnits'] = nsi['NSI_OccupancyClass'].map(mapping_scheme_max).fillna(0)

    # Reset RES1 types 
    nsi['NSI_OccupancyClass'] = original_occupancy

    # Return 
    return nsi 
##########################


########################### UPDATED JUL 1
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
def merge_occ_bid(group):
    """
    Determines and merges the occupancy type for a set of footprints assigned to the same group (either same FootprintID
    or same NSI_BID). Updates are made to combine RES1 and RES3 into a single residential entry based on mean number of units, 
    as specified by map_to_units().

    Outputs:
    - group_occ: The final occupancy class or classes for the group of points.
    
    """

     # Find number of unique occupancy types 
    occ_class_list = [item for sublist in group['NSI_OC_Update'] for item in (sublist if isinstance(sublist, list) else [sublist])]
    occ_class_series = pd.Series(occ_class_list)

    # print(occ_class_list)


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
    
    else:
        group_occ = occ_class_list
    
    return group_occ
##########################


########################### UPDATED JUL 1
def merge_duplicate_bid(nsi, list_columns, sum_columns):
    """
    Merges a group of points with the same BID a single record, updating occupancy type, residential units, 
    and other attributes. Marks points that have been absorbed into other records as such. 
    """
    all_new_rows = []
    all_ids_absorbed = set()

    # Group rows based on NSI_BID
    groups = nsi.groupby('NSI_BID')

    # Define expected length 
    expected_length = len(nsi)

    # Loop through grouped BID rows 
    for bid, group in groups: 

        if bid == 'Missing':
            continue

        else: 
            if len(group) > 1: 

                # Merge occupancy type 
                group_occ = merge_occ_bid(group)

                # Obtian first row for some simplified calculations
                data = group.iloc[0].copy()

                # For other points in group that are not the first row, record those has having been merged into another row 
                ids_absorbed = list(group.iloc[1:]['NSI_ID'].values)

                # Set occupancy information  
                data['NSI_OC_Update'] = group_occ

                # List columns
                for col in list_columns:
                    values = group[col].unique()
                    data[col] = values[0] if len(values) == 1 else list(group[col])

                # Summable columns
                for col in sum_columns:
                    data[col] = float(group[col].sum())

                # Collect all unique data updates associated with group, then append the new update tag 
                unique_data_updates = list({item for sublist in group['NSI_DataUpdate'] for item in (sublist if isinstance(sublist, list) else [sublist]) if item != ''})
                unique_data_updates.append('Absorbed data from NSI point(s) with same NSI_BID')
                data['NSI_DataUpdate'] = unique_data_updates

                # Format row
                data = data.to_frame().T

                # Rename census columns
                data = data.rename(columns={'CensusBlock_left': 'CensusBlock', 'CensusTract_left': 'CensusTract'})

                # Convert to GeoDataFrame
                data_gdf = gpd.GeoDataFrame(data, geometry='geometry')
                data_gdf.crs = group.crs

                # Set notes for updated row 
                new_row = data_gdf.copy()
                
                # Save information for future manipulation 
                all_new_rows.append(new_row)
                all_ids_absorbed.update(ids_absorbed)

    # Mark rows that have been absorbed
    nsi = drop_ids(nsi, all_ids_absorbed, 'Data merged with another NSI point within same NSI_BID')

    # Update appropriate rows 
    nsi = update_new_rows(nsi, all_new_rows, expected_length)
    
    return nsi 
##########################


########################### UPDATED JUL 1
def update_new_rows(nsi, all_new_rows, expected_length):
    """
    Update nsi data with row that has abosorbed other points and been updated accordingly. Drop previous row, and check that no data has been lost using expected_length
    """
    
    # Update NSI dataframe to reflect merged rows 
    new_rows_df = pd.concat(all_new_rows, ignore_index=True)

    # Remove duplicates based on NSI_ID and keep the one with the maximum NSI_NumPoints
    new_rows_df = new_rows_df.loc[new_rows_df.groupby('NSI_ID')['NSI_NumPoints'].idxmax()].reset_index(drop=True)
    
    # Drop and concat points
    nsi_with_rows_dropped = nsi[~nsi['NSI_ID'].isin(set(new_rows_df['NSI_ID']))]
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
    This function finds the unpaired points and footprints in NSI data and foorptinr data, based on NSI_MergeFlag and NSI_FootprintID
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
def merge_intersecting(points, footprints, crs_plot):
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
        duplicate = unique_points[unique_points.duplicated(subset='NSI_ID', keep = False)]
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
        return points, m 

    else: 
        m = "Passed Check: No overlapping footprints found"

    # Map FootprintID, Merge, and other columns directly
    points = points.merge(unique_points[['NSI_ID', 'FootprintID']], on='NSI_ID', how='left')
    points['NSI_FootprintID'] = points['FootprintID']
    points['ClosestFtpt_ID'] = points['FootprintID']
    points['NSI_MergeFlag'] = points['NSI_FootprintID'].map(lambda x: 0 if pd.isna(x) else 1)
    points['DistanceToFtpt'] = points['NSI_MergeFlag'].replace({1: 0, 0: None})

    # Drop footprint column 
    points = points.drop(columns=['FootprintID'])

    # Display and return 
    print('Data with Associated Footprints (should match row above):',points['NSI_FootprintID'].notna().sum())
    return points, m 
##########################



##########################
def check_post_merge_duplicates(nsi):
    """
    This function checks if any points are designated as both being merged into the same footprint. This should not occur given the current methodology. 
    """
    # Check for duplicate footprints in points that have already been merged
    nsi_paired = nsi[nsi['NSI_MergeFlag']!=0].copy()
    nsi_paired['NSI_FootprintID'] = nsi_paired['NSI_FootprintID'].astype(int)
    duplicates = nsi_paired[nsi_paired.duplicated(subset='NSI_FootprintID', keep=False)]
    if len(duplicates) == 0: 
        print('Passed Check: No duplicates found')
    else: 
        raise ValueError('Duplicates found: Please check process')
##########################


##########################
def recombine_dropped_data(nsi0, nsi1, nsi_length):
    """
    This function recombines updated nsi0 data and the dropped nsi1 data 
    """
    dtype_reference = nsi0 if not nsi0.empty else nsi1
    nsi1_aligned = nsi1.reindex(columns=dtype_reference.columns).astype(dtype_reference.dtypes.to_dict(), errors="ignore")

    nsi = pd.concat([nsi0, nsi1_aligned], ignore_index=True)
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
    flag1 = nsi[nsi['NSI_MergeFlag']==1].copy()

    # Use only entires that were NOT updated by absorbing other points with NSI_BID
    flag1['NSI_DataUpdate_STR'] = flag1['NSI_DataUpdate'].apply(lambda x: ', '.join(x) if isinstance(x, list) else x)
    flag1 = flag1[~flag1['NSI_DataUpdate_STR'].str.contains('same NSI_BID')]

    # Create dictionary of size limit
    for occ_type in flag1['NSI_OccupancyClass'].unique():

        occ_assigned = flag1[flag1['NSI_OccupancyClass'] == occ_type]
        occ_footprints = footprints[footprints['FootprintID'].isin(occ_assigned['NSI_FootprintID'])]
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
    then compares footprint size against the limit for a given occupancy type. If it is larger than expected, the NSI_MergeFlag is set to 99. 
    If there is a list of occupancy class values (due to merged points), the limit is set to whatever occupancy class limit is largest.

    Outputs: Entire updated nsi GeoDataFrame with rows marked as NSI_MergeFlag - 99
    """

    size_limit_dict = create_size_limit_dict(nsi_fxn, footprints)

    # Convert data to list values for ease of use 
    nsi_fxn['OC_List'] = nsi_fxn['NSI_OC_Update'].apply(convert_to_list)

    # Find size corresponding to each occupancy class in list 
    nsi_fxn['size_vals'] = nsi_fxn['OC_List'].map(lambda lst: [size_limit_dict.get(item, np.nan) for item in lst] if isinstance(lst, list) else np.nan)

    # Find max size, the largest for the list of occupancy classes in list 
    nsi_fxn['max_size'] = nsi_fxn['size_vals'].map(lambda lst: max(lst) if isinstance(lst, list) and len(lst) > 0 else np.nan)

    #  Join footprint sizes directly into nsi
    nsi_fxn = nsi_fxn.merge(footprints[['FootprintID', 'Total_SqFt']],how='left',left_on='NSI_FootprintID',right_on='FootprintID')

    # Compare and set flag 
    condition = (nsi_fxn['NSI_MergeFlag'] == mergeflag) & (nsi_fxn['Total_SqFt'] > nsi_fxn['max_size'] )
    nsi_fxn['NSI_MergeFlag'] = nsi_fxn['NSI_MergeFlag'].where(~condition, 99)

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
    drop_idx = nsi[nsi['NSI_ID'].isin(ids_to_drop)].index
    nsi.loc[drop_idx, 'NSI_DropFlag'] = 1
    nsi.loc[drop_idx, 'NSI_DropNote'] = note
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
    if int(group.iloc[0]['NSI_FootprintID']) in list(manually_assigned_occupancy['FootprintID']): 
        group_occ = manually_assigned_occupancy[manually_assigned_occupancy['FootprintID'] == group.iloc[0]['NSI_FootprintID']].iloc[0]['NSI_OccupancyClass']
    
    # Not in manually assigned occupancy 
    else:

        #### CODE TO PRINT OUT ODD OCCUPANCY PAIRINGS AND THEIR COORDINATES ####
        
        if print_odd_occupancy_pairings == True:

            ## EDUCATIONAL AND INDUSTRIAL POINTS 
            if (occ_class_series.str.contains('EDU').any()) and (occ_class_series.str.contains('IND').any()):
                df_to_print = group.copy().to_crs(crs_plot)[['NSI_FootprintID','NSI_ID','NSI_OC_Update','NSI_Population_Day', 'NSI_Population_Night','geometry']]
                df_to_print['coords'] = df_to_print['geometry'].apply(lambda point: (point.y, point.x))
                df_to_print = df_to_print.drop(columns = ['geometry'])
                print('WARNING: UNEXPECTED OCCUPANCY COMBINATION - Check Occupancy Type Manually and Assign or Drop (if no action taken, will be kept as mixed use)')
                display(df_to_print)
            
            # RESIDENTIAL AND INDUSTRIAL POINTS (IND6 EXCLUDED -- COMMONLY HAS SAME BID AS  RES1)
            if (occ_class_series.str.contains('RES1|RES2|RES3').any()) and (occ_class_series.str.contains('IND1|IND2|IND3|IND4|IND5').any()):
                df_to_print = group.copy().to_crs(crs_plot)[['NSI_FootprintID','NSI_ID','NSI_OC_Update','NSI_Population_Day', 'NSI_Population_Night','geometry']]
                df_to_print['coords'] = df_to_print['geometry'].apply(lambda point: (point.y, point.x))
                df_to_print = df_to_print.drop(columns = ['geometry'])
                print('WARNING: UNEXPECTED OCCUPANCY COMBINATION - Check Occupancy Type Manually and Assign or Drop (if no action taken, will be kept as mixed use)')
                display(df_to_print)

            # RESIDENTIAL AND GOVERNMENT POINTS 
            if (occ_class_series.str.contains('RES1|RES2|RES3').any()) and (occ_class_series.str.contains('GOV1').any()):
                df_to_print = group.copy().to_crs(crs_plot)[['NSI_FootprintID','NSI_ID','NSI_OC_Update','NSI_Population_Day', 'NSI_Population_Night','geometry']]
                df_to_print['coords'] = df_to_print['geometry'].apply(lambda point: (point.y, point.x))
                df_to_print = df_to_print.drop(columns = ['geometry'])
                print('WARNING: UNEXPECTED OCCUPANCY COMBINATION - Check Occupancy Type Manually and Assign or Drop (if no action taken, will be kept as mixed use)')
                display(df_to_print)



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
    - ids_absorbed: List of NSI_IDs to be removed after merging because points were absorbed into another row 

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
    ids_absorbed = [item for sublist in additional_rows['NSI_ID'] for item in (sublist if isinstance(sublist, list) else [sublist])]

    if use_nsi_occupancy_merge:
        # Merge occupancy type 
        group_occ = merge_occ_type(group, manually_assigned_occupancy, print_odd_occupancy_pairings, crs_plot)

        # Set occupancy information  
        data['NSI_OC_Update'] = group_occ

    size_flag = 0
    if use_size_limit:
        # Check how footprint size compares to size limit for specified occupancy to set MergeFlag = 99 flag 
        footprint_id = int(data['NSI_FootprintID'])
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
    unique_data_updates = list({item for sublist in group['NSI_DataUpdate'] for item in (sublist if isinstance(sublist, list) else [sublist]) if item != ''})
    unique_data_updates.append('Absorbed data from NSI point(s) within same footprint')
    data['NSI_DataUpdate'] = unique_data_updates

    # Get in format appropriate for merge with NSI 
    data = data.to_frame().T

    # Set Merge Flag 
    if size_flag == 1: 
        data['NSI_MergeFlag'] = 99
    else: 
        data['NSI_MergeFlag'] = merge_flag

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
    remaining_points, remaining_ftpt = find_remaining(nsi_fxn, footprints,'NSI_FootprintID','NSI_MergeFlag')
    remaining_points_with_footprints = gpd.sjoin(remaining_points, remaining_ftpt[['geometry','FootprintID']], how="inner", predicate='within')

    # Find holes enclosed by building footprints 
    holes_gdf = extract_holes(remaining_ftpt)

    if len(holes_gdf) > 0: 
        # If holes present, find associated points
        points_in_holes = gpd.sjoin(remaining_points, holes_gdf[['geometry','FootprintID']], how="inner", predicate="within")

        # Combine points within footprints and points in holes enclosed by footprints
        remaining_points_with_footprints = pd.concat([remaining_points_with_footprints, points_in_holes], ignore_index=True)

    # Assign NSI_FootprintID
    remaining_points_with_footprints['NSI_FootprintID'] = remaining_points_with_footprints['FootprintID']
    remaining_points_with_footprints = remaining_points_with_footprints.drop(columns = ['index_right','FootprintID'])

    # Get groups of points assigned to same footprint 
    nonunique_groups = remaining_points_with_footprints.groupby('NSI_FootprintID')
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
    mask = gdf['NSI_ID'].apply(lambda x: isinstance(x, int) and x == nsi_id_to_match)
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

    # Ensure all NSI_IDs are integers for lookup process
    nsi_fxn['NSI_ID'] = nsi_fxn['NSI_ID'].astype(int) 

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
            index_of_match = find_index(nsi_fxn, int(closest_point['NSI_ID'].iloc[0]))
            nsi_fxn.loc[index_of_match, ['NSI_FootprintID']] = closest_point['ClosestFtpt_ID'].iloc[0]
            nsi_fxn.loc[index_of_match, ['DistanceToFtpt']] = closest_point['DistanceToFtpt'].iloc[0]

            # Reset NSI_MergeFlag if there is still space in the building footprint 
            if size_limit_flag == 1:
                    nsi_fxn.loc[index_of_match, ['NSI_MergeFlag']] = 99

            # If size flag was not raised (paired and no space in building footprint)
            elif size_limit_flag == 0: 
                nsi_fxn.loc[index_of_match, ['NSI_MergeFlag']] = merge_flag
    

            # Drop closest points from current_points and remaining points 
            current_points = current_points[current_points['NSI_ID'] != closest_point['NSI_ID'].iloc[0]]
            unpaired_remaining_points = unpaired_remaining_points[unpaired_remaining_points['NSI_ID'] != closest_point['NSI_ID'].iloc[0]]

            # Drop associated footprint from current_polygons and remaining polygons
            current_polygons = current_polygons[current_polygons['FootprintID'] != closest_point['ClosestFtpt_ID'].iloc[0]]
            unpaired_remaining_ftpt = unpaired_remaining_ftpt[unpaired_remaining_ftpt['FootprintID'] != closest_point['ClosestFtpt_ID'].iloc[0]]
 
        # If the closest points is more than distance_limit away from the nearest building footprint, end the first merge loop (high confidence)
        else: 
            conitnue_flag = False 
            
    return nsi_fxn, unpaired_remaining_points, unpaired_remaining_ftpt, current_points, current_polygons
##########################



##########################
def pair_partial_ftpt_distance(nsi_fxn, footprints, footprints_indexed, unpaired_remaining_points, current_points, distance_limit, assigned_ids, geoid, manually_assigned_occupancy, size_limit_dict, use_size_limit, use_nsi_occupancy_merge, print_odd_occupancy_pairings, adjacent_blocks, CB_ID, merge_flag, list_columns, sum_columns, surrounding_blocks, crs_plot): # CHANGED GEOID MTL 
    """
    This function works similarly to pair_empty_ftpt_distance. However, in this case, the function is evaluating all footprints that have been designated as "not full," by having 
    their NSI_MergeFlag set as 99.
    """

    # While loop pairing with empty footprints is completed. Now check footprints that still have space remaining 
    # Find polygons that are larger than expected, given occupancy type 
    not_full_nsi = nsi_fxn[nsi_fxn['NSI_MergeFlag']==99]
    not_full_ftpt = footprints[footprints['FootprintID'].isin(not_full_nsi['NSI_FootprintID'])]
    umpaired_not_full_ftpt = not_full_ftpt.copy()

    # Set polygons based on surrounding_blocks
    if surrounding_blocks:
        not_full_polygon_adjacent = umpaired_not_full_ftpt[umpaired_not_full_ftpt['CensusBlock'].isin(adjacent_blocks[str(geoid) + '_left'].unique()) | (umpaired_not_full_ftpt['CensusBlock'] == CB_ID)]
    else:
        not_full_polygon_adjacent = umpaired_not_full_ftpt[umpaired_not_full_ftpt['CensusBlock'] == CB_ID]
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
                curr_nsi = nsi_fxn[nsi_fxn['NSI_FootprintID'] == footprint_id].copy()
                curr_nsi['ClosestFtpt_ID'] = curr_nsi['NSI_FootprintID']

                # Label NSI_FootprintID
                group['NSI_FootprintID'] = group['ClosestFtpt_ID'] 

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
                group_ids = list(group['NSI_ID'].values)
                current_points = current_points[~current_points['NSI_ID'].isin(group_ids)]
                unpaired_remaining_points = unpaired_remaining_points[~unpaired_remaining_points['NSI_ID'].isin(group_ids)]

                
                # Drop footprint from current 
                current_polygons_not_full = current_polygons_not_full[current_polygons_not_full['FootprintID'] != int(new_row.iloc[0]['NSI_FootprintID'])]
        
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
def pair_any_ftpt_distance(nsi_fxn, footprints, footprints_indexed, unpaired_remaining_points, current_points, distance_limit, assigned_ids, geoid, manually_assigned_occupancy, size_limit_dict, use_size_limit, use_nsi_occupancy_merge, print_odd_occupancy_pairings, adjacent_blocks, CB_ID, merge_flag, list_columns, sum_columns, surrounding_blocks, crs_plot):
    """
    This function works similarly to pair_empty_ftpt_distance. However, in this case, the function is evaluating all footprints, including empty footprints, footprints with NSI_MergeFlag = 99, 
     and footprints that have been paired.
    """
    # Find all footprints within Census block and adjacent Census block 
    # Set polygons based on surrounding_blocks
    if surrounding_blocks:
        all_ftpt_cb = footprints[footprints['CensusBlock'].isin(adjacent_blocks[str(geoid) + '_left'].unique()) | (footprints['CensusBlock'] == CB_ID)]
    else:
        all_ftpt_cb = footprints[footprints['CensusBlock'] == CB_ID]
    
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
            curr_nsi = nsi_fxn[nsi_fxn['NSI_FootprintID'] == footprint_id].copy()
            curr_nsi['ClosestFtpt_ID'] = curr_nsi['NSI_FootprintID']

           # Label NSI_FootprintID
            group['NSI_FootprintID'] = group['ClosestFtpt_ID'] 

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
    nsi_fxn = drop_ids(nsi_fxn, all_ids_absorbed, 'Data merged with another NSI point within same footprint')

    # Update NSI data to reflect merged rows
    if all_new_rows:
        nsi_fxn = update_new_rows(nsi_fxn, all_new_rows, expected_length)
    
    # Return information
    return nsi_fxn, unpaired_remaining_points, current_points
##########################


##########################

def distance_limit_merge(CB_list, nsi0, footprints, geoid, manually_assigned_occupancy, list_columns, sum_columns, city_blocks, crs_plot, distance_limit, use_surrounding_blocks, use_partial_footprints, use_full_footprints, merge_flag, use_size_limit, use_nsi_occupancy_merge, print_odd_occupancy_pairings): ## MTL ADDED TWO FLAGS 
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
    remaining_points, remaining_ftpt = find_remaining(nsi0, footprints, 'NSI_FootprintID','NSI_MergeFlag')
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
    progress_checkpoints = {round(len(CB_list) * i / 10) for i in range(1, 11)} 
    print(f'Processing {len(CB_list)} Census Blocks')

    # Loop through Census Blocks 
    for i in range(len(CB_list)):
        
        # Set Census Block 
        CB_ID = CB_list[i]

        # Find CBs that are adjacent to current CB 
        current_block = city_blocks[city_blocks[geoid] == CB_ID]
        adjacent_blocks = gpd.sjoin(city_blocks, current_block, predicate='touches')

        # Filter the polygons and points that are in the given CB
        points_CB = unpaired_remaining_points[unpaired_remaining_points['CensusBlock'] == CB_ID]
        if use_surrounding_blocks:
            polygon_adjacent = unpaired_remaining_ftpt[unpaired_remaining_ftpt['CensusBlock'].isin(adjacent_blocks[str(str(geoid)+'_left')].unique()) | (unpaired_remaining_ftpt['CensusBlock'] == CB_ID)]
        else:
            polygon_adjacent = unpaired_remaining_ftpt[unpaired_remaining_ftpt['CensusBlock'] == CB_ID]

        # Filter for cases where there is at least one point within the CB
        if len(points_CB):

            # Initialize current points and polygons
            current_points = points_CB.copy()
            current_polygons = polygon_adjacent.copy()

            # Merge points with any unpaired footprints within the distance limit  
            nsi0, unpaired_remaining_points, unpaired_remaining_ftpt, current_points, current_polygons = pair_empty_ftpt_distance(nsi0.copy(), footprints_indexed, unpaired_remaining_points, unpaired_remaining_ftpt, current_points, current_polygons, distance_limit, assigned_ids, use_size_limit, size_limit_dict, merge_flag)

            # Merge points with footprints that are larger than their associated occupancy class within the distance limit  
            if len(current_points) and use_partial_footprints:
                nsi0, unpaired_remaining_points, current_points = pair_partial_ftpt_distance(nsi0.copy(), footprints, footprints_indexed, unpaired_remaining_points, current_points, distance_limit, assigned_ids, geoid, manually_assigned_occupancy, size_limit_dict, use_size_limit, use_nsi_occupancy_merge, print_odd_occupancy_pairings,adjacent_blocks, CB_ID, merge_flag, list_columns, sum_columns, use_surrounding_blocks, crs_plot)

            # Merge points with any footprint within distance limit 
            if len(current_points) and use_full_footprints:
                nsi0, unpaired_remaining_points, current_points = pair_any_ftpt_distance(nsi0.copy(), footprints, footprints_indexed, unpaired_remaining_points, current_points, distance_limit, assigned_ids, geoid, manually_assigned_occupancy, size_limit_dict, use_size_limit, use_nsi_occupancy_merge, print_odd_occupancy_pairings, adjacent_blocks, CB_ID, merge_flag, list_columns, sum_columns, use_surrounding_blocks, crs_plot)

        # Print progress checkpoints 
        if counter in progress_checkpoints:
            percent = round((counter / len(CB_list)) * 100)
            print(f"{percent}% complete")
        counter += 1
    print("100% complete")
        
    return nsi0

##########################








#########################
def check_occupancy_class(value, res_types):
    """
    This function is to extract all rows containing residential data from the footprint NSI inventory. 
    """
    if isinstance(value, list):
        # If it's a list, check if any of the options are in the list
        return any(option in value for option in res_types)
    elif isinstance(value, str):
        # If it's a string, check if it's in the options list
        return value in res_types
    return False
#########################



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



# #########################
# def modify_to_single_nsi_occupancy(value):

#     """
#     This function takes a list of values of NSI occupancy classes associated with a single building footprint, and returns two values. 

#     First, it returns the single value associated with the occupancy in a deterministic version of the inventory (output). This is determined by using the mode
#     of the values, and may be designated as mixed use if there is a combination of RES1/RES2/RES3 and non-residential occupancy classes.

#     Second, it returns the mixed_output, which is returned as an empty string, unless output is mixed use, in which case it represents the non-residential occupancy class
#     casuing the mixed use designation.
#     """

#     # Preset mixed output as empty 
#     mixed_output = ''

#     # If it's a single string, just return it
#     if isinstance(value, str):
#         if 'RES1' in value:
#             output = 'RES1'
#         else:
#             output = value
    
#     # If it's an empty string, retrun nan
#     elif isinstance(value, list) and len(value) == 0:
#         output = np.nan

#     # If it's a single string, in list format, return string
#     elif isinstance(value, list) and len(value) == 1:
#         if 'RES1' in value[0]:
#             output = 'RES1'
#         else:
#             output = value[0]

#     # If it is a list of strings, use logic to decide 
#     elif isinstance(value, list):

#         # If occupancy only contains GOV2 and EDU2, it is likely a campus police station - specify GOV2 
#         if all("EDU2" in entry or "GOV2" in entry for entry in value):
#             gov = [item for item in value if 'GOV2']
#             output = np.random.choice(pd.Series(gov).mode())
        
#         # If occupancy contains GOV1 and GOV2 - specify GOV2 
#         if all("GOV1" in entry or "GOV2" in entry for entry in value):
#             gov = [item for item in value if 'GOV2']
#             output = np.random.choice(pd.Series(gov).mode())

#         # Separate resedential and non-residential
#         res = [item for item in value if 'RES1' in item or 'RES2' in item or 'RES3' in item]
#         nonres = [item for item in value if item not in res]

#         # If there is no residential data, select mode of non-residential and return 
#         if len(res) == 0:
#             output = np.random.choice(pd.Series(nonres).mode())
        
#         # Address cases that have both residential and non-residential data
#         elif len(res) != 0 and len(nonres) != 0:

#             # Get randomly selected mode of res 
#             res_mode = np.random.choice(pd.Series(res).mode())
#             if 'RES1' in res_mode:
#                 res_mode = 'RES1'
#             # Get randomly selected mode of nonres
#             nonres_mode = np.random.choice(pd.Series(nonres).mode())

#             # Set occupancy class to mixed use of whatever NSI value is, and record nonres_mode
#             output = res_mode + 'M'
#             mixed_output = nonres_mode

#         # If there is only residential data, select mode of residential and return 
#         else:
#             output = np.random.choice(pd.Series(res).mode())
    
#     else:
#         # Handle any unexpected data types as NaN
#         output = np.nan

#     return (output, mixed_output)
# #########################


######################### JUL 2
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
    cbs20.to_csv('Input_Data/Census/2020_Census_Units.csv')

    return cbs20
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