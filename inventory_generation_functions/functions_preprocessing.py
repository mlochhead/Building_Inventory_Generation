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
from shapely.geometry import Polygon
import requests
import zipfile
import io
from collections import defaultdict





##########################
def fill_census_place(place_file):
    """
    This function fills any holes in a Census place file by selecting only the first polygon present in the geometry.

    Inputs:
    - place_file: A GeoDataFrame representing the boundary of a Place (e.g., city) that may contain multiple polygons or holes.
    
    1. It extracts the coordinates of the first polygon (representing the exterior city boundary) from the geometry column.
    2. A new geometry is created from the extracted coordinates.
    3. The geometry column is updated with the newly created geometry, ensuring that only the main polygon remains.
    4. The additional columns used for this process are dropped.
    
    The function returns the place file with a cleaned geometry containing only the main boundary polygon.
    """
    # Extract first polygon coordinates (exterior city boundary)
    place_file['first_polygon'] = place_file.geometry.apply(extract_first_set_of_coordinates)

    # Creating new geometries from the first set of coordinates
    place_file['new_geometry'] = place_file['first_polygon'].apply(create_new_geometry)
    place_file = place_file.assign(geometry=place_file['new_geometry']).set_geometry('geometry', crs=place_file.crs)

    # Drop additional columns
    place_file = place_file.drop(columns=['first_polygon', 'new_geometry'])

    # Return 
    return place_file

# Internal function used only in fill_place function to extract the first set of coordinates (outer polygon in case of Hayward)
def extract_first_set_of_coordinates(geometry):
    geo_interface = geometry.__geo_interface__
    if geo_interface['type'] == 'Polygon':
        # Access the first set of coordinates
        first_set = geo_interface['coordinates'][0]
        return first_set
    return None

# Internal function used only in fill_place function to create a new Polygon from the first set of coordinates
def create_new_geometry(coords):
    return Polygon(coords)
##########################



##########################
def find_city_tracts_and_blocks(censustracts_County, censusblocks_County, census_Place, col_name): 
    """
    This function identifies Census tracts and Census blocks within a specified Place boundary.

    Inputs:
    - censustracts_County: A GeoDataFrame representing Census tracts for a specified county.
    - censusblocks_County: A GeoDataFrame representing Census blocks for a specified county.
    - census_Place: A GeoDataFrame representing the boundary of the specified Place (e.g., city).
    
    1. It performs a spatial join to find Census tracts from the county that intersect with the given Place boundary.
    2. Similarly, it finds Census blocks that intersect with the same Place boundary.
    3. The function filters the blocks to exclude any block where more than 97% of the block's area is outside of the Place boundary.
    
    It returns two GeoDataFrames: one for the Census tracts and one for the Census blocks within the specified Place.
    """
    # Find Census tracts that intersect with Place Boundary 
    censustracts_City = gpd.sjoin(censustracts_County, census_Place[['geometry']], how="inner", predicate='intersects')
    censustracts_City = censustracts_City[censustracts_County.columns]
    censustracts_City.set_index(col_name,inplace=True)

    # Find Census blocks that intersect with Place Boundary 
    censusblocks_City = gpd.sjoin(censusblocks_County, census_Place[['geometry']], how="inner", predicate='intersects')
    censusblocks_City = censusblocks_City[censusblocks_County.columns]
    censusblocks_City.set_index(col_name,inplace=True)

    # Discard blocks where more than 97% of block area is outside of Place Boundary
    censusblocks_City['intersection_area'] = censusblocks_City.geometry.intersection(census_Place.unary_union).area
    censusblocks_City['tract_area'] = censusblocks_City.geometry.area
    censusblocks_City['percent_within_city'] = censusblocks_City['intersection_area'] / censusblocks_City['tract_area'] * 100
    censusblocks_City = censusblocks_City[censusblocks_City['percent_within_city'] >= 3]
    censusblocks_City = censusblocks_City.drop(columns=['intersection_area','tract_area','percent_within_city'])

    return censustracts_City, censusblocks_City
##########################


##########################
def download_2010_census_boundaries(state, state_fips, county, county_fips):
    """
    Download 2010 tract, block, and place data for a specified county and state
    """

    #### DOWNLOAD 2010 CENSUS TRACTS FOR COUNTY
    url = f"https://www2.census.gov/geo/pvs/tiger2010st/{state_fips}_{state}/{state_fips}{county_fips}/tl_2010_{state_fips}{county_fips}_tract10.zip"
    response = requests.get(url, verify=False)
    if response.status_code == 200:
        # Open the zip file in memory
        with zipfile.ZipFile(io.BytesIO(response.content)) as z:
            # Extract to a temporary folder in memory
            z.extractall(f"./Input_Data/Census/Census2010/{state}/{county}/")
    else:
        print("Failed to download 2010 tract file:", response.status_code)
    
    #### DOWNLOAD 2010 CENSUS BLOCKS FOR COUNTY
    url = f"https://www2.census.gov/geo/pvs/tiger2010st/{state_fips}_{state}/{state_fips}{county_fips}/tl_2010_{state_fips}{county_fips}_tabblock10.zip"
    response = requests.get(url, verify=False)
    if response.status_code == 200:
        # Open the zip file in memory
        with zipfile.ZipFile(io.BytesIO(response.content)) as z:
            # Extract to a temporary folder in memory
            z.extractall(f"./Input_Data/Census/Census2010/{state}/{county}/")
    else:
        print("Failed to download 2010 block file:", response.status_code)
    
     #### DOWNLOAD 2010 PLACE FILES FOR STATE OF CALIFORNIA
    url = f"https://www2.census.gov/geo/pvs/tiger2010st/{state_fips}_{state}/{state_fips}/tl_2010_{state_fips}_place10.zip"
    response = requests.get(url, verify=False)
    if response.status_code == 200:
        # Open the zip file in memory
        with zipfile.ZipFile(io.BytesIO(response.content)) as z:
            # Extract to a temporary folder in memory
            z.extractall(f"./Input_Data/Census/Census2010/{state}/")
    else:
        print("Failed to download 2010 place file:", response.status_code)
##########################



##########################
def download_2020_census_boundaries(state, state_fips, county, county_fips):
    """
    Download 2020 tract, block, and place data for a specified county and state
    """

    #### DOWNLOAD 2020 CENSUS TRACTS
    url = f"https://www2.census.gov/geo/tiger/TIGER2020PL/STATE/{state_fips}_{state.upper()}/{state_fips}{county_fips}/tl_2020_{state_fips}{county_fips}_tract20.zip"
    response = requests.get(url, verify=False)
    if response.status_code == 200:
        # Open the zip file in memory
        with zipfile.ZipFile(io.BytesIO(response.content)) as z:
            # Extract to a temporary folder in memory
            z.extractall(f"./Input_Data/Census/Census2020/{state}/{county}/")
    else:
        print("Failed to download 2020 tract file:", response.status_code)

     #### DOWNLOAD 2020 CENSUS BLOCKS
    url = f"https://www2.census.gov/geo/tiger/TIGER2020PL/STATE/{state_fips}_{state.upper()}/{state_fips}{county_fips}/tl_2020_{state_fips}{county_fips}_tabblock20.zip"
    response = requests.get(url, verify=False)
    if response.status_code == 200:
        # Open the zip file in memory
        with zipfile.ZipFile(io.BytesIO(response.content)) as z:
            # Extract to a temporary folder in memory
            z.extractall(f"./Input_Data/Census/Census2020/{state}/{county}/")
    else:
        print("Failed to download 2020 block file:", response.status_code)

    
    # #### DOWNLOAD 2020 PLACE FILES FOR STATE OF CALIFORNIA
    url = f"https://www2.census.gov/geo/tiger/TIGER2020PL/STATE/{state_fips}_{state.upper()}/{state_fips}/tl_2020_{state_fips}_place20.zip"
    response = requests.get(url, verify=False)
    if response.status_code == 200:
        # Open the zip file in memory
        with zipfile.ZipFile(io.BytesIO(response.content)) as z:
            # Extract to a temporary folder in memory
            z.extractall(f"./Input_Data/Census/Census2020/{state}/")
    else:
        print("Failed to download 2020 place file:", response.status_code)
##########################



##########################
def assign_footprint_block_and_track(footprints, city_tracts, city_blocks, col_name):
    """
    Assign Census Block and Tract information to building footprints based on centroid location.

    Inputs:
    - footprints: A GeoDataFrame representing building footprints.
    - city_tracts: A GeoDataFrame representing the Census tracts for the specified city.
    - city_blocks: A GeoDataFrame representing the Census blocks for the specified city.

    Output:
    - footprints_city: A GeoDataFrame containing the building footprints with assigned Census Block 
                          and Census Tract information, limited to those within the blocks of interest.
    """
    # Assign block based on centroid
    footprints_centroids = footprints.copy()
    footprints_centroids.geometry = footprints.geometry.centroid
    footprints_centroids = footprints_centroids.sjoin(city_blocks, how='left')
    footprints.loc[:, 'CensusBlock'] = footprints_centroids[col_name].values

    # Assign tract based  on centorid
    footprints_centroids = footprints.copy()
    footprints_centroids.geometry = footprints.geometry.centroid
    footprints_centroids = footprints_centroids.sjoin(city_tracts, how='left')
    footprints.loc[:, 'CensusTract'] = footprints_centroids[col_name].values

    # Drop rows that are not in blocks of interest
    footprints_city = footprints[~footprints['CensusBlock'].isna()].copy()

    # Return
    return footprints_city
##########################



##########################
def find_overlapping_ftpt(gdf, overlap_limit): 
    """
    This function looks for footprints that have more than 80% overlap and returns a geodataframe containing footprint
    rows that are significantlly overlapping
    """

    # Calculate the area of each footprint
    gdf['footprint_area'] = gdf['geometry'].area

    # Perform a spatial join to find intersections between the footprints and remove self-overlaps (where the same footprint intersects with itself)
    overlaps = gpd.sjoin(gdf, gdf, how='inner', predicate='intersects')
    overlaps = overlaps[overlaps.index != overlaps['index_right']]

    # Calculate the area and percentage of the overlapping footprints
    if not overlaps.empty:
        overlaps['overlap_area'] = overlaps.apply(lambda row: gdf.loc[row.name, 'geometry'].intersection(gdf.loc[row['index_right'], 'geometry']).area, axis=1)
        overlaps['overlap_percentage'] = overlaps['overlap_area'] / gdf.loc[overlaps.index, 'footprint_area'].values
    
    else:
        overlaps['overlap_area'] = pd.Series(dtype=float)
        overlaps['overlap_percentage'] = pd.Series(dtype=float)

    # Filter to get overlaps where the overlap area is high
    if isinstance(overlap_limit, (float)):
        significant_overlaps = overlaps[overlaps['overlap_percentage'] >= overlap_limit]
    if overlap_limit == 'no_overlap':
        significant_overlaps = overlaps.copy()


    # Return gdf with significant overlapping footprints
    return significant_overlaps
##########################


##########################
def find_overlaps_to_drop(footprints, overlap):
    """
    Identifies overlapping footprint geometries and returns the Overlap_IDs of all 
    but the largest geometry in each connected group of overlapping features.
    """

    # Build adjacency graph using OBJECTIDs
    adjacency = defaultdict(set)
    for _, row in overlap.iterrows():
        left_id = row['Overlap_ID_left']
        right_id = row['Overlap_ID_right']
        adjacency[left_id].add(right_id)
        adjacency[right_id].add(left_id)

    # Find connected components (you still use pre.find_components)
    components = find_components(adjacency)

    # For each component, drop all but the largest geometry
    to_drop = set()
    for comp in components:
        comp_set = set(comp)

        # Filter rows where either left or right Overlap_ID is in the component
        sub = overlap[
            (overlap['Overlap_ID_left'].isin(comp_set)) |
            (overlap['Overlap_ID_right'].isin(comp_set))
        ].copy()

        # Gather all unique OBJECTIDs in this component and get their geometries
        all_ids = pd.unique(sub[['Overlap_ID_left', 'Overlap_ID_right']].values.ravel())

        sub_geoms = footprints[footprints['Overlap_ID'].isin(all_ids)]
        areas = sub_geoms.geometry.area
        to_drop_ids = sub_geoms.loc[areas.sort_values(ascending=False).index[1:], 'Overlap_ID']
        to_drop.update(to_drop_ids)

    return to_drop
##########################



##########################
def find_components(adjacency):
    """
    This function finds groups of nodes (e.g., overlapping footprints) that are connected to each other. This includes
    any node in the group can be reached from any other node via edges (adjacency). This is common in overture footprints and can 
    cause problems for merging data. 
    """
    visited = set()
    components = []
    for node in adjacency:
        if node not in visited:
            stack = [node]
            component = set()
            while stack:
                current = stack.pop()
                if current not in visited:
                    visited.add(current)
                    component.add(current)
                    stack.extend(adjacency[current] - visited)
            components.append(component)
    return components
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
def format_and_locate_edu1(public, private, city_tracts, city_blocks, cb_id_name): # MTL CHANGE cb_id_name new input
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
    public_city = public.sjoin(city_blocks[[cb_id_name,'geometry']], how='inner')
    private_city = private.sjoin(city_blocks[[cb_id_name,'geometry']], how='inner')

    # Assign Census Block 
    public_city = public_city.drop(columns = ['index_right'])
    public_city = public_city.rename(columns={cb_id_name:'CensusBlock'})
    private_city = private_city.drop(columns = ['index_right'])
    private_city = private_city.rename(columns={cb_id_name:'CensusBlock'})

    # Assign Census Tract 
    public_city_copy = public_city.copy().sjoin(city_tracts, how='left')
    public_city.loc[:, 'CensusTract'] = public_city_copy[cb_id_name].values
    private_city_copy = private_city.copy().sjoin(city_tracts, how='left')
    private_city.loc[:, 'CensusTract'] = private_city_copy[cb_id_name].values

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
    school_import['POINT_DropFlag'] = 0
    school_import['POINT_Source'] = 'HIFLD'
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
    edu1 = edu1.drop(columns=['CensusBlock','CensusTract','NSI_OccupancyClass','POINT_DropFlag','POINT_Source'])

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
    nsi_edu1_near_hifld['POINT_DataUpdate']='HIFLD_AND_NSI_EDU'
    remaining_school_import = school_import[~school_import.index.isin(nsi_edu1_near_hifld['index_hifld'])].copy()
    remaining_school_import['POINT_DataUpdate']='Only_HIFLD_EDU'
    school_import_w_nsi = pd.concat([remaining_school_import,nsi_edu1_near_hifld])
    if len(school_import_w_nsi) != len(school_import):
        raise ValueError('HIFLD School Data Dropped')


    # Mark EDU1 points to be dropped from NSI (data from close points already associated with HIFLD points)
    edu1_absorbed = nsi[
        (nsi['NSI_fdid'].isin(nsi_edu1_near_hifld['NSI_fdid'].unique())) & # Not paired with HIFLD data
        (nsi['NSI_OccupancyClass'] == 'EDU1')].index # Occupancy class is EDU1
    nsi.loc[edu1_absorbed,'POINT_DropFlag']=1
    nsi.loc[edu1_absorbed,'POINT_DropNote']='NSI EDU1 Points <50m from HIFLD Absorbed by HIFLD'

    if drop_unpaired_nsi_edu1:
        edu1_to_drop = nsi[
            (~nsi['NSI_fdid'].isin(nsi_edu1_near_hifld['NSI_fdid'].unique())) & # Not paired with HIFLD data
            (nsi['NSI_OccupancyClass'] == 'EDU1')].index # Occupancy class is EDU1
        nsi.loc[edu1_to_drop,'POINT_DropFlag']=1
        nsi.loc[edu1_to_drop,'POINT_DropNote']='NSI EDU1 Points >50m from HIFLD Dropped'


    # Merge new HIFLD points into NSI Dataframe 
    nsi = pd.concat([nsi,school_import_w_nsi])
    nsi['POINT_NumPoints'] = 1

    # Flag all GOV1 Points within 50m of a newly imported EDU1 point 
    if drop_gov1_near_edu1:
        gov1_near_edu1 = find_gov1_near_hifld(nsi.copy(), 'EDU1-PUB',50)
        nsi.loc[gov1_near_edu1.index,'POINT_DropFlag']=1
        nsi.loc[gov1_near_edu1.index,'POINT_DropNote']='GOV1 Point within 50m of HIFLD School'

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
def synthesize_edu1_and_HIFLD_nsi26_update(nsi, school_import, drop_unpaired_nsi_edu1, drop_edu1_far_from_hifld, gov1_near_edu1):
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
    edu1 = edu1.drop(columns=['CensusBlock','CensusTract','NSI_OccupancyClass','POINT_DropFlag','POINT_Source'])

    ## Find nearest NSI EDU points to HIFLD EDU points 
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
    nsi_edu1_near_hifld['POINT_DataUpdate']='HIFLD_AND_NSI_EDU'
    remaining_school_import = school_import[~school_import.index.isin(nsi_edu1_near_hifld['index_hifld'])].copy()
    remaining_school_import['POINT_DataUpdate']='Only_HIFLD_EDU'
    school_import_w_nsi = pd.concat([remaining_school_import,nsi_edu1_near_hifld])
    if len(school_import_w_nsi) != len(school_import):
        raise ValueError('HIFLD School Data Dropped')


    # Mark EDU1 points to be dropped from NSI (data from close points already associated with HIFLD points)
    edu1_absorbed = nsi[
        (nsi['NSI_fdid'].isin(nsi_edu1_near_hifld['NSI_fdid'].unique())) & # Not paired with HIFLD data
        (nsi['NSI_OccupancyClass'] == 'EDU1')].index # Occupancy class is EDU1
    nsi.loc[edu1_absorbed,'POINT_DropFlag']=1
    nsi.loc[edu1_absorbed,'POINT_DropNote']='Closest NSI EDU from HIFLD Absorbed by HIFLD'

    # Drop unpaired NSI points if that flag is activated
    if drop_unpaired_nsi_edu1:
        edu1_to_drop = nsi[
            (~nsi['NSI_fdid'].isin(nsi_edu1_near_hifld['NSI_fdid'].unique())) & # Not paired with HIFLD data
            (nsi['NSI_OccupancyClass'] == 'EDU1')].index # Occupancy class is EDU1
        nsi.loc[edu1_to_drop,'POINT_DropFlag']=1
        nsi.loc[edu1_to_drop,'POINT_DropNote']='NSI EDU1 Points not paired with HIFLD Dropped'
    

    # Merge new HIFLD points into NSI Dataframe 
    nsi = pd.concat([nsi,school_import_w_nsi])
    nsi['POINT_NumPoints'] = 1


    # Drop NSI EDU points outside of a radius from HIFLD points (MTL new for 2026)
    if drop_edu1_far_from_hifld:
         
            far_pub = find_edu1_far_hifld(nsi.copy(), 'EDU1-PUB', 150).index
            far_priv = find_edu1_far_hifld(nsi.copy(), 'EDU1-PRIV', 150).index
            far_both = far_pub.intersection(far_priv)

            nsi.loc[far_both, 'POINT_DropFlag'] = 1
            nsi.loc[far_both, 'POINT_DropNote'] = 'EDU1 Point more than 100m from any HIFLD School'


    # Flag all GOV1 Points within 50m of a newly imported EDU1 point 
    if gov1_near_edu1 == 'drop':

        gov1_near_edu1 = find_gov1_near_hifld(nsi.copy(), 'EDU1-PUB',150)
        nsi.loc[gov1_near_edu1.index,'POINT_DropFlag']=1
        nsi.loc[gov1_near_edu1.index,'POINT_DropNote']='GOV1 Point within 50m of HIFLD School'

        gov1_near_edu1 = find_gov1_near_hifld(nsi.copy(), 'EDU1-PUB',150)
        nsi.loc[gov1_near_edu1.index,'POINT_DropFlag']=1
        nsi.loc[gov1_near_edu1.index,'POINT_DropNote']='GOV1 Point within 50m of HIFLD School'

    elif gov1_near_edu1 == 'convert':

        gov1_near_edu1 = find_gov1_near_hifld(nsi.copy(), 'EDU1-PUB',150)
        nsi.loc[gov1_near_edu1.index,'NSI_OccupancyClass']='EDU1'
        nsi.loc[gov1_near_edu1.index,'NSI_OC_Update']='EDU1'
        nsi.loc[gov1_near_edu1.index,'POINT_DataUpdate']='GOV1 converted to EDU1 within 150m of HIFLD School'

        gov1_near_edu1 = find_gov1_near_hifld(nsi.copy(), 'EDU1-PRIV',150)
        nsi.loc[gov1_near_edu1.index,'NSI_OccupancyClass']='EDU1'
        nsi.loc[gov1_near_edu1.index,'NSI_OC_Update']='EDU1'
        nsi.loc[gov1_near_edu1.index,'POINT_DataUpdate']='GOV1 converted to EDU1 within 150m of HIFLD School'
    
    elif gov1_near_edu1 == 'keep':
        pass
    else: 
        raise ValueError('Please select "drop", "convert", or "keep" for gov1_near_edu1 flag')

    # Drop additional columns 
    nsi = nsi.drop(columns=['distance','index_hifld'])

    # Return
    return nsi 
##########################



##########################
def find_edu1_far_hifld(nsi, hifld_occ, buffer):
    """
    Locates EDU1 points outside a specified distance of a specfied occupancy class
    Input:
    - nsi: GeoDataFrame with NSI data.
    Output:
    - nsi: GeoDataFrame with EDU1 far from specified occupancy class (to be dropped from NSI)
    """
    edu1_points = nsi[nsi['NSI_OccupancyClass'] == 'EDU1']
    hifld_points = nsi[nsi['NSI_OccupancyClass'] == hifld_occ]

    # Create a buffer around HIFLD points
    hifld_points_buffer = hifld_points.copy()
    hifld_points_buffer['geometry'] = hifld_points_buffer.geometry.buffer(buffer)

    # Perform a spatial join to find GOV1 points within 100 meters of EDU1-PUB
    edu1_points_far = gpd.sjoin(edu1_points, hifld_points_buffer, how='left', predicate='within')
    edu1_points_far = edu1_points_far[edu1_points_far.index_right.isna()]

    # Return
    return edu1_points_far
##########################



##########################
def find_gov2_far_hifld(nsi, hifld_occ, buffer):
    """
    Locates EDU1 points outside a specified distance of a specfied occupancy class
    Input:
    - nsi: GeoDataFrame with NSI data.
    Output:
    - nsi: GeoDataFrame with EDU1 far from specified occupancy class (to be dropped from NSI)
    """
    gov2_points = nsi[nsi['NSI_OccupancyClass'] == 'GOV2']
    hifld_points = nsi[nsi['NSI_OccupancyClass'] == hifld_occ]

    # Create a buffer around HIFLD points
    hifld_points_buffer = hifld_points.copy()
    hifld_points_buffer['geometry'] = hifld_points_buffer.geometry.buffer(buffer)

    # Perform a spatial join to find GOV1 points within 100 meters of EDU1-PUB
    gov2_points_far = gpd.sjoin(gov2_points, hifld_points_buffer, how='left', predicate='within')
    gov2_points_far = gov2_points_far[gov2_points_far.index_right.isna()]

    # Return
    return gov2_points_far
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
    hifld_points = nsi[nsi['NSI_OccupancyClass'].str.contains(hifld_occ)]

    # Create a buffer around EDU1-PUB points with a radius of 50 meters
    hifld_points_buffer = hifld_points.copy()
    hifld_points_buffer['geometry'] = hifld_points_buffer.geometry.buffer(buffer)

    # Perform a spatial join to find GOV1 points within 100 meters of EDU1-PUB
    gov1_near_edu1_pub = gpd.sjoin(gov1_points, hifld_points_buffer, how='inner', predicate='within')

    # Return
    return gov1_near_edu1_pub
##########################


##########################
def locate_edu2(univ, univ_pts, city_tracts, city_blocks, cb_id_name): # MTL CHANGE cb_id_name new input
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
    univ_city = univ.sjoin(city_blocks[[cb_id_name,'geometry']], how='inner')
    univ_pts_city = univ_pts.sjoin(city_blocks[[cb_id_name,'geometry']], how='inner')

    # Drop duplicate polygons (based on geometry, total enrollement, and source date being duplicated)
    univ_city = univ_city.drop_duplicates(subset=["geometry", "TOT_ENROLL","SOURCEDATE"])

    print('Campus Polygons:',len(univ_city))
    print('College/Univeristy Points:',len(univ_pts_city))

    # Drop and rename columns for Census Block
    univ_city = univ_city.drop(columns = ['index_right'])
    univ_city = univ_city.rename(columns={cb_id_name:'CensusBlock'})
    univ_pts_city = univ_pts_city.drop(columns = ['index_right'])
    univ_pts_city = univ_pts_city.rename(columns={cb_id_name:'CensusBlock'})

    # Assign tract based on Centroid
    univ_city_centroids = univ_city.copy()
    univ_city_centroids.geometry = univ_city.geometry.centroid
    univ_city_centroids = univ_city_centroids.sjoin(city_tracts, how='left')
    univ_city.loc[:, 'CensusTract'] = univ_city_centroids[cb_id_name].values

    # Assign tract for points 
    univ_pts_city_copy = univ_pts_city.copy().sjoin(city_tracts, how='left')
    univ_pts_city.loc[:, 'CensusTract'] = univ_pts_city_copy[cb_id_name].values

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
    edu2_no_poly['NSI_OC_Update'] = 'EDU2'
    edu2_no_poly['POINT_DropFlag'] = 0
    edu2_no_poly['POINT_Source'] = 'HIFLD'

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
    gov1 = nsi[nsi['NSI_OccupancyClass'].isin(['GOV1'])]
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
    edu2_no_poly['NSI_OC_Update'] = 'EDU2'
    edu2_no_poly['POINT_DropFlag'] = 0
    edu2_no_poly['POINT_Source'] = 'HIFLD'

    # Return
    return edu2_no_poly 
##########################





##########################
def prepare_pts_without_gov1_nsi26_update(univ_pts_city, univ_city, nsi):
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
    gov1_edu1 = nsi[nsi['NSI_OccupancyClass'].isin(['EDU1','GOV1'])]
    polygons_with_nsi = gpd.sjoin(univ_city, gov1_edu1, how='inner', predicate='contains')
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
    edu2_no_poly['NSI_OC_Update'] = 'EDU2'
    edu2_no_poly['POINT_DropFlag'] = 0
    edu2_no_poly['POINT_Source'] = 'HIFLD'

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
    nsi.loc[gov1_in_polygons, 'NSI_OC_Update'] = 'EDU2'
    nsi.loc[gov1_in_polygons, 'POINT_DataUpdate'] = 'GOV1 Point within Campus Polyon Convereted to EDU2'
    nsi.loc[gov1_in_polygons, 'POINT_Source'] = 'HIFLD'

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
def merge_pts_with_campuses_nsi26_update(univ_city, nsi, scale_edu2_pop):
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
    gov1_edu1_in_polygons = (nsi['NSI_OccupancyClass'].isin(['GOV1','EDU1'])) & within_polygons

    # Update Occupancy Class and source for those points
    nsi.loc[gov1_edu1_in_polygons, 'NSI_OccupancyClass'] = 'EDU2'
    nsi.loc[gov1_edu1_in_polygons, 'NSI_OC_Update'] = 'EDU2'
    nsi.loc[gov1_edu1_in_polygons, 'POINT_DataUpdate'] = 'GOV1 and EDU1 Point within Campus Polyon Convereted to EDU2'
    nsi.loc[gov1_edu1_in_polygons, 'POINT_Source'] = 'HIFLD'
    nsi.loc[gov1_edu1_in_polygons, 'POINT_DropFlag'] = 0

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
def assign_census_hifld(gdf, city_blocks, city_tracts, cb_id_name): # MTL CHANGE cb_id_name new input
    """
    Assigns Census Block and Tract information to a GeoDataFrame using spatial joins with city blocks and tracts.
    This is used for fire, emergency, and police dataframes, imported from HIFLD. 
    """
    # Spatial join with city boundaries
    gdf = gdf.sjoin(city_blocks[[cb_id_name, 'geometry']], how='inner')
    
    # Drop and rename columns for Census Block
    gdf = gdf.drop(columns=['index_right'])
    gdf = gdf.rename(columns={cb_id_name: 'CensusBlock'})
    
    # Assign Census Tract
    gdf_copy = gdf.copy().sjoin(city_tracts, how='left')
    gdf['CensusTract'] = gdf_copy[cb_id_name].values
    
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
    gov2 = gov2.drop(columns=['NSI_OccupancyClass','POINT_DropFlag','POINT_Source'])

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
    nsi_gov2_near_hifld['POINT_DataUpdate']='HIFLD_AND_NSI_GOV2'
    remaining_gov2_import = new_gov2[~new_gov2.index.isin(nsi_gov2_near_hifld['index_hifld'])].copy()
    remaining_gov2_import['POINT_DataUpdate']='Only_HIFLD_GOV2'
    gov2_import_w_nsi = pd.concat([remaining_gov2_import,nsi_gov2_near_hifld])
    gov2_import_w_nsi['POINT_DropFlag'] = 0
    gov2_import_w_nsi['POINT_Source'] = 'HIFLD'
    if len(gov2_import_w_nsi) != len(new_gov2):
        raise ValueError('HIFLD GOV2 Data Dropped')

    # Mark GOV2 points to be dropped from NSI (data from close points already associated with HIFLD points)
    gov2_absorbed = nsi[
        (nsi['NSI_fdid'].isin(nsi_gov2_near_hifld['NSI_fdid'].unique())) & # Not paired with HIFLD data
        (nsi['NSI_OccupancyClass'] == 'GOV2')].index # Occupancy class is EDU1
    nsi.loc[gov2_absorbed,'POINT_DropFlag']=1
    nsi.loc[gov2_absorbed,'POINT_DropNote']='NSI GOV2 Points <50m from HIFLD Absorbed by HIFLD'

    if drop_unpaired_nsi_gov2: 
        gov2_to_drop = nsi[
            (~nsi['NSI_fdid'].isin(nsi_gov2_near_hifld['NSI_fdid'].unique())) & # Not paired with HIFLD data
            (nsi['NSI_OccupancyClass'] == 'GOV2')].index # Occupancy class is EDU1
        nsi.loc[gov2_to_drop,'POINT_DropFlag']=1
        nsi.loc[gov2_to_drop,'POINT_DropNote']='NSI GOV2 Points >50m from HIFLD or Duplicated Dropped'


    # Merge new HIFLD points into NSI Dataframe 
    nsi = pd.concat([nsi,gov2_import_w_nsi])
    nsi['POINT_NumPoints'] = 1

    # Flag all GOV1 Points within 10m of a newly imported GOV2 point 
    if drop_gov1_near_gov2: 
        gov1_near_gov2 = find_gov1_near_hifld(nsi.copy(), 'GOV2',10)
        nsi.loc[gov1_near_gov2.index,'POINT_DropFlag']=1
        nsi.loc[gov1_near_gov2.index,'POINT_DropNote']='GOV1 Point within 10m of HIFLD GOV2'

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
def synthesize_gov2_and_HIFLD_nsi26_update(nsi, new_gov2, drop_unpaired_nsi_gov2, drop_gov1_near_gov2, gov2_far_from_hifld):
    """
    This function integrates NSI EDU1 points with HIFLD government data. It identifies NSI points with occupancy class 'GOV2' and spatially joins them with HIFLD
    data within a 50-meter radius. Only one NSI point per HIFLD points is paired and vice versa (based on closest point). Matched points are merged into the HIFLD dataset, 
    and unmatched NSI GOV2 points are flagged for removal. It also removes GOV1 points within 10 meters of any imported GOV2 point to avoid duplication.

    If `plot_flag` is True, a map is generated showing original GOV2 points and HIFLD points 
    """
    # Separate out NSI GOV2 points and remove occupancy information (will be replaced with HIFLD)
    gov2 = nsi[nsi['NSI_OccupancyClass']=='GOV2']
    gov2 = gov2.drop(columns=['NSI_OccupancyClass','POINT_DropFlag','POINT_Source'])

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
    nsi_gov2_near_hifld['POINT_DataUpdate']='HIFLD_AND_NSI_GOV2'
    remaining_gov2_import = new_gov2[~new_gov2.index.isin(nsi_gov2_near_hifld['index_hifld'])].copy()
    remaining_gov2_import['POINT_DataUpdate']='Only_HIFLD_GOV2'
    gov2_import_w_nsi = pd.concat([remaining_gov2_import,nsi_gov2_near_hifld])
    gov2_import_w_nsi['POINT_DropFlag'] = 0
    gov2_import_w_nsi['POINT_Source'] = 'HIFLD'
    if len(gov2_import_w_nsi) != len(new_gov2):
        raise ValueError('HIFLD GOV2 Data Dropped')

    # Mark GOV2 points to be dropped from NSI (data from close points already associated with HIFLD points)
    gov2_absorbed = nsi[
        (nsi['NSI_fdid'].isin(nsi_gov2_near_hifld['NSI_fdid'].unique())) & # Not paired with HIFLD data
        (nsi['NSI_OccupancyClass'] == 'GOV2')].index # Occupancy class is EDU1
    nsi.loc[gov2_absorbed,'POINT_DropFlag']=1
    nsi.loc[gov2_absorbed,'POINT_DropNote']='NSI GOV2 Points <50m from HIFLD Absorbed by HIFLD'

    if drop_unpaired_nsi_gov2: 
        gov2_to_drop = nsi[
            (~nsi['NSI_fdid'].isin(nsi_gov2_near_hifld['NSI_fdid'].unique())) & # Not paired with HIFLD data
            (nsi['NSI_OccupancyClass'] == 'GOV2')].index # Occupancy class is EDU1
        nsi.loc[gov2_to_drop,'POINT_DropFlag']=1
        nsi.loc[gov2_to_drop,'POINT_DropNote']='NSI GOV2 Points >50m from HIFLD or Duplicated Dropped'


    # Merge new HIFLD points into NSI Dataframe 
    nsi = pd.concat([nsi,gov2_import_w_nsi])
    nsi['POINT_NumPoints'] = 1

    if gov2_far_from_hifld == 'drop': 

            far_fire = find_gov2_far_hifld(nsi.copy(), 'GOV2-FIRE', 50).index
            far_police = find_gov2_far_hifld(nsi.copy(), 'GOV2-POLICE', 50).index
            far_ops = find_gov2_far_hifld(nsi.copy(), 'GOV2-OPERATIONS', 50).index
            far_all = (far_fire.intersection(far_police).intersection(far_ops))
            nsi.loc[far_all, 'POINT_DropFlag'] = 1
            nsi.loc[far_all, 'POINT_DropNote'] = 'GOV2 Point more than 100m from any HIFLD GOV2'
        
    elif gov2_far_from_hifld == 'convert': 

            far_fire = find_gov2_far_hifld(nsi.copy(), 'GOV2-FIRE', 50).index
            far_police = find_gov2_far_hifld(nsi.copy(), 'GOV2-POLICE', 50).index
            far_ops = find_gov2_far_hifld(nsi.copy(), 'GOV2-OPERATIONS', 50).index
            far_all = (far_fire.intersection(far_police).intersection(far_ops))
            nsi.loc[far_all, 'NSI_OccupancyClass'] = 'GOV1'
            nsi.loc[far_all, 'NSI_OC_Update'] = 'GOV1'
            nsi.loc[far_all, 'POINT_DataUpdate'] = 'GOV2 far from HIFLD converted to GOV1'
        

    elif gov2_far_from_hifld == 'keep': 
             pass
    
    else: 
        raise ValueError('Please select "drop", "convert", or "keep" for gov2_far_from_hifld flag')


    # Flag all GOV1 Points within 10m of a newly imported GOV2 point 
    if drop_gov1_near_gov2: 
        gov1_near_gov2 = find_gov1_near_hifld(nsi.copy(), 'GOV2',10)
        nsi.loc[gov1_near_gov2.index,'POINT_DropFlag']=1
        nsi.loc[gov1_near_gov2.index,'POINT_DropNote']='GOV1 Point within 10m of HIFLD GOV2'


    # Drop additional columns 
    nsi = nsi.drop(columns=['distance','index_hifld'])

    # Return
    return nsi
##########################






##########################
def add_nsi_tracking_columns(nsi, filter_limit):
    """
    This function creates new columns to be used to track information in footprint merge process. It also changes data types for missing information within certain rows 
    """

    # Create Tracking Column in for Footprint Merge
    nsi['POINT_ID'] = range(len(nsi)) # This is an ID number that is used throughout the script to refer to each row
    nsi['POINT_FootprintID'] = pd.Series([pd.NA] * len(nsi), dtype='Int64') #None # This is the FootprintID that will be paired witht the point data throughout
    nsi['DistanceToFtpt'] = None 
    nsi['ClosestFtpt_ID'] = None
    nsi['POINT_ID_List'] = nsi['POINT_ID'] # This tracks the ID numbers associated with that row 
    nsi['POINT_NumPoints'] = 1 # This tracks the number of points consolidated into the single row 
    nsi['POINT_MergeFlag'] = 0 # This tracks at what stage the point and footprint are merged

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
    nsi['NSI_OrigFtptSource'] = nsi['NSI_OrigFtptSource'].replace({np.nan: 'Missing'})
    nsi['NSI_BID'] = nsi['NSI_BID'].replace({np.nan: 'Missing'})
    nsi['NSI_OC_Update'] = nsi['NSI_OccupancyClass'] 
    nsi = nsi.drop(columns=['NSI_fdid'])

    nsi['POINT_DropNote'] = nsi['POINT_DropNote'].replace({None : ""})
    nsi['POINT_DataUpdate'] = nsi['POINT_DataUpdate'].replace({None : ""})

    # Filter out NSI points smaller than 450 square feet 
    nsi = nsi[nsi['NSI_TotalAreaSqFt']>=filter_limit]

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
                ids_absorbed = list(group.iloc[1:]['POINT_ID'].values)

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
                unique_data_updates = list({item for sublist in group['POINT_DataUpdate'] for item in (sublist if isinstance(sublist, list) else [sublist]) if item != ''})
                unique_data_updates.append('Absorbed data from NSI point(s) with same NSI_BID')
                data['POINT_DataUpdate'] = unique_data_updates

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

    # Remove duplicates based on POINT_ID and keep the one with the maximum POINT_NumPoints
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
def drop_ids(nsi, ids_to_drop, note):
    drop_idx = nsi[nsi['POINT_ID'].isin(ids_to_drop)].index
    nsi.loc[drop_idx, 'POINT_DropFlag'] = 1
    nsi.loc[drop_idx, 'POINT_DropNote'] = note
    return nsi
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