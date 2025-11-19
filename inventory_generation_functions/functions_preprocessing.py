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