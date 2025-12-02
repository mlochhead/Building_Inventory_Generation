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

