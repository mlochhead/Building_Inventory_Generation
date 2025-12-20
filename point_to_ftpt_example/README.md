## DESCRIPTION

This folder contains Python scripts to show a general example of attributing points data to footprint data. Required inputs are a footprint file and a point file which can be loaded as Geopandas GeoDataFrames (original file type is not important). There are no required columns in the point data, other than geometry. The footprint file requires geoemtry and preferable has footprint height in feet.

To the example, first run `Preprocess_Census_and_Footprint.ipynb`, then run `Point_to_Footprint.ipynb`. 
