## DESCRIPTION

This folder contains an example of the National Synthesis Workflow, demonstrated for the case study city of Hayward, CA. 

This method synthesizes nationally-available inventory data to create a single final inventory. This workflow is recommended for future studies that aim to improve existing data without downloading, cleaning, and using local data.  To adopt this workflow other locations, input data and minor modifications to the Jupyter notebooks would be required, but the same Python functions can be used to process NSI, HIFLD, Census, and footprint data across the United States.

#### Required Input Data

- Building Footprints: dataset of buildings footprints for the location of interest. Current scripts are set up to accept GeoJSON files, but any file that can be read as a GeoPandas GeoDataframe can be used. Building footprints can be obtained through the [NHERI SimCenter's BRAILS++ tool](https://github.com/NHERI-SimCenter/BrailsPlusPlus) or from an independent source.
- US Census API key: key can be created by following the Request a Key button [at this link](https://www.census.gov/data/developers/guidance/api-user-guide.Help_&_Contact_Us.html). 
- Manual effort is required to generate mobile home polygons. Instructions for doing so are included at the point in the Jupyter notebook where the data is required. 

#### Running the Workflow

The two Jupyter Notebooks step through all steps described in the ![National Synthesis Workflow](national_workflow.pdf). 

- `Inv_Preprocess_Census_and_Footprint.ipynb` should be run first to download census data and format footprint data. 
- `Inv_Generate_Inventory.ipynb` uses the outputs from the first notebook to generate an inventory. 

This workflow was originally developed using Hayward, CA as a case study. As it is applied in new locations, developers encourage plotting the results of intermediate steps, exploring, and modifying the workflow as appropriate.

Flags to control the merge process are set per recommendations from Lochhead et al. (2026), and can be left as is when applied in new locations. 

