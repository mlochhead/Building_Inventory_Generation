## DESCRIPTION

This repository contains Python functions to process and synthesize building inventory data for regional risk assessments. This repository represents the code used in the Inventory Synthesis Framework discussed in Lochhead et al. (in press). The preprint for the paper is available [here](https://doi.org/10.31224/6008).

This repository is organized into several main folders. 

1. **`inventory_generation_functions`**: Contains Python functions used to process and synthesize inventory data from various sources.

2. **`inventory_generation_hayward`**: Contains all scripts, input data, and output data demonstrating the inventory synthesis framework for Hayward, CA. The Jupyter notebooks in this folder call functions from `inventory_generation_functions` as needed. This folder contains all code and figures used in Lochhead et al. (in press). However, the Jupyter notebook files could serve as examples for inventory generation elsewhere, particularly for the National Synthesis workflow. Some input data and minor modifications to the Jupyter notebooks would be required, but the same Python functions can be used to process NSI, HIFLD, Census, and footprint data across the United States.

3. **`inventory_generation_hayward26`** (under development): Adapts the inventory synthesis scripts for the forthcoming 2026 NSI Update using a prerelease of the data. This folder is not intended for external use at this time.

4. **`point_to_footprint_example`**: An example of one part of the inventory synthesis framework, attributing point data to footprint data. The Jupyter notebooks in this folder call functions from `inventory_generation_functions` as needed. 


## INVENTORY SYNTHESIS FRAMEWORK

All scripts in this repository support the Inventory Synthesis Framework from Lochhead et al. (under review), as shown below.

![Inventory Synthesis Framework](Inventory_Synthesis_Framework.png)


## HOW TO RUN

The scripts in this repository require Python 3.10 to run. While scripts may be compatible with other Puthon verisons, compatibility is not guaranteed.

The following instructions are intended to help new users get started with the scripts. These specific commands are not mandatory, as long as Python 3.10 and the dependencies listed in `requirements.txt` are used. The instructions below represent one way to set up the environment.

### Clone the Repository 

To begin, select a local directory for the repository, then open a terminal and run the following command:

`git clone https://github.com/mlochhead/Building_Inventory_Generation.git`

Once cloning is complete, navigate into the repository directory:

`cd Building_Inventory_Generation`

### Setting Up a Virtual Environment on macOS/Linux

Using a virtual environment is recommended for managing project dependencies. To create and activate the virtual environment, use the following commands. These commands should be executed from the terminal in the repository directory: 

`python3.10 -m venv inventory_env`

`source inventory_env/bin/activate`

After activating the virtual environment, install the required dependencies by running:

`inventory_env/bin/python -m pip install -r requirements.txt`

Once the environment is set up and activated, you can run the Jupyter notebooks and scripts in this repository.


### Setting Up a Virtual Environment on Windows

Using a virtual environment is recommended for managing project dependencies. To create and activate the virtual environment, use the following commands. These commands should be executed from the terminal in the directory where the repository has been cloned:

`python3.10 -m venv inventory_env`

`inventory_env\Scripts\activate`

After activating the virtual environment, install the required dependencies by running:

`pip install -r requirements.txt`

Once the environment is set up and activated, you can run the Jupyter notebooks and scripts in this repository.


## LICENSE

This repository is available under the BSD 3-Clause license, see LICENSE.

## CONTACT

Meredith (Mia) Lochhead, Stanford University, mlochhea@stanford.edu

