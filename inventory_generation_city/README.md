# inventory_generation_city

This module makes an R2D building inventory for each configured city. It uses these data
sources: NSI structure points, BRAILS building footprints, HIFLD facility data, and census
geography. One command does the full run. One command removes the temporary files after
the run.

## Requirements

Make sure that you have these items before you start:

- The Python environment `venv/inventory_env`. It contains geopandas, BRAILS, fiona,
  folium, and matplotlib.
- A census API key in `Input_Data/Census/census_api_key.txt`. Put only the key in the
  file, on one line.
- These input folders:
  - `Input_Data/National`: the BRAILS raw footprints for each city, and `Hazus_Cost.csv`.
  - `Input_Data/National_NSI2026`: the saved NSI 2026 download, one geojson for each city.
  - `Input_Data/Census`: the tract, block, and place shapefiles for each county and state.
  - `Input_Data/HIFLD`: the school, college, fire, police, and EOC datasets.
  - `Input_Data/MH_Manual`: the mobile home park polygons for each city.

## How to run

Run this command. You can start it from any folder. The script sets its own work folder.

```bash
venv/inventory_env/Scripts/python -u inventory_generation_city/Run_Inventory_2026.py
```

The script shows progress on the screen. It also writes the progress to
`inventory_generation_city/Inventory_2026.log`. The fast cities run first. Los Angeles
runs last. Los Angeles takes some hours.

The script saves each city inventory immediately when that city completes. If the run
stops early, the completed cities stay safe.

## The steps of the run

1. Phase -1 (optional): download new NSI data from the USACE API. The `DOWNLOAD_NSI`
   setting controls this phase. Keep it `False` to use the saved download. The API sends
   live data. A new download changes the input data.
2. Phase 0: build the processed footprints again, with the 2020 census geography.
3. Stage A, for each city: add the HIFLD data to the NSI points. This step changes the
   class of unclear points. It does not delete them.
4. Stage B, for each city: connect the NSI points to the building footprints. When the
   points on one footprint do not agree, the occupancy rules in
   `helpers.resolve_within_source_disagreement` select one class.
5. Stage CDE, for each city: build the full inventory table. Fill the missing values with
   BRAILS KNN. The occupancy class is never filled by this step. Then find the structure
   types with the Hazus rules. Then write the R2D inventory.
6. Copy the final CSV to `_INVENTORY_2026/{city}_R2D_Inventory.csv`. Write the
   augmentation audit to the log.

## Settings

The settings are at the top of `Run_Inventory_2026.py`.

| Setting | Function |
|---|---|
| `CITIES` | The cities to run, in order. The City objects are at the end of `city.py`. |
| `CENSUS_YEAR` | The census geography year. Use 2020 for the NSI 2026 workflow. |
| `USE_NSI_26` | Use the NSI 2026 school and government synthesis functions. |
| `DOWNLOAD_NSI` | Download new NSI data, or use the saved download. |

To add a city: make a new `City(...)` object at the end of `city.py`. Give the name, the
state and county FIPS codes, the bounding box, and the stories limit. Put the input files
in the folders above. Add the object to `CITIES`.

## Outputs

The result is one CSV for each city, in `_INVENTORY_2026/`. Each row is one building.
The columns are: the point coordinates, the plan area, the number of stories, the year
built, the replacement cost, the structure value, the structure type, the design level,
the height class, the occupancy class, the number of units, the NSI day and night
population, the census block and tract, and the footprint id.

The column `NSI_Occupancies` is also included. It lists the NSI occupancy classes of all
the points that were on the footprint, before the occupancy rules selected one class. The
format is `occtype:count`, for example `RES1-1SNB:3;COM1:1`. Analysis tools use this
column. It shows if a missing building type in a tract is a wrong label or a true gap.

The folder `_BASELINE_prev/` contains the same six files from the 2022 baseline run. Use
them for comparison. Git does not track these two folders. The Los Angeles CSV is larger
than the GitHub file size limit.

## How to clean up

The run makes approximately 21 GB of temporary files. Wait until the log shows
`ALL DONE`. Then run this command:

```bash
venv/inventory_env/Scripts/python inventory_generation_city/Cleanup_After_Run.py --dry-run
```

The command shows two lists: the files it will delete, and the files it will keep. Examine
the lists. Then run:

```bash
venv/inventory_env/Scripts/python inventory_generation_city/Cleanup_After_Run.py --delete
```

The cleanup keeps the inventories, the baseline, the NSI download, the run log, and all
the input folders. These files are sufficient to do the same run again.
