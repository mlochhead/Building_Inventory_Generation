# inventory_generation_city

This module makes an R2D building inventory for a city. It runs the five stages of the
National workflow, called A, B, C, D, and E:

- Stage A prepares the NSI structure points and adds HIFLD facility data to them.
- Stage B attaches the NSI points to building footprints.
- Stage C selects one value for each feature when the sources disagree.
- Stage D fills the gaps in the data.
- Stage E maps the data to the features that the simulation requires, and writes the R2D
  inventory.

Stages C, D, and E always run together in one script. This is why the driver files carry
the name `CDE`. The framework is described in
[Lochhead et al. (2026)](https://doi.org/10.1016/j.ijdrr.2026.106148).

The file `Run_Inventory_2026.py` is a complete, working example. It runs all five stages
for six cities. Use it as the template for your own runs. This guide refers to it
throughout.

## Terms

- **NSI**: the National Structure Inventory, a USACE dataset with one point per structure.
- **HIFLD**: Homeland Infrastructure Foundation-Level Data, public datasets of schools,
  colleges, fire and police stations, and emergency operations centers (EOC).
- **BRAILS**: the NHERI SimCenter tool that downloads USA Structures building footprints.
- **R2D**: the NHERI SimCenter tool for regional damage simulation. The final inventory
  CSV is in its input format.
- **FIPS code**: the census number for a state (2 digits) or county (3 digits).

## 1. Set up the environment

The code needs Python 3.10 and the packages in `requirements.txt` at the repository root.
The root README shows the full setup. In short:

```bash
python -m venv venv/inventory_env
venv/inventory_env/Scripts/pip install -r requirements.txt
```

On Windows, the environment's Python is `venv/inventory_env/Scripts/python`. On macOS and
Linux, it is `venv/inventory_env/bin/python`. The commands in this guide use the Windows
form. Replace `Scripts` with `bin` on macOS and Linux.

## 2. What you must supply yourself

The `Input_Data/` folder is not tracked in git. A fresh clone does not contain it. Most of
its content is downloaded by the workflow itself (see section 3). You must supply only
these items by hand:

1. **A census API key.** Request a free key at https://api.census.gov/data/key_signup.html.
   Put the key on one line in `Input_Data/Census/census_api_key.txt`.
2. **The HIFLD facility files.** Download these datasets as GeoJSON from the HIFLD open
   data portal and put them in `Input_Data/HIFLD/`:
   - `public-schools-geojson.geojson`
   - `private-schools-geojson.geojson`
   - `colleges-and-universities-campuses-geojson.geojson`
   - `colleges-and-universities-geojson.geojson`
   - `Fire_and_Emergency_Medical_Service_(EMS)_Stations.geojson`
   - `Local_Law_Enforcement_Locations.geojson`
   - `Local_Emergency_Operations_Centers_EOC.geojson`
   - `State_Emergency_Operations_Centers_EOC.geojson`
   - `Mobile_Home_Parks.geojson` (used to prepare the mobile home step, item 4)

   The `hifld_paths` dictionary near the top of `Run_Inventory_2026.py` shows how these
   paths are given to the workflow.
3. **The Hazus cost table.** Put `Hazus_Cost.csv` in `Input_Data/National/`. It contains
   the Hazus replacement cost rates per occupancy class.
4. **Mobile home park polygons (one file per city).** The workflow uses these polygons to
   force the RES2 occupancy inside mobile home parks. You draw them one time per city.
   The procedure comes from the example workflow notebook
   (`example_national_synthesis_workflow/Inv_Generate_Inventory.ipynb`), which contains
   the original instructions:

   1. Run `generate_mh.py`. For each city, it reads the HIFLD `Mobile_Home_Parks.geojson`
      points and writes a helper file `{city}_MH.csv` to `Input_Data/MH_Manual/`.
   2. Go to Google My Maps (https://www.google.com/maps/d/). Make a new map. Upload the
      helper CSV as a layer. The points show the mobile home parks in the city.
   3. Set the base map to satellite.
   4. Add a new layer with the name `MH_Polygons`.
   5. Use the "Draw a line" tool to draw a polygon around each mobile home park. Draw
      around the park, not around each house. Use the HIFLD points as the reference.
   6. Export the `MH_Polygons` layer as a CSV (three dots next to the layer name, then
      "Export data", then "CSV").
   7. Save the file as `Input_Data/MH_Manual/MH_{city}_Manual_Polygons.csv`.

   For a city with no mobile home parks, export the empty `MH_Polygons` layer the same
   way.

## 3. What the workflow downloads for you

Each `City` object has download methods. You call them one time for a new city. The order
matters, because the NSI download needs the city boundary.

Run these calls from the `inventory_generation_city/` folder. Some functions write cache
files with relative paths, and they only resolve from that folder.
(`Run_Inventory_2026.py` protects itself with `os.chdir` for the same reason.)

```python
from inventory_generation_city.city import City

my_city = City(
    city_name="Berkeley", state_name="California", state_abbrev="CA",
    state_fips="06", county_name="Alameda", county_fips="001",
    xbounds=(-122.34, -122.23), ybounds=(37.84, 37.91), stories_limit=30)

# 1. Census boundary shapefiles (tracts, blocks, places) from census.gov
my_city.download_census_boundaries(years=2020)

# 2. Cut the county files down to the city: boundary, tracts, blocks
my_city.process_census_download(years=2020, fill_holes=False, plot=False)

# 3. Building footprints (USA Structures) through the BRAILS scraper
my_city.download_raw_footprint()

# 4. NSI structure points inside the city boundary, from the USACE API
my_city.download_nsi(census_year=2020)

# 5. Filter and prepare the footprints
my_city.save_processed_footprint(
    min_area_ft2=450., census_year=2020, overlap_limit=0.7, plot=False)
```

Stage C also downloads the census block population and unit counts through the census
API. This happens automatically during the run and uses your API key.

Notes on the arguments:

- `fill_holes` controls the city polygon. Some cities fully surround smaller cities, and
  the census polygon has holes there. `False` keeps the true boundary and is the normal
  choice. `True` fills the holes, so the enclosed cities become part of the study area.
  Use the same value every time you process a city. A mixed set of files causes tract
  lists and boundaries that do not agree.
- The NSI API serves live data. A download on two different days can give two different
  inventories. Keep the downloaded files if you want reproducible runs.
  `Run_Inventory_2026.py` does this with its `DOWNLOAD_NSI` switch: `True` downloads new
  data, `False` reuses the saved files.
- The NSI download for a large city is more than 1 GB in one request. It can fail on a
  bad connection. The phase -1 block of `Run_Inventory_2026.py` retries three times.

## 4. Defining a city

The `City` objects for the six configured cities are at the end of `city.py`. To add a
city, make a new `City(...)` object there and add it to the `CITIES` list in your run
script. The arguments:

- `city_name` must match the census place name and a name that the BRAILS scraper can
  find (for example "Berkeley", not "Berkeley, CA").
- `state_fips` and `county_fips` are the census FIPS codes.
- `xbounds` and `ybounds` are the longitude and latitude limits of a box that contains
  the whole city, with a small margin. The workflow uses them for plot windows.
- `stories_limit` caps the number of stories that stage B accepts from the building
  height estimate. Search for the tallest building in the city and add one. Example from
  the Hayward workflow: the tallest building is approximately 11 stories, so the limit
  is 12. Buildings over the limit are reset from the mean of their occupancy class.

## 5. Running the five stages

The per-city loop in `Run_Inventory_2026.py` shows the full sequence. The calls, in
order:

```python
# ---- Stage A: prepare NSI and add HIFLD facilities
city.setup_national_preprocess_dirs()
city.augment_nsi(
    census_year=2020,
    hifld_paths=hifld_paths,        # the dict of HIFLD file paths, section 2
    min_area_filter_ft2=450.,
    plot=False,
    use_nsi_26=True,                # use the NSI-2026 school/government functions
)

# ---- Stage B: attach points to footprints
city.setup_national_footprint_attribution_dirs()
city.attribute_points_to_footprints(census_year=2020, estimate_stories=True, plot=False)
city.finalize_national_inventory(census_year=2020, review_map=False, plot=False)

# ---- Stages C, D, and E: select values, fill gaps, map to simulation features
city.setup_national_inventory_generation_dirs()
city.generate_inventory_all_fields(census_api_key=census_api_key, plot=False)   # C
city.prepare_imputation_csv()                                                   # D
city.impute_inventory_data()                                                    # D
city.infer_structure_type()                                                     # E
city.export_inventory_for_r2d()                                                 # E
```

What each stage does:

- **Stage A** (`augment_nsi`): reads the raw NSI points. Renames the fields to the
  workflow schema. Compares the points to the HIFLD school, college, fire, police, and
  EOC locations. Points that disagree with HIFLD get a new class. They are not deleted.
  Adds HIFLD facilities that NSI does not have.
- **Stage B** (`attribute_points_to_footprints`, `finalize_national_inventory`): puts
  each NSI point on its building footprint. Merges the points that share a footprint.
  Estimates the number of stories from the building height.
- **Stage C** (`generate_inventory_all_fields`): selects one value for each feature when
  the merged points disagree. For the occupancy class, the rules in
  `helpers.resolve_within_source_disagreement` select the winner, and the original class
  list is kept in the `NSI_Occupancies` column. This stage also assigns census units and
  computes the plan area and the costs.
- **Stage D** (`prepare_imputation_csv`, `impute_inventory_data`): fills the missing
  attribute values with the BRAILS KNN imputer. The occupancy class is never filled by
  imputation.
- **Stage E** (`infer_structure_type`, `export_inventory_for_r2d`): finds a structure
  type for each building from the Hazus rulesets, assigns the design level and the height
  class, and writes the final R2D inventory files.

## 6. Running everything with one command

For the six configured cities:

```bash
venv/inventory_env/Scripts/python -u inventory_generation_city/Run_Inventory_2026.py
```

The script pins its own working directory, so you can start it from any folder. It shows
progress on screen and writes it to `inventory_generation_city/Inventory_2026.log`. The
fast cities run first. Los Angeles is last and takes some hours.

The script copies each city inventory to `_INVENTORY_2026/` immediately when that city
completes (the `save_final` function). If the run stops early, the completed cities stay
safe. It also writes an audit of the HIFLD reclassifications to the log
(`save_augmentation_audit`), because the columns that record them are deleted with the
temporary files.

Settings at the top of the script:

| Setting | Function |
|---|---|
| `CITIES` | The cities to run, in order. |
| `CENSUS_YEAR` | The census geography year. Use 2020 with the 2026 NSI. |
| `USE_NSI_26` | Use the NSI-2026 school and government synthesis functions. |
| `DOWNLOAD_NSI` | Download new NSI data, or reuse the saved files. |

## 7. Check the results

After the run, make these checks before you use the inventories:

1. The last log line says `ALL DONE` with `Failed: none`.
2. The log shows one `kept {city}_R2D_Inventory.csv` line per city, with a plausible
   size. As a reference, the six configured cities range from approximately 10 MB
   (Salt Lake City) to approximately 140 MB (Los Angeles).
3. The row count of each CSV is plausible. As a reference: Salt Lake City has
   approximately 53,000 buildings, San Francisco 45,000, Washington 61,000, Seattle
   157,000, Memphis 199,000, Los Angeles 690,000.
4. No empty occupancy classes: read the CSV with pandas and confirm
   `df['OccupancyClass'].isna().sum() == 0`.
5. The log's augmentation audit block per city shows nonzero counts for the convert
   actions. All zeros means the HIFLD reclassification silently did nothing.

## 8. Outputs

The result is one CSV per city in `_INVENTORY_2026/`. Each row is one building. The
columns are: coordinates, plan area, stories, year built, replacement cost, structure
value, structure type, building type, design level, height class, occupancy class, units,
NSI day and night population, census block and tract, footprint id, and
`NSI_Occupancies`.

`NSI_Occupancies` lists every NSI occupancy class that was on the footprint before the
stage C rules selected one, as `occtype:count` pairs, for example `RES1-1SNB:3;COM1:1`.
Analysis tools use it to see if a missing building type in a tract is a wrong label or a
true gap.

`_BASELINE_prev/` holds the same six files from the 2022 baseline run, for comparison.
Git does not track these two folders. The Los Angeles CSV is larger than the GitHub file
size limit.

## 9. Cleaning up

The run makes approximately 21 GB of temporary files. Wait until the log shows
`ALL DONE`. Then:

```bash
venv/inventory_env/Scripts/python inventory_generation_city/Cleanup_After_Run.py --dry-run
```

Examine the two lists (delete and keep). Then run the same command with `--delete`. The
cleanup keeps the final inventories, the NSI downloads, the run log, and all the inputs.
These are sufficient to repeat the run.

## 10. Troubleshooting

- **The NSI download fails or hangs.** The API sends the whole city in one response, more
  than 1 GB for a large city. Try again. The run script already retries three times. If
  the download completes but the feature count in the log is implausibly small, do not
  continue. Delete the file and download again.
- **The BRAILS footprint scraper finds no city.** The scraper looks the city up by name.
  Make sure `city_name` plus `state_name` resolve to the correct place. Test with the
  full name in OpenStreetMap first.
- **Stage C fails immediately.** The census units download needs the API key file and the
  correct working directory. Make sure `Input_Data/Census/census_api_key.txt` exists, and
  start the run through `Run_Inventory_2026.py`, which sets the working directory itself.
- **Memory errors on Los Angeles.** The largest city needs the most memory in stages B
  and D. Close other large programs. Run the city alone if necessary.
- **A run stops in the middle.** Completed cities are already in `_INVENTORY_2026/`. Fix
  the cause, remove the completed cities from `CITIES`, and start the script again.
