# Copyright (c) 2025, Meredith Lochhead
# All rights reserved.
#
# This source code is licensed under the BSD 3-Clause License found in the
# LICENSE file in the root directory of this source tree.

from inventory_generation_city.city import *
import matplotlib.pyplot as plt

cities = [
    san_francisco,
    los_angeles,
    salt_lake_city,
    seattle,
    memphis,
    washington
]

census_year = 2010
hifld_path = cities[0].inp_dir / "HIFLD"

hifld_paths = {
    "EDU1": {
        "public_schools_path": hifld_path / "public-schools-geojson.geojson",
        "private_schools_path": hifld_path / "private-schools-geojson.geojson",
    },
    "EDU2": {
        "univ_campuses_path": hifld_path / "colleges-and-universities-campuses-geojson.geojson",
        "univ_points_path": hifld_path / "colleges-and-universities-geojson.geojson",
    },
    "GOV2": {
        "fire_path": hifld_path / "Fire_and_Emergency_Medical_Service_(EMS)_Stations.geojson",
        "police_path": hifld_path / "Local_Law_Enforcement_Locations.geojson",
        "local_eoc_path": hifld_path / "Local_Emergency_Operations_Centers_EOC.geojson",
        "state_eoc_path": hifld_path / "State_Emergency_Operations_Centers_EOC.geojson",
    },
}

for city in cities:
    print(f"Processing {city.city_name}")
    city.setup_national_preprocess_dirs()
    city.download_nsi(census_year=census_year)
    city.augment_nsi(
        census_year=census_year,
        hifld_paths=hifld_paths,
        min_area_filter_ft2=450.,
        plot=city.city_name != "Los Angeles",
    )
    plt.close('all')