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

for city in cities:
    print(f"Processing {city.city_name}")
    city.download_census_boundaries(years=[2010, 2020])
    city.download_raw_footprint()
    city.process_census_download(
        years=[2010, 2020],
        fill_holes=False,
        plot=city.city_name != "Los Angeles",
    )
    city.save_processed_footprint(
        min_area_ft2=450.,
        census_year=2010,
        overlap_limit=0.7,
        plot=city.city_name != "Los Angeles",
    )
    plt.close('all')