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

for city in cities:
    print(f"Processing {city.city_name}")
    city.setup_national_footprint_attribution_dirs()
    city.attribute_points_to_footprints(
        census_year=census_year,
        estimate_stories=True,
        plot=city.city_name != "Los Angeles",
    )
    city.finalize_national_inventory(
        census_year=census_year,
        review_map=True,
        plot=city.city_name != "Los Angeles",
    )
    plt.close('all')