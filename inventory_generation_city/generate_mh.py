from inventory_generation_city.city import *

census_year = 2010

cities = [
    san_francisco,
    los_angeles,
    salt_lake_city,
    seattle,
    memphis,
    washington
]

inp_dir = cities[0].inp_dir
mh_hifld_path = inp_dir / "HIFLD" / "Mobile_Home_Parks.geojson"

for city in cities:
    city.export_mh_points(
        census_year=census_year,
        mh_hifld_path=mh_hifld_path,
    )
    city.generate_inventory_fields(
        census_api_key_path=inp_dir / "Census" / "census_api_key.txt",
    )