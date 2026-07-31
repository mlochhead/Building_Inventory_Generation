import matplotlib.pyplot as plt
from inventory_generation_city.city import *
from inventory_generation_city.helpers import load_census_api_key

cities = [
    # san_francisco,
    # los_angeles,
    # salt_lake_city,
    seattle,
    memphis,
    washington
]

inp_dir = cities[0].inp_dir

# Note: the NSI 2026 methodology toggle (use_nsi_26) lives in Inv_National_A_Preprocess.py.
# The *_nsi26_update functions are all in the preprocess/augment step, so this step (CDE) has
# no NSI26-specific branch.

census_api_key = load_census_api_key(
    path=inp_dir / "Census" / "census_api_key.txt"
)

for city in cities:
    print(f"Processing {city.city_name}")
    city.generate_inventory_all_fields(
        census_api_key=census_api_key,
    )
    city.prepare_imputation_csv()
    city.impute_inventory_data()
    city.infer_structure_type()
    city.export_inventory_for_r2d()
    plt.close('all')
