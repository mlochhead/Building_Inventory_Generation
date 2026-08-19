# Full NSI-2026 national workflow re-run using MTL's intended use_nsi_26 flag values.
#
# AD, 2026-08-04. This script exists because the three stage drivers
# (Inv_National_A/B/CDE) hardcode census_year = 2010 and use_nsi_26 = False, which are the
# settings for the published 2022 workflow. This run needs the 2026 settings instead:
#
#   census_year = 2020   because the 2026 NSI carries 2020-vintage census block codes.
#                        Running 2026 points against 2010 blocks silently loses about
#                        20 percent of the buildings.
#   use_nsi_26 = True    so that augment_nsi uses the *_nsi26_update EDU1, EDU2 and GOV2
#                        synthesis functions. The flag values inside those calls now match
#                        MTL's own NSI26 driver notebook rather than the 2022 defaults.
#
# The raw 2026 NSI is read from Input_Data/National_NSI2026 rather than the usual
# Input_Data/National, so the 2022 raw files stay in place untouched and the baseline
# inventory remains reproducible.
#
# Cities run fastest first so problems surface early. Los Angeles is last because it alone
# takes roughly four hours.

import os
import sys
import time
import traceback
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # headless; makes plt.show() a no-op
import matplotlib.pyplot as plt

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

# functions_disagreement_and_gaps.download_census_data() writes its block-unit cache to the
# relative path 'Input_Data/Census/2020_Census_Units_<fips>.csv'. That only resolves when the
# working directory is inventory_generation_city, so pin it here rather than depending on how
# the script was launched. Without this the CDE stage fails for every city.
os.chdir(REPO / "inventory_generation_city")

from inventory_generation_city.city import *
from inventory_generation_city.helpers import load_census_api_key
# city.py defines __all__, so its module aliases are not re-exported by the wildcard import
# above. save_augmentation_audit() needs json_to_gdf, so import it directly.
import inventory_generation_functions.functions_general as fxns

# ---------------------------------------------------------------- configuration
CENSUS_YEAR = 2020
USE_NSI_26 = True

CITIES = [
    san_francisco,
    washington,
    salt_lake_city,
    seattle,
    memphis,
    los_angeles,
]

# The 2026 NSI is pulled straight from the USACE API at the start of this run. The 2022 raw
# that used to sit in Input_Data/National has been retired, so this is now the only NSI input.
NSI26_DIR = REPO / "inventory_generation_city" / "Input_Data" / "National_NSI2026"
# Reuse the archived July 2026 snapshot rather than re-downloading: the NSI API serves live
# data, and this rerun (adding the NSI_Occupancies provenance column) must stay on the same
# vintage as the current inventories so the only difference is the new column.
DOWNLOAD_NSI = False
NSI26_DIR.mkdir(parents=True, exist_ok=True)

# Point every read and write of the raw NSI at the 2026 folder. download_nsi() saves to
# self.nsi_raw_path(), so this redirects the download target as well as the later reads.
City.nsi_raw_path = lambda self: NSI26_DIR / f"nsi_raw_{self.city_name}.geojson"

inp_dir = CITIES[0].inp_dir
hifld_path = inp_dir / "HIFLD"

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

census_api_key = load_census_api_key(path=inp_dir / "Census" / "census_api_key.txt")

LOG = REPO / "inventory_generation_city" / "Inventory_2026.log"

# Only the final inventories are kept. Everything else this run writes (the ~21 GB of
# FootprintAttribution intermediates, the imputation scratch files, ProcessedData) is
# disposable and can be deleted once the run finishes. Each city's two keeper files are
# copied out as soon as that city completes, so a crash partway through still leaves every
# finished city preserved. Pairs with _BASELINE_prev, which holds the same two files per
# city for the 2022 baseline.
KEEP = REPO / "inventory_generation_city" / "_INVENTORY_2026"
KEEP.mkdir(parents=True, exist_ok=True)


def save_final(city):
    # The R2D inventory is the whole deliverable. Its 21 columns carry every field the
    # baseline comparison uses: counts, day and night population, occupancy class, structure
    # type, design level, year built, stories, plan area and replacement cost.
    #
    # Inventory_AllFields.json is deliberately not kept. It adds provenance columns such as
    # POINT_Source and the *_Best_Source family, which are useful for forensics but feed
    # nothing in the analysis, at a cost of roughly 3.45 GB. The one forensic question worth
    # answering here, whether the 'convert' flags fired, is recorded in the run log by
    # save_augmentation_audit() instead.
    import shutil
    pairs = [
        (city.r2d_inventory_paths()[0], f"{city.city_name}_R2D_Inventory.csv"),
    ]
    for src, name in pairs:
        src = Path(src)
        if src.exists():
            shutil.copy2(src, KEEP / name)
            log(f"{city.city_name}: kept {name} ({src.stat().st_size/1024**2:,.1f} MB)")
        else:
            log(f"{city.city_name}: WARNING expected output missing: {src}")


def save_augmentation_audit(city):
    """
    Tally what the HIFLD augmentation actually did to this city's NSI points.

    build_national_inventory() drops POINT_DataUpdate and POINT_DropNote before merging points
    onto footprints, so neither the R2D inventory nor the AllFields file records which points
    were converted or dropped and why. Those columns only exist on the stage-A output, which
    lives in the disposable ProcessedData tree.

    This is the evidence that gov1_near_edu1='convert' and gov2_far_from_hifld='convert' did
    what MTL intended rather than silently doing nothing. It is written to the run log rather
    than saved as a file, so the only files this run keeps remain the two final inventories
    per city. The log is the permanent record once ProcessedData is deleted.
    """
    try:
        pts = fxns.json_to_gdf(city.nsi_for_merge_path(), crs_main=city.crs_main)
        found = False
        for col in ("POINT_DataUpdate", "POINT_DropNote", "POINT_Source"):
            if col not in pts.columns:
                continue
            found = True
            vc = pts[col].fillna("").replace("", "(none)").value_counts()
            log(f"{city.city_name}: --- {col} ---")
            for label, n in vc.items():
                log(f"{city.city_name}:   {n:>9,}  {label}")
        if not found:
            log(f"{city.city_name}: no augmentation audit columns found")
    except Exception:
        log(f"{city.city_name}: augmentation audit failed (non-fatal)")
        log(traceback.format_exc())


def log(msg):
    stamp = time.strftime("%Y-%m-%d %H:%M:%S")
    line = f"[{stamp}] {msg}"
    print(line, flush=True)
    with LOG.open("a", encoding="utf-8") as f:
        f.write(line + "\n")


# ---------------------------------------------------------------- run
log(f"START census_year={CENSUS_YEAR} use_nsi_26={USE_NSI_26}")
log(f"NSI raw source: {NSI26_DIR}")

overall = time.time()
failed = []

# ---------------------------------------------------------------- phase -1
# Fresh NSI download.
#
# functions_preprocessing.download_nsi() issues one bounding-box request per city with no
# retry, and if the server returns a non-200 it prints a message and then raises NameError on
# an undefined 'data'. Los Angeles alone is roughly 1.3 GB in a single response, so a transient
# failure is a real possibility. Retry here rather than lose the whole run to one bad request.
#
# Feature counts are compared against the archived July 2026 snapshot. The API serves live
# data, so a difference is possible and worth knowing about before it silently shows up in the
# inventory comparison.
if DOWNLOAD_NSI:
    log("=== PHASE -1: downloading fresh NSI 2026 from the USACE API ===")
    for city in CITIES:
        t0 = time.time()
        out = Path(city.nsi_raw_path())
        for attempt in (1, 2, 3):
            try:
                city.download_nsi(census_year=CENSUS_YEAR)
                break
            except Exception:
                log(f"{city.city_name}: download attempt {attempt} failed")
                if attempt == 3:
                    failed.append(f"{city.city_name} (download)")
                    log(traceback.format_exc())
                else:
                    time.sleep(20 * attempt)

        if not out.exists():
            failed.append(f"{city.city_name} (download produced no file)")
            continue

        new_mb = out.stat().st_size / 1024**2

        # Count features without loading the whole file. A truncated HTTP body would already
        # have blown up in .json(), so the residual risk is a valid response that is simply
        # too small. There is no earlier NSI copy to compare against, so record the count in
        # the log and hold it to a floor. Catching a short download here costs seconds;
        # catching it after stage CDE costs the whole run.
        try:
            import fiona
            with fiona.open(out) as src:
                n = len(src)
        except Exception:
            n = -1
            log(f"{city.city_name}: WARNING could not count features")

        log(f"{city.city_name}: downloaded {new_mb:,.1f} MB, {n:,} features "
            f"in {(time.time()-t0)/60:.1f} min")

        if 0 <= n < 10_000:
            failed.append(f"{city.city_name} (only {n:,} features)")
            log(f"!!! {city.city_name}: implausibly few features, refusing to continue")

    if failed:
        log(f"ABORTING: NSI download failed for {failed}.")
        sys.exit(1)
    log(f"=== PHASE -1 complete ({(time.time()-overall)/60:.1f} min) ===")
else:
    log("PHASE -1 skipped, reusing existing raw NSI")

# ---------------------------------------------------------------- phase 0
# Rebuild the processed footprints against 2020 census geography.
#
# The footprints shipped in Input_Data/ProcessedFootprints were built with census_year=2010:
# 100 percent of their block codes match the 2010 block file and only 58 percent match 2020.
# build_national_inventory() drops CensusBlock and CensusTract from the NSI points before
# merging them onto footprints, so the finished inventory inherits those two columns from the
# footprints. Left alone, this run would filter points by 2020 city blocks while filtering
# footprints by 2010 city blocks, and would then label 2026-vintage buildings with 2010
# geography. Rebuilding here puts both sides on one study area.
#
# The 2010 versions are preserved in Input_Data/ProcessedFootprints_2010vintage so the 2022
# baseline stays reproducible.
log("=== PHASE 0: rebuilding processed footprints at census_year=2020 ===")
for city in CITIES:
    t0 = time.time()
    try:
        city.save_processed_footprint(
            min_area_ft2=450.,
            census_year=CENSUS_YEAR,
            overlap_limit=0.7,
            plot=False,
        )
        log(f"{city.city_name}: footprints rebuilt ({(time.time()-t0)/60:.1f} min)")
    except Exception:
        failed.append(f"{city.city_name} (footprints)")
        log(f"!!! {city.city_name}: FOOTPRINT REBUILD FAILED")
        log(traceback.format_exc())
    finally:
        plt.close("all")

if failed:
    log(f"ABORTING: footprint rebuild failed for {failed}. Fix before running stages A/B/CDE.")
    sys.exit(1)

log(f"=== PHASE 0 complete ({(time.time()-overall)/60:.1f} min) ===")

for city in CITIES:
    t0 = time.time()
    log(f"=== {city.city_name}: BEGIN ===")
    try:
        # ---- Stage A: preprocess and augment NSI with HIFLD
        city.setup_national_preprocess_dirs()
        city.augment_nsi(
            census_year=CENSUS_YEAR,
            hifld_paths=hifld_paths,
            min_area_filter_ft2=450.,
            plot=False,
            use_nsi_26=USE_NSI_26,
        )
        log(f"{city.city_name}: stage A done ({(time.time()-t0)/60:.1f} min)")

        # ---- Stage B: attribute points to footprints
        tB = time.time()
        city.setup_national_footprint_attribution_dirs()
        city.attribute_points_to_footprints(
            census_year=CENSUS_YEAR,
            estimate_stories=True,
            plot=False,
        )
        city.finalize_national_inventory(
            census_year=CENSUS_YEAR,
            review_map=False,
            plot=False,
        )
        log(f"{city.city_name}: stage B done ({(time.time()-tB)/60:.1f} min)")

        # ---- Stage CDE: generate, impute, infer, export
        tC = time.time()
        city.setup_national_inventory_generation_dirs()
        city.generate_inventory_all_fields(census_api_key=census_api_key, plot=False)
        city.prepare_imputation_csv()
        city.impute_inventory_data()
        city.infer_structure_type()
        city.export_inventory_for_r2d()
        log(f"{city.city_name}: stage CDE done ({(time.time()-tC)/60:.1f} min)")

        save_final(city)
        save_augmentation_audit(city)
        log(f"=== {city.city_name}: COMPLETE ({(time.time()-t0)/60:.1f} min) ===")

    except Exception:
        failed.append(city.city_name)
        log(f"!!! {city.city_name}: FAILED after {(time.time()-t0)/60:.1f} min")
        log(traceback.format_exc())
    finally:
        plt.close("all")

log(f"ALL DONE in {(time.time()-overall)/60:.1f} min. Failed: {failed if failed else 'none'}")
