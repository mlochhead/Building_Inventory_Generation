# Post-run cleanup. Run this ONLY after Run_Inventory_2026.py reports ALL DONE.
#
# Deletes everything the run derived, keeping the final inventories, the NSI downloads, and
# the run log. Pass --dry-run first to see what would go.
#
#   python inventory_generation_city/Cleanup_After_Run.py --dry-run
#   python inventory_generation_city/Cleanup_After_Run.py --delete

import shutil
import sys
from pathlib import Path

B = Path(__file__).resolve().parent

# ---- derived by the run, all reproducible by re-running -----------------------------
DELETE = [
    (B / "Input_Data" / "ProcessedData",        "stage A and B handoff files"),
    (B / "Input_Data" / "ProcessedFootprints",  "rebuilt by phase 0 from BRAILS raw"),
    (B / "Inventory_Outputs",                   "FootprintAttribution and InventoryGeneration"),
    (B / "R2D_Analysis",                        "final exports, already copied to _INVENTORY_2026"),
    (B / "Output_Data",                         "folium maps"),
    (B / "Figures",                             "plots"),
]

# ---- kept --------------------------------------------------------------------------
KEEP = [
    (B / "_INVENTORY_2026",                     "THE DELIVERABLE: 6 R2D inventories, 2026 run"),
    (B / "_BASELINE_prev",                      "THE COMPARISON: 6 R2D inventories, 2022 baseline"),
    (B / "Input_Data" / "National_NSI2026",     "NSI downloads, cannot be re-fetched identically"),
    (B / "Inventory_2026.log",                  "run record incl. the augmentation audit"),
    (B / "Input_Data" / "National",             "BRAILS raw footprints + Hazus_Cost.csv"),
    (B / "Input_Data" / "Census",               "census blocks, tracts, boundaries, API key"),
    (B / "Input_Data" / "HIFLD",                "school, fire, police, EOC datasets"),
    (B / "Input_Data" / "MH_Manual",            "mobile home park polygons"),
]


def size(p):
    p = Path(p)
    if p.is_file():
        return p.stat().st_size
    if p.is_dir():
        return sum(f.stat().st_size for f in p.rglob("*") if f.is_file())
    return -1


def gb(n):
    return n / 1024 ** 3


mode = sys.argv[1] if len(sys.argv) > 1 else "--dry-run"

print("=" * 76)
print("DELETE" if mode == "--delete" else "WOULD DELETE")
print("=" * 76)
tot = 0
for p, why in DELETE:
    s = size(p)
    if s < 0:
        print(f"  {p.name:26} (already gone)")
        continue
    tot += s
    print(f"  {p.name:26} {gb(s):>7,.2f} GB   {why}")
    if mode == "--delete":
        shutil.rmtree(p, ignore_errors=True)
print(f"\n  {'total':26} {gb(tot):>7,.2f} GB")

print("\n" + "=" * 76)
print("KEEP")
print("=" * 76)
ktot = 0
for p, why in KEEP:
    s = size(p)
    if s < 0:
        print(f"  {p.name:26}  MISSING   {why}")
        continue
    ktot += s
    print(f"  {p.name:26} {gb(s):>7,.2f} GB   {why}")
print(f"\n  {'total':26} {gb(ktot):>7,.2f} GB")

if mode == "--delete":
    print("\nDeleted. Re-running the pipeline regenerates everything removed above.")
else:
    print("\nDry run only. Re-run with --delete to apply.")
