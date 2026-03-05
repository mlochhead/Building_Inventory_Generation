from __future__ import annotations
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from typing import Dict, Union, Tuple
    from inventory_generation_city.helpers import YearInput, FillHolesInput, CompareInput, ColorsInput

import inventory_generation_functions.functions_preprocessing as pre
import inventory_generation_functions.functions_general as fxns
import inventory_generation_city.helpers as hf
import inventory_generation_functions.functions_point_to_ftpt as pt_ftpt
import inventory_generation_functions.functions_disagreement_and_gaps as resolve
import json
from pathlib import Path
from brails.types.region_boundary import RegionBoundary
from brails.scrapers.usa_footprint_scraper.usa_footprint_scraper import USA_FootprintScraper
from brails.utils import Importer
from brails.types.asset_inventory import AssetInventory
import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np


class City:
    base_dir: Path = Path(__file__).resolve().parent
    out_dir: Path = base_dir / "Output_Data"
    inp_dir: Path = base_dir / "Input_Data"
    fig_dir: Path = base_dir / "Figures"
    inv_dir: Path = base_dir / "Inventory_Outputs"
    r2d_dir: Path = base_dir / "R2D_Analysis"
    crs_main = "26910"  # Used for data manipulation and storage
    crs_plot = "4269"  # Used for plotting

    # Set category of certain column names
    excluded = ['geometry', 'CensusBlock', 'CensusTract', 'POINT_ID', 'NSI_OccupancyClass', 'POINT_DropFlag',
                'POINT_DropNote', 'NSI_OC_Update',
                'POINT_FootprintID', 'DistanceToFtpt', 'ClosestFtpt_ID', 'POINT_MergeFlag', 'POINT_DataUpdate']
    sum_columns = ['NSI_PopOver65_Day', 'NSI_PopUnder65_Day', 'NSI_Population_Day',
                   'NSI_PopOver65_Night', 'NSI_PopUnder65_Night', 'NSI_Population_Night', 'NSI_ContentValue',
                   'NSI_ReplacementCost',
                   'NSI_StructureValue', 'NSI_MinResUnits', 'NSI_MaxResUnits', 'POINT_NumPoints']
    list_columns = ['NSI_FoundationType', 'NSI_FoundationHeight', 'NSI_BuildingType', 'NSI_MedYearBuilt',
                    'NSI_NumberOfStories', 'POINT_Source', 'NSI_OrigSource', 'NSI_OrigFtptSource', 'NSI_BID',
                    'POINT_ID_List', 'NSI_TotalAreaSqFt']

    def __init__(self, city_name: str, state_name: str, state_abbrev: str, state_fips: str, county_name: str, county_fips: str, xbounds: tuple[float, float], ybounds: tuple[float, float], stories_limit: int):
        self.city_name = city_name
        self.state_name = state_name
        self.state_abbrev = state_abbrev
        self.state_fips = state_fips
        self.county_name = county_name
        self.county_fips = county_fips
        self.xbounds = xbounds
        self.ybounds = ybounds
        self.stories_limit = stories_limit

    @classmethod
    def maps_dir(cls) -> Path:
        return cls.out_dir / "Maps"

    def nsi_raw_path(self) -> Path:
        return self.inp_dir / f"National/nsi_raw_{self.city_name}.geojson"

    def national_preprocessed_dir(self) -> Path:
        return self.inp_dir / "ProcessedData/National"

    def national_footprint_attribution_dir(self) -> Path:
        return self.inv_dir / "Synthesized_National/FootprintAttribution"

    def national_inventory_generation_dir(self) -> Path:
        return self.inv_dir / "Synthesized_National/InventoryGeneration"

    def r2d_inventory_dir(self) -> Path:
        return self.r2d_dir / "Inventories/Synthesized_National"

    def r2d_inventory_paths(self) -> tuple[Path, Path, Path]:
        base_dir = self.r2d_inventory_dir()
        return base_dir / f"{self.city_name}_R2D_Inventory.csv", base_dir / f"{self.city_name}_R2D_Inventory.json", base_dir / f"{self.city_name}_R2D_Inventory_SAMPLED.csv"

    def setup_national_preprocess_dirs(self) -> tuple[Path, Path, Path]:
        p1 = self.national_preprocessed_dir()
        p2 = p1 / "Intermediate"
        p3 = self.fig_dir / "General"
        for p in [p1, p2, p3]:
            p.mkdir(parents=True, exist_ok=True)
        return p1, p2, p3

    def setup_national_footprint_attribution_dirs(self) -> tuple[Path, Path, Path]:
        p1 = self.national_footprint_attribution_dir()
        p2 = p1 / "Intermediate"
        p3 = self.fig_dir / "General"
        for p in [p1, p2, p3]:
            p.mkdir(parents=True, exist_ok=True)
        return p1, p2, p3

    def setup_national_inventory_generation_dirs(self) -> tuple[Path, Path]:
        p1 = self.national_inventory_generation_dir()
        p2 = self.r2d_inventory_dir()
        for p in [p1, p2]:
            p.mkdir(parents=True, exist_ok=True)
        return p1, p2

    def census_year_config(self, year: int) -> dict:
        year_suffix = str(year)[-2:]
        root = self.inp_dir / f"Census/Census{year}"
        state = self.census_download_kwargs(year)['state']

        return {
            "root_dir": root,
            "tract_shp": root / state / self.county_name / f"tl_{year}_{self.state_fips}{self.county_fips}_tract{year_suffix}.shp",
            "block_shp": root / state / self.county_name / f"tl_{year}_{self.state_fips}{self.county_fips}_tabblock{year_suffix}.shp",
            "place_shp": root / state / f"tl_{year}_{self.state_fips}_place{year_suffix}.shp",
            "place_name_col": f"NAME{year_suffix}",
            "cb_id_name": f"GEOID{year_suffix}",
        }

    def mh_points_path(self):
        return self.inp_dir / "MH_Manual" / f"{self.city_name}_MH.csv"

    def mh_manual_polygons_path(self) -> Path:
        return self.inp_dir / "MH_Manual" / f"MH_{self.city_name}_Manual_Polygons.csv"

    def census_path(self, year: int) -> Path:
        return self.census_year_config(year)["root_dir"] / f"{self.city_name}_census.geojson"

    def census_blocks_path(self, year: int) -> Path:
        return self.census_year_config(year)["root_dir"] / f"{self.city_name}_blocks.geojson"

    def census_tracts_path(self, year: int) -> Path:
        return self.census_year_config(year)["root_dir"] / f"{self.city_name}_tracts.geojson"

    def raw_footprint_path(self) -> Path:
        return self.inp_dir / f'National/BRAILS_usastr_{self.city_name}.geojson'

    def processed_footprint_path(self) -> Path:
        return self.inp_dir / f'ProcessedFootprints/Footprints_{self.city_name}.json'

    def nsi_for_merge_path(self) -> Path:
        base_dir, _, _ = self.setup_national_preprocess_dirs()
        return base_dir / f"{self.city_name}_NSI_for_Merge.json"

    def hazus_cost_path(self) -> Path:
        return self.inp_dir / f'National/Hazus_Cost.csv'

    def national_inventory_paths(self) -> Tuple[Path, Path]:
        base_dir, _, _ = self.setup_national_footprint_attribution_dirs()
        return base_dir / f"{self.city_name}_National_Inventory_Polygon.json", base_dir / f"{self.city_name}_National_Inventory_Point.json",

    def inventory_all_fields_path(self) -> Path:
        return self.national_inventory_generation_dir() / f'{self.city_name}_Inventory_AllFields.json'

    def inventory_before_imputation_path(self) -> Path:
        return self.national_inventory_generation_dir() / f'{self.city_name}_Inventory_Before_Imputation.csv'

    def imputed_inventory_path(self) -> Path:
        return self.national_inventory_generation_dir() / f'{self.city_name}_Inventory_IMPUTED.json'

    def imputed_and_inferred_inventory_path(self) -> Path:
        return self.national_inventory_generation_dir() / f'{self.city_name}_Inventory_IMPUTED_With_StructureType.json'

    def full_name(self) -> str:
        return f"{self.city_name}, {self.state_name}"

    def census_download_kwargs(self, year: int) -> dict:
        parts = self.state_name.lower().replace("_", " ").split()
        if year == 2010:
            to_join = [
                p.capitalize() if p != "of" or "the" in parts else p
                for p in parts
            ]
            state = "_".join(to_join)
        elif year == 2020:
            state = "_".join(p.upper() for p in parts)
        else:
            raise ValueError("year must be 2010 or 2020")
        return {
            "state": state,
            "state_fips": self.state_fips,
            "county": self.county_name,
            "county_fips": self.county_fips,
        }

    def assign_census_to_footprints(
            self,
            *,
            footprints: gpd.GeoDataFrame,
            census_year: int,
    ) -> tuple[gpd.GeoDataFrame, gpd.GeoDataFrame, gpd.GeoDataFrame]:
        city_blocks = gpd.read_file(self.census_blocks_path(census_year))
        city_tracts = gpd.read_file(self.census_tracts_path(census_year))

        city_footprints = pre.assign_footprint_block_and_track(
            footprints, city_tracts, city_blocks, self.census_year_config(int(census_year))["cb_id_name"]
        )
        return city_footprints, city_blocks, city_tracts

    def download_census_boundaries(self, years: YearInput) -> None:
        """
        Download census boundaries for the city for the specified year.
        """
        if isinstance(years, int):
            years = [years]
        if 2020 in years:
            pre.download_2020_census_boundaries(**self.census_download_kwargs(year=2020))
        if 2010 in years:
            pre.download_2010_census_boundaries(**self.census_download_kwargs(year=2010))

    def download_raw_footprint(self):
        region = RegionBoundary({"type": "locationName", "data": self.full_name()})
        # Download USA Structures footprints (polygons + buildingheight attribute)
        scraper = USA_FootprintScraper({"length_unit": "ft"})
        inv = scraper.get_footprints(region)
        # Export to GeoJSON
        geojson_dict = inv.get_geojson()
        out_path = self.raw_footprint_path()
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with out_path.open("w", encoding="utf-8") as f:
            json.dump(geojson_dict, f, indent=2)
        print(f"Saved: {out_path.resolve()}")

    def download_nsi(self, census_year: int):
        city_boundary_path = self.census_path(census_year)
        city_boundary = gpd.read_file(city_boundary_path).to_crs(crs=f"EPSG:{int(self.crs_plot)}")
        pre.download_nsi(city_boundary, self.crs_plot, self.nsi_raw_path())

    def augment_nsi(
            self, census_year: int,
            hifld_paths: dict,
            min_area_filter_ft2: float,
            plot: bool = True,
    ) -> None:
        nsi = gpd.read_file(self.nsi_raw_path())
        nsi = nsi.to_crs(crs=f"EPSG:{int(self.crs_main)}")
        nsi = pre.rename_nsi_data(nsi.copy())

        _, intermediate_dir, _ = self.setup_national_preprocess_dirs()
        cb_id_name = self.census_year_config(int(census_year))["cb_id_name"]

        # Merge NSI data with City-Specific Census Blocks and check for errors in NSI data
        city_blocks = gpd.read_file(self.census_blocks_path(census_year))
        city_tracts = gpd.read_file(self.census_tracts_path(census_year))
        nsi = pre.assign_point_block_and_track(nsi, city_blocks, city_tracts, cb_id_name=cb_id_name)

        # Plot Results
        fig, ax = plt.subplots(1, 1, figsize=(6, 6))
        city_blocks.plot(ax=ax)
        nsi.plot(ax=ax, color='black', markersize=0.1)
        ax.set_title(f'NSI Points within {self.city_name}')
        plt.show()

        # ##### CREATE ADDITIONAL COLUMNS TO BE USED IN FOOTPRINT MERGE #####
        nsi['NSI_OC_Update'] = None  # This will contain updated Occupancy Class values throughout merge
        nsi['POINT_DropFlag'] = 0  # This indicates whether a row should be dropped from the final inventory. 1 indicates yes, 0 indicates no
        nsi['POINT_DropNote'] = ""  # Space for notes on the reason data points are dropped
        nsi['POINT_Source'] = 'NSI'  # This tracks the original data source for each row
        nsi['POINT_DataUpdate'] = ""  # Space for notes on steps throughout update

        out_path = intermediate_dir / f"{self.city_name}_NSI.json"
        # Save inventory
        fxns.gdf_to_json(nsi, out_path)
        print(f"Saved: {out_path.resolve()}")

        # EDU1
        if "EDU1" in hifld_paths:
            nsi, m = hf.augment_nsi_edu1(
                nsi=nsi,
                crs_main=self.crs_main,
                crs_plot=self.crs_plot,
                city_blocks=city_blocks,
                city_tracts=city_tracts,
                cb_id_name=cb_id_name,
                hifld_paths=hifld_paths["EDU1"],
                plot=plot,
            )
            if plot:
                hf.show_folium_map(m, self.maps_dir() / f"{self.city_name}_NSI_EDU1_Upgrade.html")
            out_path = intermediate_dir / f"{self.city_name}_NSI_EDU1_Upgrade.json"
            fxns.gdf_to_json(nsi, out_path)
            print(f"Saved: {out_path.resolve()}")

        # EDU2
        if "EDU2" in hifld_paths:
            nsi = hf.augment_nsi_edu2(
                nsi=nsi,
                crs_main=self.crs_main,
                crs_plot=self.crs_plot,
                city_blocks=city_blocks,
                city_tracts=city_tracts,
                city_bounds={'x': self.xbounds, 'y': self.ybounds},
                cb_id_name=cb_id_name,
                hifld_paths=hifld_paths["EDU2"],
                plot=plot,
            )
            out_path = intermediate_dir / f"{self.city_name}_NSI_EDU1_EDU2_Upgrade.json"
            fxns.gdf_to_json(nsi, out_path)
            print(f"Saved: {out_path.resolve()}")

        # GOV2
        if "GOV2" in hifld_paths:
            nsi, m = hf.augment_nsi_gov2(
                nsi=nsi,
                crs_main=self.crs_main,
                crs_plot=self.crs_plot,
                city_blocks=city_blocks,
                city_tracts=city_tracts,
                cb_id_name=cb_id_name,
                hifld_paths=hifld_paths["GOV2"],
                plot=plot,
            )
            if plot:
                hf.show_folium_map(m, self.maps_dir() / f"{self.city_name}_NSI_EDU1_EDU2_GOV2_Upgrade.html")
            out_path = intermediate_dir / f"{self.city_name}_NSI_EDU1_EDU2_GOV2_Upgrade.json"
            fxns.gdf_to_json(nsi, out_path)
            print(f"Saved: {out_path.resolve()}")

        # Finalize
        nsi = pre.add_nsi_tracking_columns(nsi, min_area_filter_ft2)
        nsi = pre.compute_min_mix_units(nsi)

        # Check that all columns are assigned to a category
        fxns.check_column_assignment(nsi, self.sum_columns, self.list_columns, self.excluded)

        nsi_length = len(nsi)
        nsi0 = nsi[nsi["POINT_DropFlag"] != 1]
        nsi1 = nsi[nsi["POINT_DropFlag"] == 1]
        nsi0 = pre.merge_duplicate_bid(nsi0, self.list_columns, self.sum_columns)
        nsi = pre.recombine_dropped_data(nsi0, nsi1, nsi_length)

        out_path = self.nsi_for_merge_path()
        fxns.gdf_to_json(nsi, out_path)
        print(f"Saved: {out_path.resolve()}")

    def process_census_download(
            self,
            *,
            years: YearInput,
            fill_holes: FillHolesInput,
            plot: bool = True,
            compare_years: CompareInput = None,
            compare_colors: ColorsInput = None,
    ) -> None:
        years_list = hf.as_years(years)
        results: Dict[int, Dict[str, gpd.GeoDataFrame]] = {}
        city_name = self.city_name

        for year in years_list:
            cfg = self.census_year_config(year)
            root = cfg["root_dir"]

            tracts_shp_path = cfg["tract_shp"]
            blocks_shp_path = cfg["block_shp"]
            places_shp_path = cfg["place_shp"]

            tracts_gdf = gpd.read_file(tracts_shp_path).to_crs(crs=f"EPSG:{int(self.crs_main)}")
            blocks_gdf = gpd.read_file(blocks_shp_path).to_crs(crs=f"EPSG:{int(self.crs_main)}")
            places_gdf = gpd.read_file(places_shp_path).to_crs(crs=f"EPSG:{int(self.crs_main)}")

            place_name_col = cfg["place_name_col"]
            city_place = places_gdf[places_gdf[place_name_col] == city_name].copy()
            if city_place.empty:
                raise ValueError(f"City '{city_name}' not found in {year} places data ({place_name_col}).")

            if plot:
                fig, ax = plt.subplots(1, 1, figsize=(5, 5))
                city_place.plot(ax=ax)
                ax.set_title(f"{city_name} Census Place ({year})")
                plt.show()

            if hf.fill_holes_for_year(year, fill_holes):
                city_place = pre.fill_census_place(city_place.copy())
                if plot:
                    fig, ax = plt.subplots(1, 1, figsize=(5, 5))
                    city_place.plot(ax=ax)
                    ax.set_title(f"{city_name} Census Place Filled ({year})")
                    plt.show()

            root.mkdir(parents=True, exist_ok=True)
            city_place.to_file(self.census_path(year), driver="GeoJSON")
            city_geom_only = city_place[["geometry"]].copy()

            censustracts_city, censusblocks_city = pre.find_city_tracts_and_blocks(
                tracts_gdf, blocks_gdf, city_geom_only, cfg["cb_id_name"]
            )

            if plot:
                fig, ax = plt.subplots(1, 2, figsize=(12, 12))
                censustracts_city.plot(ax=ax[0], color="tab:blue", alpha=0.5, edgecolor="k")
                city_geom_only.boundary.plot(ax=ax[0], color="red", linewidth=2, label=f"{city_name} City Boundary")
                censusblocks_city.plot(ax=ax[1], color="tab:blue", alpha=0.5, edgecolor="k")
                city_geom_only.boundary.plot(ax=ax[1], color="red", linewidth=2, label=f"{city_name} City Boundary")
                ax[0].legend(loc="upper right")
                ax[1].legend(loc="lower right")
                ax[0].set_title(f"Selected Census Tracts ({year})")
                ax[1].set_title(f"Selected Census Blocks ({year})")
                plt.show()

            censustracts_city.to_file(self.census_tracts_path(year), driver="GeoJSON")
            censusblocks_city.to_file(self.census_blocks_path(year), driver="GeoJSON")

            results[year] = {"city": city_geom_only, "tracts": censustracts_city, "blocks": censusblocks_city}

        if plot:
            years_to_compare = hf.resolve_compare_years(years_list, compare_years)

            if years_to_compare:
                if compare_years is True:
                    ordered_years = years_list
                else:
                    ordered_years = years_to_compare

                colors = hf.resolve_compare_colors(compare_colors, len(ordered_years))

                hf.compare_blocks_folium(
                    city=self,
                    years_to_compare=ordered_years,
                    compare_colors=colors,
                    crs_plot=self.crs_plot,
                    out_dir=self.maps_dir(),
                )

    def save_processed_footprint(
            self,
            *,
            min_area_ft2: float,
            census_year: int,
            overlap_limit: Union[float, str],
            plot: bool = True,
    ) -> None:
        footprints_path = self.raw_footprint_path()
        footprints = hf.load_footprints_for_processing(footprints_path=footprints_path, crs_main=self.crs_main)

        footprints, duplicate_geometries = hf.filter_footprints(
            footprints=footprints,
            min_area_ft2=min_area_ft2,
        )

        city_footprints, city_blocks, city_tracts = self.assign_census_to_footprints(
            footprints=footprints,
            census_year=census_year,
        )

        city_footprints, overlap = hf.find_overlaps(
            city_footprints=city_footprints,
            overlap_limit=overlap_limit,
        )

        if len(overlap):
            kept, dropped, ids_to_drop = hf.drop_overlaps(
                city_footprints=city_footprints,
                overlap=overlap,
            )
            city_footprints, updated_overlap = hf.find_overlaps(city_footprints=kept, overlap_limit=overlap_limit)
            assert len(updated_overlap) == 0
            if plot:
                hf.save_dropped_vs_kept_map(
                    kept=kept,
                    dropped=dropped,
                    overlap=overlap,
                    crs_plot=self.crs_plot,
                    overlap_limit=overlap_limit,
                    city_name=self.city_name,
                    out_dir=self.maps_dir(),
                )

        if plot:
            hf.plot_city_footprints(
                city_blocks=city_blocks,
                city_footprints=city_footprints,
                city_name=self.city_name,
            )

        out_path = self.processed_footprint_path()
        hf.finalize_and_save_city_footprints(
            city_footprints=city_footprints,
            out_path=out_path,
        )

        print("Saved:", out_path)

    def attribute_points_to_footprints(
            self,
            census_year: int,
            estimate_stories: bool,
            plot: bool = True,
    ) -> None:
        _, dir_intermediate, _ = self.setup_national_footprint_attribution_dirs()
        footprints = fxns.json_to_gdf(self.processed_footprint_path(), crs_main=self.crs_main)
        footprints = pt_ftpt.estimate_ftpt_size_for_merge(footprints.copy(), estimate_stories)

        city_blocks = gpd.read_file(self.census_blocks_path(year=census_year))
        city_tracts = gpd.read_file(self.census_tracts_path(year=census_year))
        city_blocks['CensusBlock'] = city_blocks[self.census_year_config(year=census_year)["cb_id_name"]]

        points = fxns.json_to_gdf(self.nsi_for_merge_path(), crs_main=self.crs_main)
        ##### DISPLAY NUMBER OF POINTS #####
        nsi_length = len(points)  # Used for tracking purposes
        print('NSI:', len(points))
        print('Footprints:', len(footprints))

        fxns.check_column_assignment(points, self.sum_columns, self.list_columns, self.excluded)
        if plot:
            hf.plot_footprints_and_points_folium(
                footprints=footprints, points=points, crs_plot=self.crs_plot,
                out_path=self.maps_dir() / f"{self.city_name}_footprints_and_nsi_points.html",
            )

        # MergeFlag1
        points = hf.mergeflag1(
            points=points,
            footprints=footprints,
            nsi_length=nsi_length,
            plot=plot,
            crs_plot=self.crs_plot,
            map1_path=self.maps_dir() / f"{self.city_name}_mergeflag1_overlapping.html",
            map2_path=self.maps_dir() / f"{self.city_name}_mergeflag1_remaining.html",
        )
        out_path = dir_intermediate / f"{self.city_name}_MergeFlag1.json"
        fxns.gdf_to_json(points.copy(), out_path)
        print(f"Saved: {out_path.resolve()}")

        # MergeFlag2
        points = hf.mergeflag2(
            points=points,
            footprints=footprints,
            nsi_length=nsi_length,
            list_columns=self.list_columns,
            sum_columns=self.sum_columns,
            plot=plot,
            crs_plot=self.crs_plot,
            map_remaining_path=self.maps_dir() / f"{self.city_name}_mergeflag2_remaining.html",
        )
        out_path = dir_intermediate / f"{self.city_name}_MergeFlag2.json"
        fxns.gdf_to_json(points.copy(), out_path)
        print(f"Saved: {out_path.resolve()}")

        points = hf.mergeflag2_drop(
            points=points,
            footprints=footprints,
            nsi_length=nsi_length,
            plot=plot,
            crs_plot=self.crs_plot,
            map_dropped_path=self.maps_dir() / f"{self.city_name}_mergeflag2_dropped.html",
        )
        out_path = dir_intermediate / f"{self.city_name}_MergeFlag2_dropped.json"
        fxns.gdf_to_json(points.copy(), out_path)
        print(f"Saved: {out_path.resolve()}")

        # MergeFlag3 (10m, 100m)
        for merge_flag in [310, 3100]:
            points = hf.mergeflag3(
                points=points,
                footprints=footprints,
                bounding_geometry=city_blocks,
                bounding_id_name="CensusBlock",
                nsi_length=nsi_length,
                list_columns=self.list_columns,
                sum_columns=self.sum_columns,
                plot=plot,
                crs_plot=self.crs_plot,
                map_detail_path=self.maps_dir() / f"{self.city_name}_mergeflag{merge_flag}_detail.html",
                merge_flag=merge_flag,
            )
            out_path = dir_intermediate / f"{self.city_name}_MergeFlag{merge_flag}.json"
            fxns.gdf_to_json(points.copy(), out_path)
            print(f"Saved: {out_path.resolve()}")

    def finalize_national_inventory(
        self,
        *,
        census_year: int,
        review_map: bool = True,
        plot: bool = True,
    ) -> None:
        _, dir_intermediate, _ = self.setup_national_footprint_attribution_dirs()

        footprints = fxns.json_to_gdf(self.processed_footprint_path(), crs_main=self.crs_main)
        points = fxns.json_to_gdf(
            dir_intermediate / f"{self.city_name}_MergeFlag3100.json", crs_main=self.crs_main
        )

        if review_map:
            city_blocks = gpd.read_file(self.census_blocks_path(year=census_year))
            hf.review_remaining_points_map(
                points=points,
                footprints=footprints,
                blocks=city_blocks,
                plot=plot,
                crs_plot=self.crs_plot,
                map_path=self.maps_dir() / f"{self.city_name}_mergeflag3100_remaining_review.html",
            )

        points = hf.drop_remaining_after_mergeflag3100(
            points=points, footprints=footprints, nsi_length=len(points)
        )
        fxns.gdf_to_json(
            points.copy(), dir_intermediate / f"{self.city_name}_National_MergedPoints.json"
        )

        ftpt_inv_poly, ftpt_inv_point = hf.build_national_inventory(
            points=points,
            footprints=footprints,
            plot=plot,
            crs_plot=self.crs_plot,
            xbounds=self.xbounds,
            ybounds=self.ybounds,
        )

        poly_path, point_path = self.national_inventory_paths()

        fxns.gdf_to_json(
            ftpt_inv_poly, poly_path,
        )
        print(f"Saved: {poly_path.resolve()}")

        fxns.gdf_to_json(
            ftpt_inv_point, point_path,
        )
        print(f"Saved: {point_path.resolve()}")

    def export_mh_points(
            self,
            *,
            census_year: int,
            mh_hifld_path: Path | str,
    ) -> tuple[gpd.GeoDataFrame, Path]:
        """
        Build a city-specific mobile-home helper CSV (with geometry_wkt) for manual polygon tracing.
        """
        mobile_all = gpd.read_file(mh_hifld_path).to_crs(crs=f"EPSG:{int(self.crs_main)}")

        city_blocks = gpd.read_file(self.census_blocks_path(census_year))
        city_tracts = gpd.read_file(self.census_tracts_path(census_year))
        cb_id_name = self.census_year_config(int(census_year))["cb_id_name"]

        mobile = pre.assign_census_hifld(mobile_all, city_blocks, city_tracts, cb_id_name)

        out_csv_path = self.mh_points_path()
        out_csv_path = Path(out_csv_path)
        out_csv_path.parent.mkdir(parents=True, exist_ok=True)

        mobile_plot = mobile.to_crs(crs=f"EPSG:{int(self.crs_plot)}").copy()
        mobile_plot["geometry_wkt"] = mobile_plot.geometry.apply(lambda g: g.wkt if g is not None else None)
        mobile_plot.drop(columns="geometry").to_csv(out_csv_path, index=False)

        print(f"Saved: {out_csv_path.resolve()}")
        return mobile, out_csv_path

    def generate_inventory_all_fields(
            self,
            *,
            census_api_key: str,
            reset_very_high_stories: str = "Mean_of_Occupancy_Class",
            plot: bool = True,
    ) -> None:
        dir_generation, _ = self.setup_national_inventory_generation_dirs()

        _, point_path = self.national_inventory_paths()

        inv_raw = fxns.json_to_gdf(
            point_path,
            self.crs_main,
        )
        inv_mod = inv_raw.copy()

        inv_mod = hf.resolve_within_source_disagreement(inv_mod)

        mh_polygons_path = self.mh_manual_polygons_path()

        if mh_polygons_path.exists():
            inv_mod = hf.apply_mobile_home_override(
                inv_mod=inv_mod,
                crs_main=self.crs_main,
                crs_plot=self.crs_plot,
                mh_polygons_path=mh_polygons_path,
                plot=plot,
                map_path=self.maps_dir() / f"{self.city_name}_mh_override.html",
            )

        blocks20 = gpd.read_file(self.census_blocks_path(year=2020))
        inv_mod = hf.assign_number_of_units_with_census_2020(
            inventory=inv_mod,
            census_blocks_2020=blocks20,
            crs_main=self.crs_main,
            crs_plot=self.crs_plot,
            xbounds=self.xbounds,
            ybounds=self.ybounds,
            census_api_key=census_api_key,
            state_fips=self.state_fips,
            county_fips=self.county_fips,
            plot=plot,
        )


        inv_mod = hf.compute_stories(
            inv_mod=inv_mod,
            stories_limit=self.stories_limit,
            reset_very_high_stories=reset_very_high_stories,
        )

        inv_mod = hf.compute_plan_area(
            inv_mod=inv_mod,
            crs_plot=self.crs_plot,
            plot=plot,
        )

        inv_mod = hf.compute_cost_fields(
            inv_mod=inv_mod,
            hazus_cost_path=self.hazus_cost_path(),
            crs_plot=self.crs_plot,
            plot=plot,
        )

        # Export Inventory
        fxns.gdf_to_json(inv_mod, self.inventory_all_fields_path())
        print("Saved:", self.inventory_all_fields_path().resolve())

    def prepare_imputation_csv(
            self
    ) -> None:
        data = fxns.json_to_gdf(self.inventory_all_fields_path(), self.crs_main)
        # Remove cases of footprints with no data
        data = data[data['National_Flag'] == 1]

        # Convert to format of R2D - keep missing data (for imputation purposes)
        for_imputation = data.copy().to_crs(self.crs_plot)
        for_imputation['Longitude'] = for_imputation['geometry'].x
        for_imputation['Latitude'] = for_imputation['geometry'].y

        # Separate required columns for imputation
        for_imputation = for_imputation[
            ['Latitude', 'Longitude', 'PlanArea_Best', 'Stories_Best', 'NSI_MedYearBuilt_Single',
             'ReplacementCost_Best', 'StructureValue_Best', 'OccupancyClass_Best', 'NSI_BuildingType_Single',
             'Units_Best', 'NSI_Population_Night', 'NSI_Population_Day', 'CensusBlock', 'CensusTract', 'FootprintID']]

        # Standardize columns for imputation and R2D
        for_imputation = for_imputation.rename(columns={
            'PlanArea_Best': 'PlanArea',
            'Stories_Best': 'NumberOfStories',
            'NSI_MedYearBuilt_Single': 'YearBuilt',
            'OccupancyClass_Best': 'OccupancyClass',
            'NSI_BuildingType_Single': 'BuildingType',
            'ReplacementCost_Best': 'ReplacementCost',
            'StructureValue_Best': 'StructureValue',
            'NSI_Population_Night': 'NightPopulation',
            'NSI_Population_Day': 'DayPopulation',
            'Units_Best': 'NumberOfUnits'})

        # Convert None for imputation types
        for_imputation['BuildingType'] = for_imputation['BuildingType'].replace('None', np.nan)

        # Add index
        for_imputation.insert(0, 'index', range(len(for_imputation)))

        # Export inventory
        for_imputation.to_csv(self.inventory_before_imputation_path(), index=False)
        print("Saved:", self.inventory_before_imputation_path().resolve())

    def impute_inventory_data(self) -> None:
        # Specify data for imputation
        file_path = self.inventory_before_imputation_path()

        # create an Import to get the classes
        importer = Importer()
        knn_imputer_class = importer.get_class("KnnImputer")

        # Load inventory
        inventory = AssetInventory()
        inventory.read_from_csv(str(file_path), keep_existing=True, id_column='index')

        #### IMPUTE DATA USING BRAILS ####
        imputer = knn_imputer_class(inventory, n_possible_worlds=1,
                                    exclude_features=['PlanArea', 'ReplacementCost', 'StructureValue', 'OccupancyClass',
                                                      'FootprintID', 'BuildingType'])
        new_inventory = imputer.impute()

        # Conver to pandas geodataframe
        inv_geoj = new_inventory.get_geojson()
        gdf = gpd.GeoDataFrame.from_features(inv_geoj["features"])

        # Correct data type for population
        gdf['NightPopulation'] = gdf['NightPopulation'].replace('', 0)
        gdf['DayPopulation'] = gdf['DayPopulation'].replace('', 0)

        ## SAVE IMPUTED INVENTORY
        fxns.gdf_to_json(gdf, self.imputed_inventory_path())
        print("Saved:", self.imputed_inventory_path().resolve())

    def infer_structure_type(self) -> None:
        data = fxns.json_to_gdf(self.imputed_inventory_path(), self.crs_main)
        # SET FLAGS
        use_bldg_type = False  # Use building type (material) to constrain list of possible structure types
        allow_mh_only_for_res2 = True  # Allows structure type 'MH' only when the occupancy class is RES2 (diverges from Hazus)
        no_urm = True  # Doesn't allow the assignment of URM buildings, due to efforts to retrofit those buildings (diverges from Hazus)
        res3ab_to_res1_flag = True  # Adopts structure types used for 'RES1' to be assigned for RES3A and RES3B. 2-4 unit structures are likely structurally more similar to single family homes than to large apartment buildings.

        # SET VARIABLE NAMES
        occ_key = 'OccupancyClass'
        nstory_key = 'NumberOfStories'
        year_key = 'YearBuilt'
        strtype_key = 'StructureType'
        bldgtype_key = 'BuildingType'
        n_pw = 1

        # CALL FUNCTION TO INFER STRUCTURE TYPE
        bldg_properties_df = resolve.infer_structure_type(data.copy(), self.state_name, occ_key, nstory_key, year_key,
                                                          bldgtype_key, strtype_key, n_pw, use_bldg_type,
                                                          allow_mh_only_for_res2, no_urm, res3ab_to_res1_flag)

        # EXPORT
        fxns.gdf_to_json(bldg_properties_df, self.imputed_and_inferred_inventory_path())
        print("Saved:", self.imputed_and_inferred_inventory_path().resolve())

    def export_inventory_for_r2d(self):
        bldg_properties_df = fxns.json_to_gdf(self.imputed_and_inferred_inventory_path(), self.crs_main)

        # Convert to format of R2D - remove missing data
        bldg_properties_df_nomissing = bldg_properties_df[
            ~((bldg_properties_df['StructureType'].isna()) | (bldg_properties_df['StructureType'] == 'na'))].copy()
        print(len(bldg_properties_df[((bldg_properties_df['StructureType'].isna()) | (
                    bldg_properties_df['StructureType'] == 'na'))].copy()),
              'points dropped due to missing structure type')

        # Create appropriate columns
        r2d = bldg_properties_df_nomissing.copy()
        r2d['Longitude'] = r2d['geometry'].x
        r2d['Latitude'] = r2d['geometry'].y
        r2d = r2d[
            ['Latitude', 'Longitude', 'PlanArea', 'NumberOfStories', 'YearBuilt', 'ReplacementCost', 'StructureValue',
             'StructureType', 'BuildingType', 'OccupancyClass_clean', 'OccupancyClass', 'NumberOfUnits',
             'NightPopulation', 'DayPopulation', 'CensusBlock', 'CensusTract', 'FootprintID', 'geometry']]

        # Assign design level and height class (used in regional analysis)
        r2d = resolve.find_design_level(r2d, 'StructureType', 'YearBuilt', 'DesignLevel')
        r2d = resolve.find_height_class(r2d, 'StructureType', 'NumberOfStories', 'HeightClass')

        # Add id
        r2d.insert(0, 'id', range(len(r2d)))

        # Rename occupancy class columns for R2D use
        r2d = r2d.rename(columns={'OccupancyClass': 'OccupancyClass_Actual',
                                  'OccupancyClass_clean': 'OccupancyClass'})

        r2d_inventory_csv_path, r2d_inventory_json_path, _ = self.r2d_inventory_paths()

        # Save inventory
        r2d.to_csv(r2d_inventory_csv_path, index=False)
        fxns.gdf_to_json(r2d, r2d_inventory_json_path)

        # Randomly sample for R2D test run
        # sampled_df = r2d.sample(n=50, random_state=1, replace=False)
        # sampled_df = sampled_df.drop(columns='id')
        # sampled_df.insert(0, 'id', range(len(sampled_df)))
        # sampled_df.to_csv(r2d_inventory_sampled_csv_path, index=False)
        print("Saved:", r2d_inventory_csv_path.resolve())
        print("Saved:", r2d_inventory_json_path.resolve())
        # print("Saved:", r2d_inventory_sampled_csv_path.resolve())


san_francisco = City(
    city_name="San Francisco",
    state_name="California",
    state_abbrev="CA",
    state_fips="06",
    county_name="San Francisco",
    county_fips="075",
    xbounds=(-122.52, -122.35),
    ybounds=(37.70, 37.83),
    stories_limit=61,
)
los_angeles = City(
    city_name="Los Angeles",
    state_name="California",
    state_abbrev="CA",
    state_fips="06",
    county_name="Los Angeles",
    county_fips="037",
    xbounds=(-118.67, -118.15),
    ybounds=(33.70, 34.34),
    stories_limit=73,
)
salt_lake_city = City(
    city_name="Salt Lake City",
    state_name="Utah",
    state_abbrev="UT",
    state_fips="49",
    county_name="Salt Lake",
    county_fips="035",
    xbounds=(-112.10, -111.80),
    ybounds=(40.68, 40.86),
    stories_limit=40,
)
seattle = City(
    city_name="Seattle",
    state_name="Washington",
    state_abbrev="WA",
    state_fips="53",
    county_name="King",
    county_fips="033",
    xbounds=(-122.46, -122.22),
    ybounds=(47.49, 47.74),
    stories_limit=76,
)
memphis = City(
    city_name="Memphis",
    state_name="Tennessee",
    state_abbrev="TN",
    state_fips="47",
    county_name="Shelby",
    county_fips="157",
    xbounds=(-90.10, -89.75),
    ybounds=(34.98, 35.25),
    stories_limit=37,
)
washington = City(
    city_name="Washington",
    state_name="District of Columbia",
    state_abbrev="DC",
    state_fips="11",
    county_name="District of Columbia",
    county_fips="001",
    xbounds=(-77.12, -76.91),
    ybounds=(38.80, 38.995),
    stories_limit=20,
)

__all__ = [
    "City",
    "san_francisco",
    "los_angeles",
    "salt_lake_city",
    "seattle",
    "memphis",
    "washington"
]