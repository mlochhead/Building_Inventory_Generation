from __future__ import annotations
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from typing import Dict, Sequence, Tuple, Union
    from inventory_generation_city.city import City
    YearInput = Union[int, Sequence[int]]
    FillHolesInput = Union[bool, Dict[int, bool]]
    CompareInput = Union[bool, int, Sequence[int], None]
    ColorsInput = Union[str, Sequence[str], None]

from pathlib import Path
import folium
import webbrowser
import geopandas as gpd
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import inventory_generation_functions.functions_preprocessing as pre
import inventory_generation_functions.functions_general as fxns
import inventory_generation_functions.functions_point_to_ftpt as pt_ftpt
import inventory_generation_functions.functions_disagreement_and_gaps as resolve
import matplotlib.patches as mpatches
import matplotlib.lines as mlines
from folium.plugins import FastMarkerCluster


def as_years(years: YearInput) -> list[int]:
    if isinstance(years, (list, tuple, set)):
        yrs = [int(y) for y in years]
    else:
        yrs = [int(years)]
    bad = [y for y in yrs if y not in (2010, 2020)]
    if bad:
        raise ValueError(f"Unsupported year(s): {bad}. Supported: 2010, 2020.")
    return sorted(set(yrs))


def fill_holes_for_year(year: int, fill_holes: FillHolesInput) -> bool:
    if isinstance(fill_holes, dict):
        return bool(fill_holes.get(year, False))
    return bool(fill_holes)


def resolve_compare_years(years_list: list[int], compare_years: CompareInput) -> list[int]:
    if not compare_years:
        return []
    if compare_years is True:
        return years_list
    if isinstance(compare_years, int):
        return [int(compare_years)]
    req = as_years(compare_years)
    missing = [y for y in req if y not in years_list]
    if missing:
        raise ValueError(f"compare_years includes year(s) not in years: {missing}. years={years_list}")
    return req


def resolve_compare_colors(compare_colors: ColorsInput, n: int) -> list[str]:
    if isinstance(compare_colors, str):
        if n != 1:
            raise ValueError(f"compare_colors is a single str but {n} years are being compared.")
        return [compare_colors]
    if compare_colors is None:
        raise ValueError("compare_colors must be provided when compare_years is enabled.")
    colors = list(compare_colors)
    if len(colors) != n:
        raise ValueError(f"compare_colors must have length {n}, got {len(colors)}")
    return colors


def compare_blocks_folium(
    *,
    city: City,
    years_to_compare: list[int],
    compare_colors: Sequence[str],
    crs_plot: int | str,
    out_dir: str | Path,
) -> folium.Map:
    out_dir_p = Path(out_dir)
    out_dir_p.mkdir(parents=True, exist_ok=True)
    if len(compare_colors) != len(years_to_compare):
        raise ValueError(
            f"compare_colors must have length {len(years_to_compare)} to match years_to_compare={years_to_compare}, "
            f"got {len(compare_colors)}"
        )

    blocks_by_year: Dict[int, gpd.GeoDataFrame] = {}
    for y in years_to_compare:
        blocks_by_year[y] = gpd.read_file(city.census_blocks_path(y))

    first = years_to_compare[0]
    c = blocks_by_year[first].to_crs(crs=f"EPSG:{int(crs_plot)}").geometry.iloc[0].centroid
    m = folium.Map(location=[c.y, c.x], zoom_start=12)

    for y, color in zip(years_to_compare, compare_colors):
        folium.GeoJson(
            blocks_by_year[y].to_crs(crs=f"EPSG:{int(crs_plot)}"),
            name=f"Blocks {y}",
            style_function=lambda _feat, c=color: {"color": c},
        ).add_to(m)

    folium.LayerControl().add_to(m)
    tag = "_".join(str(y) for y in years_to_compare)
    html_path = out_dir_p / f"{city.city_name}_census_blocks_compare_{tag}.html"
    show_folium_map(m, html_path)
    return m


def load_footprints_for_processing(
    *,
    footprints_path: str | Path,
    crs_main,
) -> gpd.GeoDataFrame:
    fp_path = Path(footprints_path)
    footprints = gpd.read_file(fp_path)

    if "BuildingHeight" in footprints.columns:
        footprints["BuildingHeight"] = pd.to_numeric(footprints["BuildingHeight"], errors="coerce")
    elif "buildingheight" in footprints.columns:
        footprints["buildingheight"] = pd.to_numeric(footprints["buildingheight"], errors="coerce")
        footprints = footprints.rename(columns={"buildingheight": "BuildingHeight"})
    else:
        raise KeyError("Expected 'BuildingHeight' or 'buildingheight' in footprint GeoJSON properties.")

    footprints = footprints.to_crs(crs=f"EPSG:{crs_main}")
    return footprints[["BuildingHeight", "geometry"]].copy()


def filter_footprints(
    *,
    footprints: gpd.GeoDataFrame,
    min_area_ft2: float = 450.0,
) -> tuple[gpd.GeoDataFrame, gpd.GeoDataFrame]:
    gdf = footprints.copy()
    gdf["area_ft2"] = gdf.geometry.area * 10.7639

    footprints_450 = gdf[gdf["area_ft2"] >= float(min_area_ft2)].copy()
    print(len(gdf) - len(footprints_450), "footprints dropped due to size limit")
    print(len(footprints_450), "footprints remaining")

    duplicate_geometries = footprints_450[footprints_450.duplicated(subset="geometry")].copy()
    footprints_450 = footprints_450.drop_duplicates(subset="geometry", keep="first").copy()
    return footprints_450, duplicate_geometries


def plot_city_footprints(
    *,
    city_blocks: gpd.GeoDataFrame,
    city_footprints: gpd.GeoDataFrame,
    city_name: str,
    all_footprints: gpd.GeoDataFrame | None = None,
) -> None:
    if all_footprints is None:
        fig, ax = plt.subplots(1, 1, figsize=(12, 12))
        city_blocks.plot(ax=ax)
        city_footprints.plot(ax=ax, color="black", markersize=0.1)
        ax.set_title(f"Filtered for {city_name} Blocks")
        plt.show()
        return

    fig, ax = plt.subplots(1, 2, figsize=(12, 12))
    city_blocks.plot(ax=ax[0])
    all_footprints.plot(ax=ax[0], color="black", markersize=0.1)
    ax[0].set_title("All Footprints")

    city_blocks.plot(ax=ax[1])
    city_footprints.plot(ax=ax[1], color="black", markersize=0.1)
    ax[1].set_title(f"Filtered for {city_name} Blocks")
    plt.show()


def plot_example_tract(
    *,
    city_tracts: gpd.GeoDataFrame,
    city_footprints: gpd.GeoDataFrame,
    cb_id_name: str,
    tract_index: int = 4,
) -> None:
    if cb_id_name not in city_tracts.columns:
        return
    if len(city_tracts) <= tract_index:
        return
    if "CensusTract" not in city_footprints.columns:
        return

    tract = city_tracts[cb_id_name].iloc[tract_index]

    fig, ax = plt.subplots(1, 1, figsize=(8, 8))
    city_tracts.boundary.plot(ax=ax, color="black", linewidth=1)
    city_footprints.plot(ax=ax, color="black")
    city_tracts[city_tracts[cb_id_name] == tract].plot(ax=ax, color="#c0d6fa")
    city_footprints[city_footprints["CensusTract"] == tract].plot(ax=ax, color="#0050d4")

    xmin, ymin, xmax, ymax = city_footprints.total_bounds
    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymin, ymax)
    ax.set_title("Footprints and Tract Highlighted for Specified Tract")
    plt.show()


def find_overlaps(
    *,
    city_footprints: gpd.GeoDataFrame,
    overlap_limit: float | str,
) -> tuple[gpd.GeoDataFrame, gpd.GeoDataFrame]:
    gdf = city_footprints.copy()
    gdf["Overlap_ID"] = range(len(gdf))
    overlap = pre.find_overlapping_ftpt(gdf.copy(), overlap_limit)

    if len(overlap) == 0:
        gdf = gdf.drop(columns="Overlap_ID")
        print("No Significantly Overlapping Footprints Found")

    return gdf, overlap


def drop_overlaps(
    *,
    city_footprints: gpd.GeoDataFrame,
    overlap: gpd.GeoDataFrame,
) -> tuple[gpd.GeoDataFrame, gpd.GeoDataFrame, list[int]]:
    ids_to_drop = pre.find_overlaps_to_drop(city_footprints, overlap)
    dropped = city_footprints[city_footprints["Overlap_ID"].isin(ids_to_drop)].copy()
    kept = city_footprints[~city_footprints["Overlap_ID"].isin(ids_to_drop)].copy()
    return kept, dropped, ids_to_drop


def save_dropped_vs_kept_map(
    *,
    kept: gpd.GeoDataFrame,
    dropped: gpd.GeoDataFrame,
    overlap: gpd.GeoDataFrame,
    crs_plot,
    overlap_limit: float | str,
    city_name: str,
    out_dir: str | Path,
    zoom_start: int = 15,
) -> Path:
    out_dir_p = Path(out_dir)
    out_dir_p.mkdir(parents=True, exist_ok=True)

    tag = str(overlap_limit).replace(".", "p")
    out_path = out_dir_p / f"dropped_vs_kept_overlap_{tag}_footprints_{city_name}.html"

    overlap_plot = overlap.to_crs(crs=f"EPSG:{int(crs_plot)}")
    c = overlap_plot.geometry.iloc[0].centroid

    m = folium.Map(location=[c.y, c.x], zoom_start=zoom_start)
    folium.GeoJson(kept.to_crs(crs=f"EPSG:{int(crs_plot)}"), style_function=lambda _f: {"color": "blue"}).add_to(m)
    folium.GeoJson(dropped.to_crs(crs=f"EPSG:{int(crs_plot)}"), style_function=lambda _f: {"color": "red"}).add_to(m)
    show_folium_map(m, out_path)
    return out_path


def save_overlap_map(
    *,
    overlap: gpd.GeoDataFrame,
    crs_plot,
    overlap_limit: float | str,
    city_name: str,
    out_dir: str | Path,
) -> Path:
    out_dir_p = Path(out_dir)
    out_dir_p.mkdir(parents=True, exist_ok=True)
    out_path = out_dir_p / f"overlap_{str(overlap_limit).replace('.', 'p')}_footprints_{city_name}.html"

    overlap_plot = overlap.to_crs(crs=f"EPSG:{int(crs_plot)}")
    c = overlap_plot.geometry.iloc[0].centroid
    m = folium.Map(location=[c.y, c.x], zoom_start=12)
    folium.GeoJson(overlap_plot, style_function=lambda _f: {"color": "red"}).add_to(m)
    show_folium_map(m, out_path)
    return out_path


def finalize_and_save_city_footprints(
    *,
    city_footprints: gpd.GeoDataFrame,
    out_path: str | Path,
) -> tuple[gpd.GeoDataFrame, Path]:
    gdf = city_footprints.rename(
        columns={"area_ft2": "FootprintArea", "BuildingHeight": "FootprintHeight"}
    ).copy()

    gdf["FootprintArea"] = pd.to_numeric(gdf["FootprintArea"], errors="coerce")
    gdf["FootprintHeight"] = pd.to_numeric(gdf["FootprintHeight"], errors="coerce")

    gdf["FootprintID"] = range(len(gdf))

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    fxns.gdf_to_json(gdf, str(out_path))

    return gdf, out_path


def show_folium_map(m: folium.Map, p: Path) -> None:
    if not isinstance(m, folium.Map):
        if isinstance(m, str):
            print(m)
        return
    m.save(str(p))
    webbrowser.open(p.resolve().as_uri())


def augment_nsi_edu1(
        *,
        nsi: gpd.GeoDataFrame,
        crs_main: Union[str, int],
        crs_plot: Union[str, int],
        city_blocks: gpd.GeoDataFrame,
        city_tracts: gpd.GeoDataFrame,
        cb_id_name: str,
        hifld_paths: dict,
        plot: bool,
) -> Tuple[gpd.GeoDataFrame, Union[folium.Map, str]]:
    public = gpd.read_file(hifld_paths["public_schools_path"]).to_crs(crs=f"EPSG:{int(crs_main)}")
    private = gpd.read_file(hifld_paths["private_schools_path"]).to_crs(crs=f"EPSG:{int(crs_main)}")

    public_city, private_city, school_import = pre.format_and_locate_edu1(
        public, private, city_tracts, city_blocks, cb_id_name
    )

    if plot:
        fig, ax = plt.subplots(1, 3, figsize=(15, 12))
        city_blocks.plot(ax=ax[0], alpha=0.2)
        public_city.plot(ax=ax[0], color="green", markersize=1)
        ax[0].set_title("Public Schools")
        city_blocks.plot(ax=ax[1], alpha=0.2)
        private_city.plot(ax=ax[1], color="red", markersize=1)
        ax[1].set_title("Private Schools")
        city_blocks.plot(ax=ax[2], alpha=0.2)
        school_import.plot(ax=ax[2], color="black", markersize=1)
        ax[2].set_title("All Schools")
        plt.show()

    school_import = school_import.drop(columns=['NAME'])
    nsi, m = pre.synthesize_edu1_and_HIFLD(
        nsi, school_import, crs_plot, plot_flag=plot, drop_unpaired_nsi_edu1=True, drop_gov1_near_edu1=True
    )
    return nsi, m

def augment_nsi_edu2(
        *,
        nsi: gpd.GeoDataFrame,
        crs_main: Union[str, int],
        crs_plot: Union[str, int],
        city_blocks: gpd.GeoDataFrame,
        city_tracts: gpd.GeoDataFrame,
        city_bounds: dict[str, Tuple[float, float]],
        cb_id_name: str,
        hifld_paths: dict,
        plot: bool,
) -> gpd.GeoDataFrame:
    univ = gpd.read_file(hifld_paths["univ_campuses_path"]).to_crs(crs=f"EPSG:{int(crs_main)}")
    univ_pts = gpd.read_file(hifld_paths["univ_points_path"]).to_crs(crs=f"EPSG:{int(crs_main)}")

    univ_city, univ_pts_city = pre.locate_edu2(
        univ, univ_pts, city_tracts, city_blocks, cb_id_name
    )

    if plot:
        fig, ax = plt.subplots(1, 1, figsize=(5, 5))
        city_blocks.plot(ax=ax, alpha=0.2)
        univ_city.boundary.plot(ax=ax, color="green")
        univ_pts_city.plot(ax=ax, color="red", markersize=10)
        univ_line = mlines.Line2D([], [], color="green", label="Campus Polygons")
        univ_pts_line = mlines.Line2D([], [], color="red", marker="o", linestyle="None", markersize=4, label="College/University HIFLD Points")
        ax.legend(handles=[univ_line, univ_pts_line], loc="lower right")
        plt.show()

    # HIFLD EDU2 vs NSI EDU2 plot
    edu2 = nsi[nsi["NSI_OccupancyClass"] == "EDU2"]
    if plot:
        if (not edu2.empty) or (not univ_pts_city.empty) or (not univ_city.empty):
            fig, ax = plt.subplots(figsize=(7, 7))

            if len(edu2) > 0:
                edu2.to_crs(crs_plot).plot(ax=ax, color="tab:orange", markersize=10)
            else:
                print("No EDU2 points in NSI data")

            if len(univ_pts_city) > 0:
                univ_pts_city.to_crs(crs_plot).plot(ax=ax, color="tab:blue", markersize=10)

            if len(univ_city) > 0:
                univ_city.to_crs(crs_plot).boundary.plot(ax=ax, color="black")

            ax.set_xlim(city_bounds['x'])
            ax.set_ylim(city_bounds['y'])

            handles = [
                mpatches.Patch(color="tab:blue", label="HIFLD: Campus Points"),
                mpatches.Patch(color="tab:orange", label="NSI: EDU2 Points"),
                mlines.Line2D([], [], color="black", label="HIFLD Campus Polygons"),
            ]
            ax.legend(handles=handles)
            plt.title("HIFLD UNIVERSITY VS NSI EDU2 DATA")
            plt.show()

    edu2_no_campus = pre.prepare_pts_without_campuses(univ_pts_city, univ_city, nsi.copy())
    nsi = pd.concat([nsi, edu2_no_campus])

    edu2_no_gov1 = pre.prepare_pts_without_gov1(univ_pts_city, univ_city, nsi.copy())
    nsi = pd.concat([nsi, edu2_no_gov1])

    nsi = pre.merge_pts_with_campuses(univ_city, nsi.copy(), scale_edu2_pop=True)

    # RUN CHECK: Following numbers should match (within scaling/rounding error)
    for school_num in range(len(univ_city)):
        one_univ = univ_city.iloc[[school_num]]
        one_univ_convert = nsi.sjoin(
            one_univ[["UNIQUEID", "POPULATION", "TOT_ENROLL", "TOT_EMP", "geometry"]],
            how="inner",
        )
        print("\nNSI Day", sum(one_univ_convert["NSI_Population_Day"]))
        print("Imported School Population", one_univ["POPULATION"].values)

    return nsi

def augment_nsi_gov2(
        *,
        nsi: gpd.GeoDataFrame,
        crs_main: Union[str, int],
        crs_plot: Union[str, int],
        city_blocks: gpd.GeoDataFrame,
        city_tracts: gpd.GeoDataFrame,
        cb_id_name: str,
        hifld_paths: dict,
        plot: bool,
) -> Tuple[gpd.GeoDataFrame, Union[folium.Map, str]]:
    fire = gpd.read_file(hifld_paths["fire_path"]).to_crs(crs=f"EPSG:{int(crs_main)}")
    police = gpd.read_file(hifld_paths["police_path"]).to_crs(crs=f"EPSG:{int(crs_main)}")
    local_eoc = gpd.read_file(hifld_paths["local_eoc_path"]).to_crs(crs=f"EPSG:{int(crs_main)}")
    state_eoc = gpd.read_file(hifld_paths["state_eoc_path"]).to_crs(crs=f"EPSG:{int(crs_main)}")

    fire = pre.assign_census_hifld(fire, city_blocks, city_tracts, cb_id_name)
    police = pre.assign_census_hifld(police, city_blocks, city_tracts, cb_id_name)
    local_eoc = pre.assign_census_hifld(local_eoc, city_blocks, city_tracts, cb_id_name)
    state_eoc = pre.assign_census_hifld(state_eoc, city_blocks, city_tracts, cb_id_name)

    fire = fire[["geometry", "CensusBlock", "CensusTract"]]
    fire["NSI_OccupancyClass"] = "GOV2-FIRE"
    police = police[["geometry", "CensusBlock", "CensusTract"]]
    police["NSI_OccupancyClass"] = "GOV2-POLICE"
    local_eoc = local_eoc[["geometry", "CensusBlock", "CensusTract"]]
    local_eoc["NSI_OccupancyClass"] = "GOV2-OPERATIONS"
    state_eoc = state_eoc[["geometry", "CensusBlock", "CensusTract"]]
    state_eoc["NSI_OccupancyClass"] = "GOV2-OPERATIONS"

    gov2_import = pd.concat([fire, police, local_eoc, state_eoc], ignore_index=True, sort=False)

    nsi, m = pre.synthesize_gov2_and_HIFLD(
        nsi, gov2_import, crs_plot, plot_flag=plot, drop_unpaired_nsi_gov2=True, drop_gov1_near_gov2=True
    )
    return nsi, m


def plot_footprints_and_points_folium(
    *,
    footprints: gpd.GeoDataFrame,
    points: gpd.GeoDataFrame,
    crs_plot: int | str,
    out_path: Path | str,
    zoom_start: int = 12,
    footprint_color: str = "gray",
    point_color: str = "blue",
    point_radius: int = 1,
    cluster_points: bool = True,   # new
) -> folium.Map:
    if footprints.empty:
        raise ValueError("footprints is empty")

    ftpt_plot = footprints.copy().to_crs(crs=f"EPSG:{int(crs_plot)}")
    pt_plot = points.copy().to_crs(crs=f"EPSG:{int(crs_plot)}")

    c = ftpt_plot.geometry.iloc[0].centroid
    m = folium.Map(location=[c.y, c.x], zoom_start=zoom_start)
    folium.GeoJson(ftpt_plot, style_function=lambda _f: {"color": footprint_color}).add_to(m)

    if cluster_points:
        coords = list(zip(pt_plot.geometry.y, pt_plot.geometry.x))
        FastMarkerCluster(coords).add_to(m)
    else:
        for _, row in pt_plot.iterrows():
            folium.CircleMarker(
                location=[row.geometry.y, row.geometry.x],
                radius=point_radius,
                color=point_color,
                fill=True,
                fill_color=point_color,
            ).add_to(m)

    show_folium_map(m, Path(out_path))
    return m


def mergeflag1(
        *,
        points: gpd.GeoDataFrame,
        footprints: gpd.GeoDataFrame,
        nsi_length: int,
        plot: bool,
        crs_plot: int | str,
        map1_path: Path | None = None,
        map2_path: Path | None = None,
) -> gpd.GeoDataFrame:
    points0 = points[points["POINT_DropFlag"] != 1]
    points1 = points[points["POINT_DropFlag"] == 1]

    points0, m1 = pt_ftpt.merge_intersecting(points0, footprints, crs_plot, plot)
    points0 = pt_ftpt.update_mergeflag99(points0, footprints, mergeflag=1)

    points = pt_ftpt.recombine_dropped_data(points0, points1, nsi_length)
    pt_ftpt.check_post_merge_duplicates(points.copy())

    show_folium_map(m1, map1_path)

    remaining = points[(points["POINT_DropFlag"] != 1) & (points["POINT_FootprintID"].isna())]

    if plot and map2_path is not None:
        plot_footprints_and_points_folium(
            footprints=footprints,
            points=remaining,
            crs_plot=crs_plot,
            out_path=map2_path,
            footprint_color="gray",
            point_color="blue",
        )

    return points


def mergeflag2(
    *,
    points: gpd.GeoDataFrame,
    footprints: gpd.GeoDataFrame,
    nsi_length: int,
    list_columns: list[str],
    sum_columns: list[str],
    plot: bool,
    crs_plot: int | str,
    map_remaining_path: Path | None = None,
    print_odd_occupancy_pairings: bool = False,
    use_size_limit: bool = True,
    use_nsi_occupancy_merge: bool = True,
) -> gpd.GeoDataFrame:
    manually_assigned_occupancy = pd.DataFrame({"FootprintID": [], "POINT_OccupancyClass": []})
    ids_to_drop: list = []
    points = pt_ftpt.drop_ids(points, ids_to_drop, "Manually dropped due to occupancy class incompatibility")

    points0 = points[points["POINT_DropFlag"] != 1]
    points1 = points[points["POINT_DropFlag"] == 1]

    points0 = pt_ftpt.address_overlapping_points(
        points0.copy(),
        footprints.copy(),
        list_columns,
        sum_columns,
        manually_assigned_occupancy,
        use_size_limit,
        use_nsi_occupancy_merge,
        print_odd_occupancy_pairings,
        crs_plot,
    )

    points = pt_ftpt.recombine_dropped_data(points0, points1, nsi_length)
    points["POINT_FootprintID"] = points["POINT_FootprintID"].astype("Int64")
    pt_ftpt.check_post_merge_duplicates(points.copy())

    remaining = points[(points["POINT_DropFlag"] != 1) & (points["POINT_FootprintID"].isna())]
    print(len(remaining))

    if plot and map_remaining_path is not None:
        plot_footprints_and_points_folium(
            footprints=footprints,
            points=remaining,
            crs_plot=crs_plot,
            out_path=map_remaining_path,
            footprint_color="gray",
            point_color="blue",
        )

    return points


def mergeflag2_drop(
    *,
    points: gpd.GeoDataFrame,
    footprints: gpd.GeoDataFrame,
    nsi_length: int,
    plot: bool,
    crs_plot: int | str,
    map_dropped_path: Path | None = None,
    drop_classes: tuple[str, ...] = ("IND4", "IND5", "GOV1"),
) -> gpd.GeoDataFrame:
    points0 = points[points["POINT_DropFlag"] != 1]
    points1 = points[points["POINT_DropFlag"] == 1]

    remaining_points, remaining_ftpt = pt_ftpt.find_remaining(
        points0, footprints, "POINT_FootprintID", "POINT_MergeFlag"
    )
    points_to_drop = remaining_points[remaining_points["NSI_OccupancyClass"].isin(drop_classes)]
    ids_to_drop = [
        item
        for sublist in points_to_drop["POINT_ID"]
        for item in (sublist if isinstance(sublist, list) else [sublist])
    ]
    print(len(ids_to_drop), "points to drop")

    points0 = pt_ftpt.drop_ids(points0, ids_to_drop, "IND4, IND5, or GOV1 point outside of footprint")
    points = pt_ftpt.recombine_dropped_data(points0, points1, nsi_length)
    points["POINT_FootprintID"] = points["POINT_FootprintID"].astype("Int64")
    pt_ftpt.check_post_merge_duplicates(points.copy())

    if plot and map_dropped_path is not None:
        plot_footprints_and_points_folium(
            footprints=remaining_ftpt,
            points=points_to_drop,
            crs_plot=crs_plot,
            out_path=map_dropped_path,
            footprint_color="gray",
            point_color="red",
        )

    return points


def mergeflag3(
    *,
    points: gpd.GeoDataFrame,
    footprints: gpd.GeoDataFrame,
    bounding_geometry: gpd.GeoDataFrame,
    bounding_id_name: str,
    nsi_length: int,
    list_columns: list[str],
    sum_columns: list[str],
    plot: bool,
    crs_plot: int | str,
    merge_flag: int,
    map_detail_path: Path | None = None,
    print_odd_occupancy_pairings: bool = False,
    use_size_limit: bool = True,
    use_nsi_occupancy_merge: bool = True,
) -> gpd.GeoDataFrame:
    if merge_flag not in (310, 3100):
        raise ValueError("merge_flag must be 310 or 3100")

    distance_limit = float(str(merge_flag)[1:])  # 310 -> 10, 3100 -> 100
    use_surrounding_bgs = (merge_flag == 310)

    # Manual overrides
    manually_assigned_occupancy = pd.DataFrame({"FootprintID": [], "POINT_OccupancyClass": []})
    ids_to_drop: list = []
    points = pt_ftpt.drop_ids(
        points, ids_to_drop, "Manually dropped due to occupancy class incompatibility"
    )

    # For detail map: remaining footprints before this merge
    pre_merge_points = points[points["POINT_DropFlag"] == 0].copy()
    _, remaining_ftpt = pt_ftpt.find_remaining(
        pre_merge_points, footprints, "POINT_FootprintID", "POINT_MergeFlag"
    )

    # Split
    points0 = points[points["POINT_DropFlag"] != 1].copy()
    points1 = points[points["POINT_DropFlag"] == 1].copy()

    # Merge
    points0 = pt_ftpt.distance_limit_merge(
        bounding_geometry[bounding_id_name].unique(),
        points0.copy(),
        footprints,
        bounding_id_name,
        manually_assigned_occupancy,
        list_columns,
        sum_columns,
        bounding_geometry.copy(),
        crs_plot,
        distance_limit=distance_limit,
        use_surrounding_bgs=use_surrounding_bgs,
        prioritize_empty_footprints=True,
        prioritize_partial_footprints=True,
        use_full_footprints=True,
        merge_flag=merge_flag,
        use_size_limit=use_size_limit,
        use_nsi_occupancy_merge=use_nsi_occupancy_merge,
        print_odd_occupancy_pairings=print_odd_occupancy_pairings,
    )

    points = pt_ftpt.recombine_dropped_data(points0, points1, nsi_length)
    points["POINT_FootprintID"] = points["POINT_FootprintID"].astype("Int64")
    pt_ftpt.check_post_merge_duplicates(points.copy())

    # Detail map (remaining footprints + newly merged points)
    merged_points = points[points["POINT_MergeFlag"] == merge_flag]

    if plot and map_detail_path is not None:
        plot_footprints_and_points_folium(
            footprints=remaining_ftpt,
            points=merged_points,
            crs_plot=crs_plot,
            out_path=map_detail_path,
            footprint_color="blue",
            point_color="red",
        )

    return points


def review_remaining_points_map(
    *,
    points: gpd.GeoDataFrame,
    footprints: gpd.GeoDataFrame,
    blocks: gpd.GeoDataFrame,
    plot: bool,
    crs_plot: int | str,
    map_path: Path | None = None,
) -> tuple[gpd.GeoDataFrame, gpd.GeoDataFrame, gpd.GeoDataFrame]:
    # Split by drop flag
    points0 = points[points["POINT_DropFlag"] != 1].copy()

    # Find footprints associated with partially full points
    not_full_nsi = points0[points0["POINT_MergeFlag"] == 99]
    not_full_ftpt = footprints[
        footprints["FootprintID"].isin(not_full_nsi["POINT_FootprintID"])
    ]

    # Remaining points + footprints
    points0["POINT_FootprintID"] = points0["POINT_FootprintID"].apply(
        lambda x: int(x) if pd.notna(x) else x
    )
    remaining_points, remaining_ftpt = pt_ftpt.find_remaining(
        points0, footprints, "POINT_FootprintID", "POINT_MergeFlag"
    )

    # Display remaining population
    total_pop_night = remaining_points["NSI_Population_Night"].dropna().sum()
    total_pop_day = remaining_points["NSI_Population_Day"].dropna().sum()
    total_replacement = remaining_points["NSI_ReplacementCost"].dropna().sum()
    print("Night Population in Remaining Points:", total_pop_night)
    print("Day Population in Remaining Points:", total_pop_day)
    print("Replacement Cost in Remaining Points:", total_replacement)

    if footprints.empty:
        return remaining_points, remaining_ftpt, not_full_ftpt

    if plot and map_path is not None:
        ftpt_plot = footprints.copy().to_crs(crs_plot)
        blocks_plot = blocks.copy().to_crs(crs_plot)
        remaining_ftpt_plot = remaining_ftpt.copy().to_crs(crs_plot)
        not_full_ftpt_plot = not_full_ftpt.copy().to_crs(crs_plot)
        rem_plot = remaining_points.copy().to_crs(crs_plot)

        c = ftpt_plot.geometry.iloc[0].centroid
        m = folium.Map(location=[c.y, c.x], zoom_start=12)

        # Census blocks in gray
        if not blocks_plot.empty:
            folium.GeoJson(
                blocks_plot, style_function=lambda _f: {"color": "gray"}
            ).add_to(m)

        # Remaining + not-full footprints
        if not remaining_ftpt_plot.empty:
            folium.GeoJson(
                remaining_ftpt_plot, style_function=lambda _f: {"color": "blue"}
            ).add_to(m)

        if not not_full_ftpt_plot.empty:
            folium.GeoJson(
                not_full_ftpt_plot, style_function=lambda _f: {"color": "green"}
            ).add_to(m)

        # Paired points in gray (all points that *do* have footprints)
        paired_points = points0[points0["POINT_FootprintID"].notna()].copy()
        for _, row in paired_points.to_crs(crs_plot).iterrows():
            folium.CircleMarker(
                location=[row.geometry.y, row.geometry.x],
                radius=1,
                color="gray",
                fill=True,
                fill_color="gray",
                popup=row[["NSI_OccupancyClass", "POINT_ID", "POINT_FootprintID"]],
            ).add_to(m)

        # Remaining points in red
        for _, row in rem_plot.iterrows():
            folium.CircleMarker(
                location=[row.geometry.y, row.geometry.x],
                radius=1,
                color="red",
                fill=True,
                fill_color="red",
                popup=row[["NSI_OccupancyClass", "POINT_ID", "POINT_FootprintID", "CensusBlock"]],
            ).add_to(m)

        show_folium_map(m, map_path)
    return remaining_points, remaining_ftpt, not_full_ftpt


def drop_remaining_after_mergeflag3100(
    *,
    points: gpd.GeoDataFrame,
    footprints: gpd.GeoDataFrame,
    nsi_length: int,
) -> gpd.GeoDataFrame:
    points0 = points[points["POINT_DropFlag"] != 1].copy()
    points1 = points[points["POINT_DropFlag"] == 1].copy()

    points0["POINT_FootprintID"] = points0["POINT_FootprintID"].apply(
        lambda x: int(x) if pd.notna(x) else x
    )
    remaining_points, _ = pt_ftpt.find_remaining(
        points0, footprints, "POINT_FootprintID", "POINT_MergeFlag"
    )

    points0 = pt_ftpt.drop_ids(
        points0,
        remaining_points["POINT_ID"].values,
        "Points remaining after MergeFlag 3100",
    )

    points = pt_ftpt.recombine_dropped_data(points0, points1, nsi_length)
    points["POINT_FootprintID"] = points["POINT_FootprintID"].astype("Int64")
    pt_ftpt.check_post_merge_duplicates(points.copy())
    return points


def build_national_inventory(
    *,
    points: gpd.GeoDataFrame,
    footprints: gpd.GeoDataFrame,
    plot: bool,
    crs_plot: int | str,
    xbounds: tuple[float, float],
    ybounds: tuple[float, float],
) -> tuple[gpd.GeoDataFrame, gpd.GeoDataFrame]:
    points0 = points[points["POINT_DropFlag"] != 1].copy()

    points0 = points0.drop(columns=["geometry"])
    points0 = points0.drop(
        columns=[
            "CensusTract",
            "CensusBlock",
            "POINT_DropFlag",
            "POINT_DropNote",
            "DistanceToFtpt",
            "ClosestFtpt_ID",
            "POINT_ID",
            "NSI_OccupancyClass",
            "POINT_DataUpdate",
        ],
        errors="ignore",
    )
    points0 = points0.rename(columns={"NSI_OC_Update": "NSI_OccupancyClass"})
    points0 = points0.rename(columns={"POINT_ID_List": "POINT_ID"})

    float_columns = [
        "NSI_ContentValue",
        "NSI_StructureValue",
        "NSI_ReplacementCost",
        "NSI_PopOver65_Night",
        "NSI_PopUnder65_Night",
        "NSI_Population_Night",
        "NSI_PopOver65_Day",
        "NSI_PopUnder65_Day",
        "NSI_Population_Day",
    ]
    int_columns = [
        "POINT_NumPoints",
        "POINT_FootprintID",
        "POINT_MergeFlag",
        "NSI_MinResUnits",
        "NSI_MaxResUnits",
    ]

    for col in float_columns:
        if col in points0.columns:
            points0[col] = points0[col].astype(float)

    for col in int_columns:
        if col in points0.columns:
            points0[col] = points0[col].astype(int)

    points0["NSI_OccupancyClass"] = points0["NSI_OccupancyClass"].apply(
        pt_ftpt.convert_to_list
    )
    points0["POINT_ID"] = points0["POINT_ID"].apply(pt_ftpt.convert_to_list)

    ftpt_inv = footprints.copy()
    orig_inv_length = len(ftpt_inv)
    points0["National_Flag"] = 1

    ftpt_inv = ftpt_inv.merge(
        points0, left_on="FootprintID", right_on="POINT_FootprintID", how="left"
    )
    if len(ftpt_inv) != orig_inv_length:
        raise ValueError("Footprints Dropped - Step 1")

    if len(ftpt_inv[ftpt_inv["National_Flag"] == 1]) != len(points0):
        raise ValueError("Footprints Dropped - Step 2")

    ftpt_inv = ftpt_inv.drop(columns=["POINT_FootprintID"])
    ftpt_inv["National_Flag"] = ftpt_inv["National_Flag"].fillna(0)

    if plot:
        fig, ax = plt.subplots(figsize=(5, 5))
        ftpt_inv.to_crs(crs_plot).plot(
            ax=ax, color="tab:blue", edgecolor="tab:blue", alpha=0.5, label="All Footprints"
        )
        plt.title("Static Map of Footprint Inventory")
        plt.xlabel("Longitude")
        plt.ylabel("Latitude")
        plt.xlim(xbounds)
        plt.ylim(ybounds)
        plt.show()

    ftpt_inv_point = ftpt_inv.copy()
    ftpt_inv_point = ftpt_inv_point.rename(columns={"geometry": "ftpt_geometry"})
    ftpt_inv_point["geometry"] = ftpt_inv_point["ftpt_geometry"].centroid
    ftpt_inv_point = ftpt_inv_point.set_geometry("geometry")
    ftpt_inv_point["Footprint_Flag"] = 1

    if plot:
        fig, ax = plt.subplots(figsize=(5, 5))
        ftpt_inv_point.to_crs(crs_plot).plot(
            ax=ax,
            color="tab:blue",
            edgecolor="tab:blue",
            markersize=0.1,
            alpha=0.5,
            label="All Footprints",
        )
        plt.title("Static Map of Point Inventory")
        plt.xlabel("Longitude")
        plt.ylabel("Latitude")
        plt.xlim(xbounds)
        plt.ylim(ybounds)
        plt.show()

    return ftpt_inv, ftpt_inv_point


def load_census_api_key(path: Path | str) -> str:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Census API key file not found: {p}")
    return p.read_text(encoding="utf-8").strip()


def plot_source_map(
    *,
    inv_mod: gpd.GeoDataFrame,
    source_col: str,
    color_map: dict[str, str],
    crs_plot: int | str,
    title: str,
) -> None:
    inv_mod2 = inv_mod[inv_mod["National_Flag"] != 0].copy()
    if inv_mod2.empty:
        return
    inv_mod2["color"] = inv_mod2[source_col].map(color_map)
    inv_mod2["color"] = inv_mod2["color"].fillna("red")

    fig, ax = plt.subplots(figsize=(10, 8))
    inv_mod2.to_crs(crs_plot).plot(ax=ax, color=inv_mod2["color"], markersize=0.1)
    for value, color in color_map.items():
        ax.scatter([], [], color=color, label=value)
    ax.legend(title=title)
    plt.show()


def resolve_within_source_disagreement(
    inv_mod: gpd.GeoDataFrame,
) -> gpd.GeoDataFrame:
    modified_to_single_solo = [
        "NSI_BuildingType",
        "NSI_MedYearBuilt",
        "NSI_NumberOfStories",
        "NSI_TotalAreaSqFt",
    ]
    for col in modified_to_single_solo:
        inv_mod[col + "_Single"] = inv_mod[col].apply(resolve.modify_to_single_val)

    inv_mod[["NSI_FoundationType_Single", "NSI_FoundationHeight_Single"]] = (
        inv_mod[["NSI_FoundationType", "NSI_FoundationHeight"]]
        .apply(resolve.modify_to_single_val_paired, axis=1)
        .apply(pd.Series)
    )

    inv_mod[
        ["NSI_OccupancyClass_Single", "NSI_OccupancyClass_MixedUse"]
    ] = inv_mod["NSI_OccupancyClass"].apply(resolve.modify_to_single_nsi_occupancy).apply(pd.Series)
    inv_mod["NSI_OccupancyClass_Single"] = inv_mod["NSI_OccupancyClass_Single"].fillna("")
    inv_mod.loc[
        inv_mod["NSI_OccupancyClass_Single"].str.contains("RES1", na=False),
        "NSI_OccupancyClass_Single",
    ] = "RES1"
    return inv_mod


def apply_mobile_home_override(
    *,
    inv_mod: gpd.GeoDataFrame,
    crs_main: int | str,
    crs_plot: int | str,
    mh_polygons_path: Path | str,
    plot: bool,
    map_path: Path | None = None,
) -> gpd.GeoDataFrame:
    inv_mod = inv_mod.copy()
    mh_polygons_path = Path(mh_polygons_path)

    ### USE MANUALLY GENERATED POLYGONS TO FORCE RES2 OCCUPANCY
    mh_polygons = gpd.read_file(mh_polygons_path)
    mh_polygons.set_crs(epsg=4326, inplace=True)
    mh_polygons.to_crs(epsg=int(crs_main), inplace=True)

    # Find inventory points that are within RES2 polygons and reset
    # This converts footprints, even that do not currently contain NSI data. Flag is updated to reflect there is now data from a national source
    in_polygons = gpd.sjoin(inv_mod, mh_polygons)
    inv_mod.loc[in_polygons.index, "NSI_OccupancyClass_Single"] = "RES2"
    inv_mod.loc[in_polygons.index, "National_Flag"] = 1

    # Since all data is already integrated into NSI_OccupancyClass_Single, set to be equal to OccupancyClass_Best
    inv_mod["OccupancyClass_Best"] = inv_mod["NSI_OccupancyClass_Single"]

    # Modify so RES2 are all single unit, single story structures (assumed based on spot checking google maps)
    res2 = inv_mod["OccupancyClass_Best"].str.contains("RES2", na=False)
    inv_mod.loc[res2, "NSI_MinResUnits"] = 1
    inv_mod.loc[res2, "NSI_MaxResUnits"] = 1
    inv_mod.loc[res2, "NSI_NumberOfStories_Single"] = 1

    if plot and not mh_polygons.empty and map_path is not None:
        plot_footprints_and_points_folium(
            footprints=mh_polygons,
            points=inv_mod.loc[in_polygons.index],
            crs_plot=crs_plot,
            out_path=Path(map_path),
            zoom_start=12,
            footprint_color="gray",
            point_color="red",
            point_radius=1,
            cluster_points=False,
        )

    return inv_mod

def assign_number_of_units_with_census_2020(
    *,
    inventory: gpd.GeoDataFrame,
    census_blocks_2020: gpd.GeoDataFrame,
    crs_main: int | str,
    crs_plot: int | str,
    xbounds: tuple[float, float] | None,
    ybounds: tuple[float, float] | None,
    census_api_key: str,
    state_fips: str,
    county_fips: str,
    plot: bool = False,
) -> gpd.GeoDataFrame:
    inventory = inventory.copy()
    inv_length = len(inventory)

    # Find corresponding 2020 Census Block
    inventory = inventory.sjoin(census_blocks_2020[['GEOID20', 'geometry']], how='left')
    inventory = inventory.rename(columns={'GEOID20': 'CensusBlock_2020'})

    # Filter inventory for footprints with NSI points
    inventory0 = inventory[inventory['National_Flag'] == 0]
    inventory = inventory[inventory['National_Flag'] == 1]

    # Filter inventory for footprints that are outside vs inside 2020 census blocks
    inventory_outside2020blocks = inventory[inventory['CensusBlock_2020'].isna()].copy()

    if plot:
        # Plot points
        fig, ax = plt.subplots()
        census_blocks_2020.plot(ax=ax, color='pink')
        inventory_outside2020blocks.plot(ax=ax, markersize=5)
        ax.set_title(f'Footprints Outside 2020 Census Blocks \n {len(inventory_outside2020blocks)} Points Reassigned to Nearest Census Block')
        plt.show()

    # Assign footprints outside of census block to nearest census block
    inventory_outside2020blocks['Nearest'] = inventory_outside2020blocks['geometry'].apply(
        lambda point: resolve.outside_ftpt_nearest_cb(point, census_blocks_2020.to_crs(crs_main)[['GEOID20', 'geometry']]))
    inventory.loc[inventory_outside2020blocks.index, 'CensusBlock_2020'] = inventory_outside2020blocks['Nearest'].values

    ### DOWNLOAD POPULATION AND NUMBER OF UNITS FOR 2020 CENSUS BLOCKS
    cbs = resolve.download_census_data(census_api_key, census_blocks_2020, state_fips, county_fips)

    if plot:
        ## PLOT TO MAKE SURE DATA DOWNLOADED FOR CORRECT CENSUS BLOCKS
        merged = cbs.merge(census_blocks_2020, left_on='cb_code', right_on='GEOID20')
        census_gdf = gpd.GeoDataFrame(merged, geometry=census_blocks_2020.geometry, crs=census_blocks_2020.crs)
        fig, ax = plt.subplots(1, 2, figsize=(15, 5), sharex=True, sharey=True)
        census_gdf.to_crs(crs_plot).plot(ax=ax[0], column='POP', cmap='viridis', legend=True, vmin=0, vmax=400)
        census_gdf.to_crs(crs_plot).plot(ax=ax[1], column='UNITS', cmap='viridis', legend=True, vmin=0, vmax=200)
        ax[0].set_title('POPULATION')
        ax[1].set_title('UNITS')
        ax[0].set_xlim(xbounds)
        ax[0].set_ylim(ybounds)
        plt.show()

    ### ESTIMATE THE NUMBER OF UNITS USING CENSUS INFORMATION AND WITH POPULATION SCALING ###

    # Pre-set modification flag to be 0
    inventory['Flag_ModifiedByCensus'] = 0
    inventory0['Flag_ModifiedByCensus'] = 0

    # Assign number of units using information from census block
    inventory2 = resolve.assign_units_from_censusblock(inventory.copy(), 'CensusBlock_2020', cbs)

    # Re-combine updated inventory with footprints with no NSI points
    inventory = resolve.recombine_dropped_data(inventory2, inventory0, inv_length)

    # Drop appropriate columns
    inventory = inventory.drop(columns=['index_right', 'CensusBlock_2020'])
    inv_mod = inventory.copy()

    # Assign best value for number of units
    inv_mod['Units_Best'] = inv_mod.apply(
        lambda row: (row['Units_CensusEstimate'] if pd.notna(row['Units_CensusEstimate'])
                     else (row['NSI_MinResUnits'] if pd.notna(row['NSI_MinResUnits'])
                           else np.nan)), axis=1)

    inv_mod['Units_Best_Source'] = inv_mod.apply(
        lambda row: ('Units_CensusEstimate' if pd.notna(row['Units_CensusEstimate'])
                     else ('NSI_MinResUnits' if pd.notna(row['NSI_MinResUnits'])
                           else 'None')), axis=1)

    # Modulate RES3 Occupancy based on updated number of units
    inv_mod['OccupancyClass_Best'] = inv_mod.apply(resolve.update_res_occ, axis=1)

    if plot:
        color_map = {
            "NSI_MinResUnits": "#d2edc7",
            "Units_CensusEstimate": "purple",
            "NSI_MeanUnits": "red",
            "None": "black",
        }
        plot_source_map(
            inv_mod=inv_mod,
            source_col="Units_Best_Source",
            color_map=color_map,
            crs_plot=crs_plot,
            title="Units Source",
        )
    return inv_mod


def compute_stories(
    *,
    inv_mod: gpd.GeoDataFrame,
    stories_limit: int,
    reset_very_high_stories: str,
) -> gpd.GeoDataFrame:
    # Select which method to use to modify the very tall structures using "reset_very_high_stories"
    # Selecting "Mean_of_Occupancy_Class" will reset the number of stories in the footprints exceeding stories_limit to be the mean number of stories for the given occupancy class
    # Selecting "Scale_from_Units" will set the number of stories by scaling from the best estimate of the number of units. The average unit is assumed to be 1000 square feet, so estimated stories are computed as (units * 1000) / (footprint area)
    # Selecting "No_Reset" does not modify the number of stories relative to the original NSI data

    if reset_very_high_stories == "Mean_of_Occupancy_Class":
        inv_mod["Stories_Best"] = resolve.reset_very_high_stories_to_mean(inv_mod, stories_limit)
    elif reset_very_high_stories == "Scale_from_Units":
        inv_mod["Stories_Best"] = resolve.reset_very_high_stories_by_units(inv_mod, stories_limit)
    elif reset_very_high_stories == "No_Reset":
        inv_mod["Stories_Best"] = inv_mod["NSI_NumberOfStories_Single"]
    else:
        raise ValueError("reset_very_high_stories must be Mean_of_Occupancy_Class, Scale_from_Units, or No_Reset")
    return inv_mod


def compute_plan_area(
    *,
    inv_mod: gpd.GeoDataFrame,
    crs_plot: int | str,
    plot: bool,
):
    inv_mod["PlanArea_Best"] = inv_mod.apply(
        lambda row: (
            (row["NSI_TotalAreaSqFt_Single"] / row["Stories_Best"])
            if pd.notna(row["NSI_TotalAreaSqFt_Single"]) and pd.notna(row["Stories_Best"])
            else (row["FootprintArea"] if pd.notna(row["FootprintArea"]) else np.nan)
        ),
        axis=1,
    )

    inv_mod["PlanArea_Best_Source"] = inv_mod.apply(
        lambda row: "NSI_TotalAreaSqFt_Over_Stories"
        if pd.notna(row["NSI_TotalAreaSqFt_Single"]) and pd.notna(row["Stories_Best"])
        else ("FootprintArea" if pd.notna(row["FootprintArea"]) else "None"),
        axis=1,
    )
    if plot:
        color_map = {
            "NSI_TotalAreaSqFt_Over_Stories": "#d2edc7",
            "FootprintArea": "purple",
            "None": "red",
        }
        plot_source_map(
            inv_mod=inv_mod,
            source_col="PlanArea_Best_Source",
            color_map=color_map,
            crs_plot=crs_plot,
            title="Plan Area Source",
        )

    return inv_mod


def compute_cost_fields(
    *,
    inv_mod: gpd.GeoDataFrame,
    hazus_cost_path: Path | str,
    crs_plot: int | str,
    plot: bool,
) -> gpd.GeoDataFrame:
    """
    Combined equivalent of:
      1) compute_replacement_costs(...)
      2) compute_structure_value(...)

    Notes:
    - NSI has both replacement and structure fields.
    - Hazus provides ReplacementCost_Hazus; in structure pass this is used as the
      Hazus fallback proxy exactly like the original code.
    """
    specs = [
        {
            "include_scaling_for_contents": True,
            "nsi_col": "NSI_ReplacementCost",
            "best_col": "ReplacementCost_Best",
            "source_col": "ReplacementCost_Best_Source",
            "nsi_label": "NSI_ReplacementCost",
            "map_title": "Replacement Cost Source",
            "hazus_hist_all": "Hazus Replacement Cost - All Values",
            "hazus_hist_hi": "Hazus Replacement Cost - Values > $10M",
            "nsi_hist_all": "NSI Replacement Cost - All Values",
            "nsi_hist_hi": "NSI Replacement Cost - Values > $10M",
            "x_label": "NSI_ReplacementCost",
            "y_label": "ReplacementCost_Hazus",
        },
        {
            "include_scaling_for_contents": False,
            "nsi_col": "NSI_StructureValue",
            "best_col": "StructureValue_Best",
            "source_col": "StructureValue_Best_Source",
            "nsi_label": "NSI_StructureValue",
            "map_title": "Structure Value Source",
            "hazus_hist_all": "Hazus Replacement Cost - All Values",
            "hazus_hist_hi": "Hazus Replacement Cost - Values > $10M",
            "nsi_hist_all": "NSI Structure Value - All Values",
            "nsi_hist_hi": "NSI Structure Value - Values > $10M",
            "x_label": "NSI_StructureValue",
            "y_label": "ReplacementCost_Hazus",
        },
    ]

    out = inv_mod.copy()
    hazus_conversion = pd.read_csv(hazus_cost_path)

    for s in specs:
        # Match original structure function behavior (drop prior Hazus temp col before second pass)
        out = out.drop(columns="ReplacementCost_Hazus", errors="ignore")

        out = resolve.compute_hazus_replacement_cost(
            out.copy(),
            hazus_conversion,
            s["include_scaling_for_contents"],
        )

        # Remove NSI zeros
        out.loc[out[s["nsi_col"]] == 0, s["nsi_col"]] = np.nan

        # Best value selection
        out[s["best_col"]] = out.apply(
            lambda row: row[s["nsi_col"]]
            if pd.notna(row[s["nsi_col"]])
            else (row["ReplacementCost_Hazus"] if pd.notna(row["ReplacementCost_Hazus"]) else np.nan),
            axis=1,
        )
        out[s["source_col"]] = out.apply(
            lambda row: s["nsi_label"]
            if pd.notna(row[s["nsi_col"]])
            else ("Hazus" if pd.notna(row["ReplacementCost_Hazus"]) else "None"),
            axis=1,
        )

        if plot:
            # Hazus histogram
            fig, ax = plt.subplots(1, 2, figsize=(10, 5))
            out["ReplacementCost_Hazus"].plot.hist(ax=ax[0], bins=100)
            out[out["ReplacementCost_Hazus"] > 1e7]["ReplacementCost_Hazus"].plot.hist(ax=ax[1], bins=100)
            ax[0].set_title(s["hazus_hist_all"])
            ax[1].set_title(s["hazus_hist_hi"])

            # NSI histogram
            fig, ax = plt.subplots(1, 2, figsize=(10, 5))
            out[s["nsi_col"]].plot.hist(ax=ax[0], bins=100)
            out[out[s["nsi_col"]] > 1e7][s["nsi_col"]].plot.hist(ax=ax[1], bins=100)
            ax[0].set_title(s["nsi_hist_all"])
            ax[1].set_title(s["nsi_hist_hi"])

            # Source map
            plot_source_map(
                inv_mod=out,
                source_col=s["source_col"],
                color_map={s["nsi_label"]: "#d2edc7", "Hazus": "purple", "None": "red"},
                crs_plot=crs_plot,
                title=s["map_title"],
            )

            # Scatter diagnostics
            valid = out[[s["nsi_col"], "ReplacementCost_Hazus"]].dropna()
            if not valid.empty:
                max_val = float(valid.max().max())
                min_val = float(valid.min().min())

                for bound in [max_val, 7e6, 1e6]:
                    alpha = 1 if bound == max_val else 0.01
                    plt.scatter(out[s["nsi_col"]], out["ReplacementCost_Hazus"], alpha=alpha)
                    plt.plot([min_val, max_val], [min_val, max_val], color="red", linestyle="--", label="x = y")
                    if bound != max_val:
                        plt.xlim([0, bound])
                        plt.ylim([0, bound])
                    plt.xlabel(s["x_label"])
                    plt.ylabel(s["y_label"])
                    plt.show()

    out.drop(columns="ReplacementCost_Hazus", errors="ignore", inplace=True)
    return out
