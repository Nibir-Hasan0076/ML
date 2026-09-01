"""Bangladesh administrative-boundary loading + name normalisation.

Downloads / reads geoBoundaries simplified GeoJSON for Bangladesh (ADM1
division, ADM2 district, ADM3 upazila) from data/geo/ and maps the messy
DISTRICT / DIVISION / UPZMUNCC values in the measles surveillance file onto
the official shape names.
"""
import os

import geopandas as gpd

GEO_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                       "data", "geo")

# ---------------------------------------------------------------------------
# geoBoundaries ADM2 district name aliases (our messy values -> official)
# ---------------------------------------------------------------------------
DISTRICT_ALIAS = {
    "DHAKA NORTH": "DHAKA", "DHAKA NORTH CC.": "DHAKA",
    "DHAKA SOUTH": "DHAKA", "DHAKA SOUTH CC.": "DHAKA",
    "BRAHMANBARIA": "BRAHAMANBARIA",
    "JHALAKATI": "JHALOKATI",
    "KHAGRACHARI": "KHAGRACHHARI",
    "MOULVI BAZAR": "MAULVIBAZAR",
    "NARSINGDHI": "NARSINGDI",
    "NETROKONA": "NETRAKONA",
    "NOAGOAN": "NAOGAON",
    "NOWABGANJ": "NAWABGANJ",
    "PANCHAGHARH": "PANCHAGARH",
    "PEROJPUR": "PIROJPUR",
    "SARIATPUR": "SHARIATPUR",
    "THAKURGOAN": "THAKURGAON",
}

DIVISION_ALIAS = {
    "RAJSHAHI": "RAJSHANI",  # geoBoundaries ADM1 typo
}


_TITLE_FIX = {
    "Cox'S Bazar": "Cox's Bazar",
}


def norm_district(s):
    s = str(s).strip().upper().replace(" CC.", "").replace(" CC", "")
    s = DISTRICT_ALIAS.get(s, s)
    s = s.title()  # Title Case to match geoBoundaries shapeName
    return _TITLE_FIX.get(s, s)


def norm_division(s):
    s = str(s).strip().upper()
    return DIVISION_ALIAS.get(s, s)


def load_boundaries(level):
    """level in {ADM1, ADM2, ADM3} -> GeoDataFrame with official shapeName."""
    fname = {
        "ADM1": "geoBoundaries-BGD-ADM1_simplified.geojson",
        "ADM2": "geoBoundaries-BGD-ADM2_simplified.geojson",
        "ADM3": "geoBoundaries-BGD-ADM3_simplified.geojson",
    }[level]
    path = os.path.join(GEO_DIR, fname)
    return gpd.read_file(path)


def upazila_to_district_map():
    """upazila shapeName -> containing district shapeName (via ADM3->ADM2)."""
    adm3 = load_boundaries("ADM3")
    adm2 = load_boundaries("ADM2")
    adm3p = adm3.to_crs(epsg=3106)
    adm2p = adm2.to_crs(epsg=3106)
    cent = gpd.GeoDataFrame(
        {c: adm3p[c] for c in adm3p.columns if c != "geometry"},
        geometry=adm3p.geometry.centroid, crs=adm3p.crs)
    joined = gpd.sjoin(cent, adm2p[["shapeName", "geometry"]],
                       how="left", predicate="within")
    joined = joined.rename(
        columns={"shapeName_left": "upazila", "shapeName_right": "district"})
    return joined[["upazila", "district"]]