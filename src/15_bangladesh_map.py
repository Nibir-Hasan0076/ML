"""Step 15 - Bangladesh choropleth maps of measles burden & outbreak risk.

Reads the saved artefacts (raw case file, outbreak cell features) and draws:

  Panel A  Affected area      - confirmed measles case counts per district
                               (whole surveillance file, 2026).
  Panel B  Case-level risk    - mean predicted measles probability from the
                               case-level filter applied to confirmed/highly
                               suspect cases per district (raw case counts of
                               confirmed measles is used directly; risk flags
                               districts with >75th percentile burden).
  Panel C  Outbreak risk      - per-district outbreak risk = fraction of an
                               upazila's weeks flagged as outbreak cells by the
                               Step-14 upazila ensemble, aggregated to district.

Deliverables:
  figures/bangladesh_map.png        single 1x3 figure
  figures/bangladesh_burden_map.png / _outbreak_risk_map.png  individual maps
  output/outbreak/district_risk_map.csv   district-level table used
"""
import json
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import geopandas as gpd
from matplotlib.lines import Line2D

import pipeline as P
import bd_map

GEO = bd_map
OUT_DIR = os.path.join(P.OUT, "outbreak")
FIG_DIR = P.FIG

# ---------------------------------------------------------------------------
# Data assembly
# ---------------------------------------------------------------------------
def per_district_burden(df):
    """Confirmed measles cases per normalized district."""
    m = df[df["CLASS"] == P.MEASLES_CLASS].copy()
    m["nrm"] = m["DISTRICT"].map(GEO.norm_district)
    cnt = m.groupby("nrm").size().rename("measles_cases")
    return cnt


def per_district_outbreak_risk():
    """Fraction of upazila-weeks flagged outbreak -> district risk.

    Uses the saved upazila cell labels (OUTBREAK) from Step 14. For each
    upazila, risk = SUM(OUTBREAK)/N_weeks. Upazilas are mapped to districts
    through the geoBoundaries ADM3 -> ADM2 spatial join.
    """
    cells = pd.read_csv(os.path.join(OUT_DIR, "cell_UPZMUNCC.csv"))
    upz_dist = GEO.upazila_to_district_map()   # official names
    adb3 = upz_dist.set_index("upazila")["district"]

    # normalize our upazila -> official ADM3 name
    def nmu(s):
        return GEO.norm_district(s).title()
    nm = {}
    import re
    for u in cells["UPZMUNCC"].unique():
        cand = str(u).strip().upper()
        cand = (cand.replace(" MUN.", "").replace(" MUN", "")
                .replace(" CC.", "").replace(" CC", ""))
        cand = cand.replace("-", " ").replace("_", " ")
        found = None
        for c in [cand, cand.title()]:
            if c in adb3.index:
                # duplicated names -> disambiguate via our district suffix
                vals = adb3.loc[c]
                if isinstance(vals, pd.Series):
                    # pick the district hinted by our value, else first
                    hint = str(u).upper()
                    for d in vals.unique():
                        if d.upper().replace(" ", "") in hint.replace(" ", ""):
                            found = d
                            break
                    if found is None:
                        found = vals.iloc[0]
                else:
                    found = vals
                break
        if found is None:
            nm[u] = None
        else:
            nm[u] = found

    cells["district"] = cells["UPZMUNCC"].map(nm)
    cells = cells.dropna(subset=["district"])
    grp = cells.groupby("district").agg(
        n_weeks=("OUTBREAK", "size"),
        outbreak_weeks=("OUTBREAK", "sum"),
        total_cases=("cases", "sum"),
    )
    grp["outbreak_risk"] = grp["outbreak_weeks"] / grp["n_weeks"]
    return grp


# ---------------------------------------------------------------------------
# Plotting helpers
# ---------------------------------------------------------------------------
def assign_colors(values, cmap="YlOrRd", vmin=None, vmax=None, log=False):
    values = np.asarray(values, dtype=float)
    if vmin is None:
        vmin = np.nanmin(values) if values.size else 0
    if vmax is None:
        vmax = np.nanmax(values) if values.size else 1
    norm = mcolors.LogNorm(vmin=max(vmin, 1e-6), vmax=vmax) if log \
        else mcolors.Normalize(vmin=vmin, vmax=vmax)
    cm = plt.get_cmap(cmap)
    colors = [cm(norm(v)) if np.isfinite(v) else "#d0d0d0" for v in values]
    return colors, norm, cm


def plot_map(ax, gdf, value_col, title, cmap="YlOrRd", vmin=None, vmax=None,
             log=False, label="", edge="#666666"):
    valid = gdf.dropna(subset=[value_col])
    vals = valid[value_col].values
    colors, norm, cm = assign_colors(vals, cmap, vmin, vmax, log)
    gdf.plot(ax=ax, color="#e8e8e8", edgecolor=edge, linewidth=0.35)
    if len(valid):
        valid.plot(ax=ax, color=colors, edgecolor=edge, linewidth=0.35)
    ax.axis("off")
    ax.set_title(title, fontsize=11)
    # colorbar
    sm = plt.cm.ScalarMappable(cmap=cm, norm=norm)
    sm.set_array([])
    cax = ax.inset_axes([0.02, 0.06, 0.04, 0.45])
    cb = plt.colorbar(sm, cax=cax)
    cb.set_label(label, fontsize=8)
    cb.ax.tick_params(labelsize=7)
    return ax


def main():
    df = P.load_raw()

    adm2 = GEO.load_boundaries("ADM2")
    adm1 = GEO.load_boundaries("ADM1")

    # merge district burden
    burden = per_district_burden(df)
    adm2 = adm2.merge(burden, left_on="shapeName", right_index=True,
                      how="left")

    # merge district outbreak risk
    risk = per_district_outbreak_risk()
    adm2 = adm2.merge(risk, left_on="shapeName", right_index=True, how="left")

    # save district-level table
    tbl = adm2.drop(columns="geometry").copy()
    tbl.to_csv(os.path.join(OUT_DIR, "district_risk_map.csv"), index=False)

    # ---------------- Panel A: burden (affected areas) ----------------
    fig, axes = plt.subplots(1, 3, figsize=(18, 6.2))
    ax = axes[0]
    vmax = burden.max()
    plot_map(ax, adm2, "measles_cases",
             "A. Affected areas\n(confirmed measles cases, 2026)",
             cmap="YlOrRd", vmin=0, vmax=vmax, log=True,
             label="confirmed cases")

    # ---------------- Panel B: case-level risk ----------------
    ax = axes[1]
    # binary high/very-high burden flag relative to the 75th percentile
    q75 = adm2["measles_cases"].quantile(0.75)
    adm2["risk_flag"] = (adm2["measles_cases"] >= q75).astype(int)
    plot_map(ax, adm2, "risk_flag",
             "B. High risk districts\n(> 75th percentile case burden)",
             cmap="Reds", vmin=0, vmax=1, label="high risk (1=yes)")

    # ---------------- Panel C: outbreak risk ----------------
    ax = axes[2]
    plot_map(ax, adm2, "outbreak_risk",
             "C. Outbreak risk\n(Step-14 upazila ensemble, per district)",
             cmap="YlGnBu", vmin=0, vmax=1, label="outbreak risk")

    fig.suptitle("Bangladesh - Measles burden & outbreak risk (2026)",
                 fontsize=14, y=0.98)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, "bangladesh_map.png"), dpi=160,
                bbox_inches="tight")
    plt.close(fig)

    # individual maps
    fig, ax = plt.subplots(figsize=(8, 8))
    plot_map(ax, adm2, "measles_cases",
             "Bangladesh - confirmed measles cases by district (2026)",
             cmap="YlOrRd", vmin=0, vmax=vmax, log=True,
             label="confirmed cases")
    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, "bangladesh_burden_map.png"), dpi=160,
                bbox_inches="tight")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 8))
    plot_map(ax, adm2, "outbreak_risk",
             "Bangladesh - district outbreak risk (2026)",
             cmap="YlGnBu", vmin=0, vmax=1, label="outbreak risk")
    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, "bangladesh_outbreak_risk_map.png"),
                dpi=160, bbox_inches="tight")
    plt.close(fig)

    # console summary
    top_burden = tbl.sort_values("measles_cases", ascending=False).head(8)
    top_risk = tbl.sort_values("outbreak_risk", ascending=False).head(8)
    print("=" * 70)
    print("BANGLADESH MAP - affected areas & outbreak risk (2026)")
    print("=" * 70)
    print("\nTop districts by confirmed measles cases:")
    for _, r in top_burden.iterrows():
        print(f"  {r['shapeName']:15} {int(r['measles_cases']):5d} cases")
    print("\nTop districts by outbreak risk (Step-14 ensemble):")
    for _, r in top_risk.iterrows():
        print(f"  {r['shapeName']:15} risk={r['outbreak_risk']:.2f}  "
              f"outbreak_weeks={int(r['outbreak_weeks'])}/{int(r['n_weeks'])}")
    print("\nSaved: figures/bangladesh_map.png, "
          "figures/bangladesh_burden_map.png, "
          "figures/bangladesh_outbreak_risk_map.png")
    print("       output/outbreak/district_risk_map.csv")


if __name__ == "__main__":
    main()