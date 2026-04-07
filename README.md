# Code and Data for:

## Emissions from shipping increase marine phytoplankton biomass in oligotrophic regions

This repository contains the code used in the study titled:

**Emissions from shipping increase marine phytoplankton biomass in oligotrophic regions**,  
submitted to **Science**.

---

## 1. Data Sources

All analyses rely on publicly available datasets:

- **Chlorophyll *a* concentration**
  - NASA Earth Observations, Aqua MODIS Level-3 Global Mapped Chlorophyll (CHL) Data, version 2022.0:  
    https://neo.gsfc.nasa.gov/ (dataset MY1DMM_CHLORA)

- **NOₓ emissions from international shipping**
  - CEDS_GBD-MAPS: Global Anthropogenic Emission Inventory (1970–2017):  
    McDuffie, E. E. et al. (2020). *Earth System Science Data*, 12, 3413–3442.  
    https://zenodo.org/records/3754964

- **Net primary productivity (NPP)**
  - CAFE model outputs:  
    Silsbe, G. M. (2025). Monthly global net phytoplankton production and phytoplankton absorbed energy derived from the CAFE model [Data set]. Zenodo.  
    https://doi.org/10.5281/zenodo.15497141

- **Nitrogen concentration at 60 m depth**
  - World Ocean Atlas 2023, Volume 4: Dissolved Inorganic Nutrients:  
    Garcia, H. E. et al. (2024).

- **Global shipping routes**
  - Benden, P. (2022). Global Shipping Lanes (v1.3.1) [Data set]. Zenodo.  
    https://doi.org/10.5281/zenodo.6361763

---

## 2. Code

### `Fig_1A_S1-S5_moving_window_regression_annual.py`

Performs moving-window linear regression (11×11 pixels, log1p-transformed) between satellite-derived chlorophyll *a* concentrations and shipping NOₓ emissions for each year from 2013 to 2017. For each pixel, the regression yields slope, Pearson correlation coefficient (r), R², and p-value. Only pixels with R² ≥ 0.6 are retained.

Outputs per year:
- Shapefiles: positive slope, negative slope, all r, r ≥ 0, r < 0, used to produce maps in ArcGIS Pro. The 2017 r shapefiles support Fig. 1A; the 2013–2016 r shapefiles support figs. S1–S4; the 2017 slope shapefiles support Fig. 3B.
- Static maps: R² map, r map, and slope map, rendered with Cartopy for quick visual inspection. The 2013–2017 r maps correspond to figs. S1–S5.

Pooled outputs:
- Excel summary containing per-year and pooled (2013–2017) area-weighted NOₓ emission statistics (mean, SD, median, IQR, min, max, total area) for positively correlated pixels.

Console outputs:
- Per-year pixel count and total area (km²) for positive-slope pixels with R² ≥ 0.6.
- Average total area across all years.
- Pooled 2013–2017 report line with area-weighted median, IQR, mean ± SD, total area, and pixel count.

---

### `Fig_2A_nmean_density_2017.py`

Runs moving-window regression on 2017 chlorophyll *a* and NOₓ emission rasters, then maps each regression pixel to the nearest grid cell in a 1° World Ocean Atlas nitrogen concentration dataset (mean N at 60 m depth). Pixels are grouped into three categories based on regression results (R² ≥ 0.6): significantly positive correlation, significantly negative correlation, and all grid cells. For each group, an area-weighted kernel density estimate (KDE) of nitrogen concentration is computed and plotted, with x-axis zoomed to 0–8 µmol kg⁻¹.

Outputs:
- PNG showing three overlaid KDE curves (positive in orange, negative in blue, all grids in grey dashed) with a red dashed line marking the 2 µmol kg⁻¹ N-limitation threshold. This corresponds to Fig. 2A.

---

### `Fig_2B_chlor_nox_correlated_area_histogram.py`

Performs moving-window linear regression (11×11 pixels, log1p-transformed) between chlorophyll *a* and shipping NOₓ emissions for each year from 2013 to 2017, then extracts NOₓ emission values at pixels with significant correlations (R² ≥ 0.6). Pixels are split by slope sign into positively and negatively correlated groups. For each group, an area-weighted histogram is computed using spherical pixel areas.

Outputs per year + pooled (2013–2017):
- PNG histogram (positive slope only, blue bars). Shows the distribution of NOₓ emission rates across ocean areas with significant positive chlorophyll–NOₓ correlations.
- PNG histogram (negative slope only, yellow bars). Same structure for negatively correlated areas.
- PNG combined histogram (positive blue + negative yellow overlaid on shared bin edges, with a unified legend showing pixel count and total area for each group). The 2017 combined histogram corresponds to Fig. 2B.

---

### `Fig_3A_global_linear_fit.py`

Extracts pixel-level chlorophyll *a* concentrations and NOₓ emission values from annual rasters (2013–2017) within five predefined AOIs using GeoJSON masks. A pooled ordinary least squares regression is fitted using data from AOI 1–4 only, excluding AOI 5 due to its anomalously high response. The regression yields slope, intercept, R², and 95% confidence intervals for the slope and intercept.

Outputs:
- PNG scatter plot showing all five AOIs color-coded, with the pooled linear fit line and 95% CI band. This corresponds to the dashed black global linear fit line shown across all AOI panels in Fig. 3A.

Console outputs:
- Per-AOI record count after pixel extraction.
- Total record count across all AOIs.
- Global linear fit parameters from sklearn (n, slope, intercept, R², mean NOₓ, mean Chl *a*).
- OLS 95% CI for slope and intercept (point estimate, confidence bounds, half-width). These values are reported in the main text and Table 1 as the global model sensitivity.

---

### `Fig_3A_per_aoi_scatter.py`

Extracts pixel-level chlorophyll *a* and NOₓ values from annual rasters (2013–2017) within each of five AOIs using GeoJSON masks, and fits AOI-specific regression models. AOI 1 uses a piecewise linear fit with automatic breakpoint optimization; AOI 2–5 use simple linear regression. AOI 5 additionally reports OLS 95% confidence intervals for the slope, both pooled across all years and for 2013 alone. Each AOI panel also plots the global linear fit line (from Fig_3A_global_linear_fit.py) as a reference.

Outputs:
- One PNG scatter plot per AOI, each showing year-colored points, the AOI-specific fit line (red solid), and the global linear fit line (black dashed). These correspond to the five sub-panels in Fig. 3A.

Console outputs:
- Global linear line parameters and R² evaluated against all AOI data.
- AOI 1 piecewise fit: breakpoint, per-segment slope/intercept/R², and overall R². These values are reported in Table 1.
- AOI 2–4 linear fit: slope, intercept, R² per AOI. These values are reported in Table 1.
- AOI 5 linear fit with OLS 95% CI for the slope (all years pooled and 2013 only). These values are reported in the main text and Table 1.

---

### `Fig_4_npp_nox_moving_window_regression.py`

Performs the same moving-window linear regression as Fig_1A_S1-S5_moving_window_regression_annual.py, but replaces chlorophyll *a* with NPP as the dependent variable. The regression algorithm, window size (11×11), log1p transformation, filtering criteria (R² ≥ 0.6, minimum 70 valid pixels), and output structure are identical to the chlorophyll version.

Outputs per year:
- Shapefiles: positive slope, negative slope, all r, r ≥ 0, r < 0. These shapefiles are overlaid with the chlorophyll regression shapefiles in ArcGIS Pro to produce the NPP–NOₓ correlation grids shown in Fig. 4.
- Static maps: R², r, and slope maps rendered with Cartopy for visual inspection.

Console outputs:
- Per-year pixel count and total area (km²) for positive-slope pixels with R² ≥ 0.6.
- Average total area across all years.

---

### `Fig_5_npp_nox_aoi_scatter.py`

Extracts pixel-level NPP and NOₓ emission values from annual rasters (2013–2017) within five AOIs using GeoJSON masks. Structure mirrors Fig_3A_global_linear_fit.py but uses NPP instead of chlorophyll *a* as the dependent variable. A pooled OLS regression is fitted using AOI 1–4 only, with AOI 5 fitted separately due to its distinct response. Both fits report slope, intercept, R², and 95% confidence intervals.

Outputs:
- PNG scatter plot showing all five AOIs color-coded with the pooled AOI 1–4 linear fit line. This corresponds to Fig. 5.

Console outputs:
- Global linear fit parameters from sklearn (n, slope, intercept, R², mean NOₓ, mean NPP) using AOI 1–4.
- OLS 95% CI for slope and intercept (AOI 1–4 pooled). These values are reported in the main text.
- AOI 5 linear fit with OLS 95% CI for the slope. These values are reported in the main text.

---

### `Fig_S6-S7_seasonal_chlor_nox_regression.py`

Aggregates monthly chlorophyll *a* and NOₓ emission rasters into seasonal composites for summer (Jun–Aug) and winter (Dec–Feb) of 2017. Chlorophyll is averaged across valid months; NOₓ fluxes (kg m⁻² s⁻¹) are integrated over each month's seconds to yield seasonal totals (g m⁻²). Pixels require at least 2 valid months out of 3. The regression algorithm is identical to the annual version.

Outputs per season (summer and winter):
- PNG r map, R² map, and slope map for each season. The r maps correspond to fig. S6 (summer 2017) and fig. S7 (winter 2017).

Intermediate files:
- Seasonal chlor mean and NOₓ total GeoTIFFs for each season, written to a tmp directory and read back for regression.

---

### `Fig_S8-S19_monthly_chlor_nox_regression.py`

Performs moving-window regression between monthly chlorophyll *a* and NOₓ emissions for each month of 2017 (January–December). The regression algorithm is identical to the annual and seasonal versions. NOₓ monthly mean fluxes (kg m⁻² s⁻¹) are integrated over the actual number of seconds in each month to yield monthly totals (g m⁻²), consistent with the seasonal script.

Outputs per month:
- PNG r map, R² map, and slope map for each month. The r maps for January–December 2017 correspond to figs. S8–S19.

---

## 3. AOI Definitions

The `all_AOIs_geojson/` directory contains GeoJSON polygon boundaries for the five areas of interest used by the AOI-level scripts (Fig_3A_*, Fig_5_*):

| File | Region |
|------|--------|
| `AOI1_NW_Pacific_off_Japan.geojson` | Eastern Japan–North Pacific Ocean |
| `AOI2_Central_South_Atlantic.geojson` | Central South Atlantic Ocean |
| `AOI3_North_Atlantic.geojson` | North Atlantic Ocean |
| `AOI4_NE_Pacific.geojson` | NE Pacific Ocean |
| `AOI5_Eastern_Pacific.geojson` | Eastern Pacific Ocean, offshore Central America |

---

## 4. Computing Environment

- **Hardware**
  - Intel® Core™ Ultra 9 Processor 285K (36M Cache, up to 5.70 GHz)
  - All analyses were executed on CPU; no GPU was used.

- **Key Software Versions**
  - Python: 3.12
  - ArcGIS Pro: 3.5.3
  - NumPy: 1.26.3
  - pandas: 2.2.3
  - rasterio: 1.4.3
  - scipy: 1.15.1
  - matplotlib: 3.9.1
  - Cartopy: 0.24.1
  - geopandas: 1.0.1
  - scikit-learn: 1.6.1
  - statsmodels: 0.14.4
  - seaborn: 0.13.2
  - folium: 0.19.4
  - openpyxl: 3.1.5

---

## Citation

If you reference this work, please cite the associated manuscript:

> H. Lan, X. Zhang, V. J. Coles, G. M. Silsbe, Emissions from shipping increase marine phytoplankton biomass in oligotrophic regions. *Science* (submitted).

Full citation details will be updated upon publication.
