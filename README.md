# Code and Data for:

## Satellite-detectable impacts of shipping nitrogen emissions on the surface ocean

This repository contains the code used in the study titled:

**Satellite-detectable impacts of shipping nitrogen emissions on the surface ocean**,  
in revision at ***Nature Sustainability***.

---

## 1. Data Sources

All analyses rely on publicly available datasets:

- **Chlorophyll *a* concentration**
  - NASA Ocean Biology Distributed Active Archive Center, Aqua MODIS Level-3 Global Mapped Chlorophyll (CHL) Data, version R2022.0:  
    https://doi.org/10.5067/AQUA/MODIS/L3M/CHL/2022.0

- **Chlorophyll *a* concentration (OC-CCI)**
  - ESA Ocean Colour Climate Change Initiative chlorophyll *a* product, version 6.0, NERC EDS Centre for Environmental Data Analysis:  
    Sathyendranath, S. et al. (2019). *Sensors*, 19, 4285.  
    https://catalogue.ceda.ac.uk/uuid/b0ec72a28b6a4829a33ed9adc215d5bc  
    https://doi.org/10.5285/5011d22aae5a4671b0cbc7d05c56c4f0

- **NOₓ emissions from international shipping**
  - CEDS_GBD-MAPS: Global Anthropogenic Emission Inventory (1970–2017):  
    McDuffie, E. E. et al. (2020). *Earth System Science Data*, 12, 3413–3442.  
    https://zenodo.org/records/3754964

- **Net primary productivity (NPP)**
  - CAFE model outputs:  
    Silsbe, G. M. (2024). Monthly global net phytoplankton production and phytoplankton absorbed energy derived from the CAFE model (Version 1.0.0) [Data set]. Zenodo.  
    https://doi.org/10.5281/zenodo.12586956

- **Sea surface temperature (SST) and photosynthetically available radiation (PAR)**
  - MODIS-Aqua Level-3 monthly products, NASA Ocean Biology Distributed Active Archive Center:  
    https://oceandata.sci.gsfc.nasa.gov/l3/

- **Mixed layer depth (MLD)**
  - HYCOM mixed layer depth, as used by the CAFE model (taken from the CAFE inputs).

- **Mean nitrate concentration in the upper 60 m**
  - World Ocean Atlas 2023, Volume 4: Dissolved Inorganic Nutrients:  
    Garcia, H. E. et al. (2024). https://doi.org/10.25923/39qw-7j08  
    https://doi.org/10.25921/va26-hv25

- **Ocean basin mask**
  - World Ocean Atlas 2023 basin mask (basinmask_01.msk):  
    https://www.ncei.noaa.gov/access/world-ocean-atlas-2023/

- **Global shipping routes**
  - Benden, P. (2022). Global Shipping Lanes (v1.3.1) [Data set]. Zenodo.  
    https://doi.org/10.5281/zenodo.6361763

---

## 2. Code

#### `Surrogate_102_all_years.py`

Performs the moving-window linear regression (11×11 pixels, log1p-transformed, at least 70 valid pixels, |slope| ≥ 0.01, R² ≥ 0.6) between chlorophyll *a* and shipping NOₓ emissions for each year from 2013 to 2017, on the real emission field and on 101 surrogate emission fields built from it by shifting the field 10° to 60° east or west, 10° to 20° north or south, both at the same time, or rotating it by 180°. The chlorophyll *a* field stays fixed.

Outputs per year:
- Results table with the number of windows entering the screen, retained windows, positive- and negative-slope windows and positive- and negative-correlation areas for the real field and every surrogate field.
- Window files with the retained windows of every field and all fitted windows of the real field.

These outputs are the inputs of all other scripts, which read the window files of the real field or compare their own window counts with the results table.

---

#### `Surrogate_102_all_years_oc_cci.py`

The same as Surrogate_102_all_years.py with the OC-CCI chlorophyll *a* fields. Its outputs are the inputs of the OC-CCI scripts (Table_S15_S16_S18_surrogate_test_occci.py, Fig_S32_to_S36_Table_S17_maps_overlap_occci.py).

---

#### `Fig_1A_annual_window_shapefiles.py`

Performs the moving-window regression between chlorophyll *a* and shipping NOₓ emissions for each year from 2013 to 2017 and writes the retained windows as shapefiles.

Outputs per year:
- Shapefiles: positive slope, negative slope, all r, r ≥ 0, r < 0, used to produce maps in ArcGIS Pro. The 2017 r shapefiles support Fig. 1(a); the 2017 positive-slope shapefile supports the central map of Fig. 3.

---

#### `Fig_2_nitrate_density_nox_histogram.py`

Maps each 2017 window to the nearest grid cell of the 1° World Ocean Atlas nitrate climatology (mean nitrate concentration in the upper 60 m) and computes an area-weighted kernel density estimate of nitrate concentration for three groups: retained windows with a positive slope, retained windows with a negative slope and all windows entering the screen. It then extracts the NOₓ emission at every retained window of 2013 to 2017, splits the windows by slope sign and computes area-weighted histograms of the emission rate using spherical pixel areas. Requires the window files of Surrogate_102_all_years.py.

Outputs:
- Nitrate density curves with the 2 µmol kg⁻¹ threshold used to define nitrogen-limited regions. This corresponds to Fig. 2(a).
- Combined histogram of positively and negatively correlated areas for 2017. This corresponds to Fig. 2(b).
- Excel summary containing per-year and pooled (2013–2017) area-weighted NOₓ emission statistics (mean, SD, median, IQR, min, max, total area) for positively and negatively correlated windows.

---

#### `Fig_3_Table_1_Table_S22_aoi_fits.py`

Extracts pixel-level chlorophyll *a* concentrations and NOₓ emission values from annual rasters (2013–2017) within five predefined AOIs using GeoJSON masks and fits the regression models of Table 1 on untransformed values: a pooled ordinary least squares regression using AOI 1–4 only, a piecewise linear fit with breakpoint optimization for AOI 1, simple linear regressions for AOI 2–5, and the yearly fits of AOI 2–5 for 2013 to 2017. The fits are repeated with annual SST, PAR and MLD added as covariates to give the slopes after control.

Outputs:
- Table 1: the regression models of chlorophyll *a* on NOₓ emissions for the globe (AOI 1–4 pooled) and the five AOIs, with the slope and its 95% confidence interval half-width, intercept and R² before the control, the slope after the control and the AOI 1 breakpoint.
- One scatter plot per AOI, each showing year-colored points, the AOI-specific fit line, the global linear fit line pooled across AOI 1–4 and the AOI-specific fit after control, plus a legend strip. These correspond to Fig. 3(a)–(e).
- Table S22: AOI 5 by year, and the pooled slope against the range of yearly slopes in AOI 2–5, with and without the covariates controlled.

---

#### `Fig_S21_to_S27_Table_S3_to_S5_npp.py`

Performs the same moving-window regression as Surrogate_102_all_years.py but replaces chlorophyll *a* with NPP as the dependent variable, for each year from 2013 to 2017, and compares the retained NPP windows with the retained chlorophyll *a* windows of the same year. Also fits NPP on NOₓ emissions within the five AOIs. Requires the window files of Surrogate_102_all_years.py.

Outputs:
- Tables of the retained chlorophyll *a* and NPP windows of 2017 with their joint classes (positive in both, chlorophyll *a* positive with no retained NPP window, chlorophyll *a* positive with a negative NPP window, NPP positive only), labeled by AOI. These are overlaid in ArcGIS Pro to produce Fig. 4(a)–(e).
- Global map of the joint classes with the AOI outlines. This corresponds to Fig. S21.
- Maps of the retained NPP–NOₓ windows for 2013–2017. These correspond to Figs. S22–S26.
- Scatter plot of NPP against NOₓ emissions in the five AOIs with the pooled fit. This corresponds to Fig. S27.
- Table S3: overlap between the chlorophyll *a*–NOₓ and NPP–NOₓ analyses in 2017 by region (A) and within-window comparison of the two fields (B).
- Table S4: NPP–NOₓ retained windows and their overlap with the chlorophyll *a* retained windows, 2013–2017.
- Table S5: implied C:N stoichiometry of the NPP sensitivities to shipping NOₓ emissions.

---

#### `Fig_S1_to_S5_annual_maps.py`

Performs the moving-window regression between chlorophyll *a* and shipping NOₓ emissions for each year from 2013 to 2017 and maps the retained windows colored by r on a Robinson projection with the nitrate basemap and the shipping routes.

Outputs per year:
- Map of the retained windows. The 2013–2017 maps correspond to Figs. S1–S5.

---

#### `Table_S1_interannual_window_overlap.py`

Performs the moving-window regression for the five years on one shared grid and measures the recurrence of the retained windows from year to year: the windows retained in all five years, the overlap between adjacent years and the turnover, by count and by area.

Outputs:
- Table S1: recurrence of the retained windows from 2013 to 2017.

---

#### `Table_S2_n_limitation_base_rate.py`

Compares the fraction of retained positive-slope windows of 2017 located in nitrogen-limited water (mean nitrate concentration in the upper 60 m below 2 µmol kg⁻¹) with the base rate among all windows entering the screen, by count and by area, with a one-sided Fisher exact test. Requires the window files of Surrogate_102_all_years.py.

Outputs:
- Table S2: nitrogen limitation among windows entering the screen and retained windows with positive slopes in 2017, with the retained-to-all ratio and the Fisher exact test.

---

#### `Table_S6_Fig_S6_to_S19_temporal_aggregation.py`

Aggregates the monthly chlorophyll *a* and NOₓ emission rasters of 2017 into seasonal composites for summer (Jun–Aug) and winter (Dec–Feb) and uses the twelve monthly fields, with NOₓ expressed as a rate in the unit of the annual field, and performs the moving-window regression on each. The regression algorithm is identical to the annual version.

Outputs:
- r maps of the retained windows for summer and winter 2017. These correspond to Figs. S6 and S7.
- r maps of the retained windows for January–December 2017. These correspond to Figs. S8–S19.
- Table S6: retained windows of the chlorophyll *a*–NOₓ regression at annual, seasonal and monthly aggregation in 2017.

---

#### `Fig_S20_slope_distribution.py`

Performs the 2017 moving-window regression and plots the distribution of the window slopes before and after screening, over the full slope range and by sign of slope over the central range.

Outputs:
- Histograms of the window slopes over the full slope range and by sign of slope over the central range. These correspond to Fig. S20(a) and Fig. S20(b).

---

#### `Table_S7_to_S12_Fig_S28_S29_surrogate_test.py`

Performs the surrogate-field test of Supplementary Note 1. For every year, each of the 101 surrogate fields is compared with the real field on that surrogate's own windows (the common windows) with two statistics, the positive-slope window rate and the positive-correlation area, and the rank of the real field among the 102 fields gives the empirical p value. For 2017 the comparison is also made within each basin of the World Ocean Atlas basin mask and by displacement direction. Requires the results tables and window files of Surrogate_102_all_years.py.

Outputs:
- Table S7: composition of the surrogate family.
- Table S8: global results of the surrogate-field test by year.
- Table S9: change in the two statistics with displacement distance.
- Table S10: basin definitions and baseline screening results.
- Table S11: basin-level surrogate test.
- Table S12: basin results by displacement direction.
- Histograms of the two statistics over the surrogate fields. This corresponds to Fig. S28.
- Decay of the two statistics with displacement distance. This corresponds to Fig. S29.
- Basin-level test table, read by Table_S15_S16_S18_surrogate_test_occci.py.

---

#### `Fig_S30_surrogate_six_panel_map.py`

Maps the retained windows of the real 2017 emission field and of five surrogate fields (the 10° westward shift, the 15° southward shift, the shift of 20° west and 10° north, the 60° westward shift and the rotation by 180°) with the shipping routes and the basin boundaries. Requires the window files of Surrogate_102_all_years.py.

Outputs:
- Six-panel map. This corresponds to Fig. S30.

---

#### `Fig_S31_Table_S13_S14_conditions.py`

Evaluates the three conditions for a detectable effect of Supplementary Note 2 (nitrogen limitation, route presence and low background variability) in every window entering the screen from 2013 to 2017 and computes the retained-to-all ratio of the retained positive-slope windows for each condition, by the number of conditions met, by year and by basin. Requires the window files of Surrogate_102_all_years.py.

Outputs:
- Map of the number of conditions met in 2017 with the positive-correlation area of 2017 and of 2013–2016 overlaid. This corresponds to Fig. S31.
- Table S13 (A–E): the three conditions for a detectable effect and the retained-to-all ratio of each condition.
- Table S14: the three conditions by basin in 2017.
- Window-level condition table of 2017, read by Fig_S37_Table_S20_S21_covariate_windows.py.

---

#### `Table_S15_S16_S18_surrogate_test_occci.py`

Repeats the global surrogate-field test with the OC-CCI chlorophyll *a* fields for every year and the 2017 basin-level test for both chlorophyll *a* products side by side. Requires the outputs of Surrogate_102_all_years_oc_cci.py and the basin-level test table of Table_S7_to_S12_Fig_S28_S29_surrogate_test.py.

Outputs:
- Table S15: global surrogate-field test with the OC-CCI fields by year (layout of Table S8).
- Table S16: decay of the two statistics with displacement distance with the OC-CCI fields (layout of Table S9).
- Table S18: basin-level surrogate-field test in 2017 with the MODIS-Aqua and the OC-CCI fields side by side.

---

#### `Fig_S32_to_S36_Table_S17_maps_overlap_occci.py`

Performs the moving-window regression with the OC-CCI and the MODIS-Aqua chlorophyll *a* fields for each year from 2013 to 2017, maps the retained windows of the two products side by side in the style of Figs. S1–S5, and measures the overlap of the two retained-window sets year by year.

Outputs:
- Maps of the retained windows for 2013–2017 with OC-CCI and MODIS-Aqua side by side. These correspond to Figs. S32–S36, OC-CCI in Fig. S32(a)–S36(a) and MODIS-Aqua in Fig. S32(b)–S36(b).
- Table S17: overlap of the retained windows between the MODIS-Aqua and the OC-CCI fields, 2013–2017.

---

#### `Table_S19_aoi_fits_two_products_occci.py`

Repeats the emission-alone fits of Table 1 within the five AOIs with the MODIS-Aqua and the OC-CCI chlorophyll *a* fields side by side.

Outputs:
- Table S19: sensitivity of chlorophyll *a* to NOₓ emissions in the five AOIs (the fits of Table 1) computed with the MODIS-Aqua and the OC-CCI fields.

---

#### `Fig_S37_Table_S20_S21_covariate_windows.py`

Computes, within every retained positive-slope window of 2017, the correlation of chlorophyll *a* and of the emission field with SST, PAR and MLD, then repeats the window regression with SST alone and with SST, PAR and MLD as covariates, keeps the windows in which adding the emission field raises R² by more than the threshold, and compares the real field with every surrogate field on the common windows by the number and the area of these windows. Requires the window-level condition table of Fig_S31_Table_S13_S14_conditions.py and the window files of Surrogate_102_all_years.py.

Outputs:
- Table S20: co-variation of chlorophyll *a* and the emission field with SST, PAR and MLD within the retained positive-slope windows of 2017, by basin.
- Table S21: comparison with displaced routes after controlling for the covariates.
- Histograms of the number and the area of the windows keeping an emission effect over the surrogate fields. This corresponds to Fig. S37.

---

## 3. AOI Definitions

The `all_AOIs_geojson/` directory contains GeoJSON polygon boundaries for the five areas of interest used by the AOI-level scripts (Fig_3_Table_1_Table_S22_aoi_fits.py, Fig_S21_to_S27_Table_S3_to_S5_npp.py, Table_S19_aoi_fits_two_products_occci.py):

| File | Region |
|------|--------|
| `AOI1_NW_Pacific_off_Japan.geojson` | Eastern Japan–North Pacific Ocean |
| `AOI2_Central_South_Atlantic.geojson` | Central South Atlantic Ocean |
| `AOI3_North_Atlantic.geojson` | North Atlantic Ocean |
| `AOI4_NE_Pacific.geojson` | NE Pacific Ocean |
| `AOI5_Eastern_Pacific.geojson` | Eastern Pacific Ocean, offshore Central America |

---

## 4. Computing Environment and Reproduction

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
  - statsmodels: 0.14.4
  - netCDF4: 1.7.2
  - openpyxl: 3.1.5

- **System Requirements**
  - Runs on any standard desktop or laptop computer with Python 3.12 installed.
  - No non-standard hardware required (CPU only; no GPU needed).
  - Tested on Windows 11.

- **Installation**
  - Install the required packages via pip:
    ```
    pip install numpy==1.26.3 pandas==2.2.3 rasterio==1.4.3 scipy==1.15.1 matplotlib==3.9.1 cartopy==0.24.1 geopandas==1.0.1 statsmodels==0.14.4 netCDF4==1.7.2 openpyxl==3.1.5
    ```
    Typical installation time is under 10 minutes on a normal desktop computer.

- **How to Run**
  - Each script is run independently from the command line, for example:
    ```
    python Fig_2_nitrate_density_nox_histogram.py
    ```
    Update the data directory paths at the top of each script to point to your local copies of the datasets listed in Section 1.
  - Run the two surrogate scripts first; the other scripts read their outputs as described in Section 2:
    ```
    python Surrogate_102_all_years.py
    python Surrogate_102_all_years_oc_cci.py
    ```
  - Shapefile outputs are then visualized in ArcGIS Pro to produce Fig. 1(a), the central map of Fig. 3 and Fig. 4(a)–(e).

- **Reproduction**
  - Each script corresponds to specific figures and quantitative results, as detailed in Section 2.
  - Slope, R², confidence interval, rank and p values reported in the main text, Table 1 and the Supplementary Information are written to the tables of the corresponding scripts and printed to the console when they are run.

---


## Citation

If you reference this work, please cite the associated manuscript:

> H. Lan, X. Zhang, V. J. Coles, G. M. Silsbe, Satellite-detectable impacts of shipping nitrogen emissions on the surface ocean. *Nature Sustainability* (in revision).

Full citation details will be updated upon publication.
