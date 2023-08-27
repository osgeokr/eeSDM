# eeSDM: Google Earth Engine-based SDM

![](https://github.com/osgeokr/eeSDM/blob/main/IMG/predict_sdm.png?raw=True)

`eeSDM` is a Python package designed for conducting species distribution modeling(SDM) using Google Earth Engine. This package provides researchers in ecology, environmental science, and data science with an efficient toolset to explore and predict the relationship between species distribution and environmental factors.

## Features

- Preprocessing of GBIF Occurrence Data (e.g., heatmap plotting, duplicate removal)
- Multicollinearity Removal for Environmental Variables (VIF)
- Generation of Pseudo-Absence Data (Full extent, spatial constraints, and environmental profiling)
- Spatial Grid Generation
- SDM SDM fitting and prediction
- Compute Variable Importance scores and visualization
- Accuracy assessment (e.g., EUC-ROC, EUC-PR, Sensitivity, Specificity) and Curve Plotting
- Potential Distribution Plotting using Optimal Thresholds

## Installation

To install the `eeSDM` package, you can use the following pip command:

```bash
pip install eeSDM
```

## Usage

Here's a simple example of how to use the geokakao package:
```python
import eeSDM
```

```python
# Plot Yearly & Monthly data distribution
eeSDM.plot_data_distribution(gdf)
```
![](https://github.com/osgeokr/eeSDM/blob/main/IMG/data_distribution_plot.png?raw=True)

```python
# Plot heatmap
eeSDM.plot_heatmap(gdf)
```
![](https://github.com/osgeokr/eeSDM/blob/main/IMG/heatmap_plot.png?raw=True)

```python
# Apply the function to the raw data with the specified GrainSize
Data = eeSDM.remove_duplicates(data_raw, GrainSize)
```

```python
# Perform filtering using VIF (Variance Inflation Factor)
# Apply the function to remove variables with high multicollinearity
# Obtain the list of remaining column names after VIF-based filtering
filtered_PixelVals_df, bands = eeSDM.filter_variables_by_vif(PixelVals_df)
```

```python
# Plot correlation heatmap
eeSDM.plot_correlation_heatmap(filtered_PixelVals_df, h_size=6)
```
![](https://github.com/osgeokr/eeSDM/blob/main/IMG/correlation_heatmap_plot.png?raw=True)

```python
# Generate Random Pseudo-Absence Data in the Entire Area of Interest
AreaForPA = eeSDM.generate_pa_full_area(Data, GrainSize, AOI)

# Generate Spatially Constrained Pseudo-Absence Data (Presence Buffer)
AreaForPA = eeSDM.generate_pa_spatial_constraint(Data, GrainSize, AOI)

# Generate Environmental Pseudo-Absence Data (Environmental Profiling)
AreaForPA = eeSDM.generate_pa_environmental_profiling(Data, GrainSize, AOI, predictors)
```
![](https://github.com/osgeokr/eeSDM/blob/main/IMG/generate_pa.png?raw=True)

```python
# Create a grid of polygons over a specified geometry
Grid = eeSDM.createGrid(AOI, scale=50000)
```

```python
# Fit SDM
results = eeSDM.batchSDM(Grid, Data, AreaForPA, GrainSize, bands, predictors, numiter, split=0.7, seed=None)
```

```python
# Plot Average Variable Importance
eeSDM.plot_avg_variable_importance(results, numiter)
```
![](https://github.com/osgeokr/eeSDM/blob/main/IMG/avg_variable_importance_plot.png?raw=True)

```python
# Calculate AUC-ROC and AUC-PR
eeSDM.calculate_and_print_auc_metrics(images, TestingDatasets, GrainSize, numiter)

# Calculate Sensitivity and Specificity
eeSDM.calculate_and_print_ss_metrics(images, TestingDatasets, GrainSize, numiter)

# Plot ROC and PR curves
eeSDM.plot_roc_pr_curves(images, TestingDatasets, GrainSize, numiter)

# Potential Distribution Map using the optimal threshold
DistributionMap2 = eeSDM.create_DistributionMap2(images, TestingDatasets, GrainSize, numiter, ModelAverage)
```
![](https://github.com/osgeokr/eeSDM/blob/main/IMG/roc_pr_curves_plot.png?raw=True)

## [Case Study 1: Habitat Suitability and Potential Distribution Modeling of Fairy Pitta (Pitta nympha) Using Presence-Only Data](https://github.com/osgeokr/eeSDM/blob/main/eeSDM_Case%20Study%201_Pitta%20nimpha.ipynb)

## References

The content of this packges presents a conversion and enhancement of JavaScript source code provided by researchers from the Smithsonian Conservation Biology Institute. The original JavaScript code has been translated and refined into Python to achieve the same objectives.

1. Crego, R. D., Stabach, J. A., & Connette, G. (2022). Implementation of species distribution models in Google Earth Engine. *Diversity and Distributions*, 28, 904–916. [DOI](https://doi.org/10.1111/ddi.13491)
2. [Smithsonian SDMinGEE GitHub Repository](https://smithsonian.github.io/SDMinGEE/)
