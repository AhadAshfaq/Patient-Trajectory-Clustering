# Clustering Patient Lab Trajectories from Electronic Health Records
## Overview
This repository provides a modular data science pipeline for clustering hospital patient lab trajectories using unsupervised machine learning. Designed for working with time series extracted from MIMIC-IV Electronic health records, the project supports time-series preprocessing, multiple distance metrics, and advanced clustering algorithms for medical time series analysis, enabling exploration and identification of meaningful patient subgroups, such as clinical patterns preceding hospital discharge through advanced clustering and comprehensive evaluation.

## Features
### Data Extraction & Preprocessing
* SQL-based cohort selection for pre-discharge (e.g., 7-day) lab data
* Customisable lab feature selection by measurement frequency
* Temporal discretisation (e.g., aggregation into 24-hour bins)
* Versatile missing data handling: Normalization, Interpolation, KNN imputation
* Support for both numeric and binary (present/absent) lab datasets

### Multiple Input Representations
* Non-imputed numeric vector (for pairwise-measured distance computation)
* Non-imputed normalized numeric vector (for pairwise-measured distance computation)
* Normalized & imputed (complete) numeric time series vector
* Binary matrices indicating lab presence/absence

### Flexible Distance Matrices
* Euclidean, Manhattan, Mahalanobis, and Cosine distances
* Dynamic Time Warping (DTW: standard (tslearn) and fast (dtaidistance) implementations)
* Binary metrics: Hamming, Jaccard, Dice

### Clustering Algorithms
* KMeans
* KMedoids
* Agglomerative Hierarchical Clustering (supporting all standard linkages and metric types)
* Density Based Spatial Clustering of Applications with Noise - DBSCAN
* Spectral Clustering

### Comprehensive Evaluation
* Silhouette Score
* Calinski–Harabasz Index
* Davies–Bouldin Index
* Fisher’s Exact Test for associating cluster results with target outcomes (with multiple-testing correction)
* Cluster Feature Analysis - Mann Whitney Test

### Output & Logging
* Experiment results saved to parameterised folders (all relevant thresholds/choices encoded in folder names)
* Intermediate and final artefacts: preprocessed datasets, distance matrices, cluster labels, evaluation scores, and statistical results
* Possibility to aggregate all output metrics in master CSV files

### Reproducibility & Extensibility
* Modular OOP codebase for easy maintenance or extension (new methods, representations, or studies)
* Central configuration for all pipeline options and experiment flags
* Results caching for fast repeatability

## Directory Structure
```
├── main.py                    # Pipeline orchestration script
├── imports.py                 # Central package imports
├── hyperparameters.py         # Central config of all parameters and experiment toggles
├── utils.py                   # Preprocessing functions and helpers 
├── clustering.py              # Distance metrics, clustering algorithms, and evaluators 
├── /preprocessing_data        # Input cohort/lab data (user-supplied) and all necessary preprocessed data
├── /distance_matrices         # Load/save required distance matrix  
├── /metric_results            # Outputs: parameterised subfolders for all experiments that include all generated plots & CSVs
├── /cluster_interpretation    # Spectral Cluster-level stats & plots (see cluster_analysis.py for functions to load/aggregate cluster assignments, Fisher results,                                  lab features, statistical test CSVs, and generate boxplots/barplots)
├── /Result_figures            # Advanced plots, tables, and clustering analysis outputs
├── README.md
```

## Installation
### 1. Clone the Repository
```
git clone https://github.com/ahadashfaq94/BIONET_PROJECT.git
cd BIONET_PROJECT
```
### 2. Install Dependencies
```
pip install -r requirements.txt
```

### 3. Data Preparation
This project requires two dataset files, which you need to acquire by following instructions at [MIMIC-IV Clinical Database on PhysioNet](https://physionet.org/content/mimiciv/3.1/):

1. **Cohort File (`cohort1_target.csv`):**  
   Contains patient-level and admission-level metadata including patient IDs, admission IDs, and key outcomes (e.g., admission and discharge times , target labels).  

2. **Lab Events File (`labevents.csv`):**  
   Contains detailed laboratory measurement records linked to patient admissions, with timestamps and lab test identifiers.  

Researchers must obtain access via credentialed request and download the relevant extraction. `Alternatively`, users can prepare and provide their dataset files in the same format.

3. **Data Placement:**
   Place the required cohort and lab events files inside the /data directory at the root of the project. Ensure that the file paths in the load_data() function within main.py correctly point to these files before running the pipeline.

## Usage
To run the pipeline with the current configuration:
```
python main.py
```

* Adjust experiment settings (lab feature selection, window length, clustering methods, etc.) in 'hyperparameters.py'.
* Modular execution flags allow running only specific pipeline stages or algorithm variants.

## Clustering Analysis & Clinical Significance
The cluster_analysis.py module produces advanced evaluation and interpretation outputs:
* **Clustering Results 1:** Compare clustering algorithms and thresholds using standard validity indices.
* **Clustering Results 2:** Evaluate distance metrics for agglomerative clustering and highlight best performers.
* **Clustering Results 3:** Assess effects of imputation and binary/numeric encoding on clustering quality.
* **Significance Tables and Plots:** Identify clusters with significant Fisher’s test results, summarize cluster sizes and p-values, and visualize discriminatory laboratory markers across patient groups (all statistical tests FDR-corrected).

### How to Use:
To run the cluster analysis and generate summary figures/tables:
```
python cluster_analysis.py
```
See /Result_figures for example output figures and summary tables.

## Customization & Extensibility
* Add Input Types: Extend or modify data processing routines in utils.py.
* Add Distance Metrics or Clustering Methods: Implement in clustering.py.
* Enable/Disable Pipeline Stages: Set the relevant toggles & parameters in hyperparameters.py (RUN_FLAGS).
* Save new outputs: Use or extend provided helper functions for fully reproducible results.

## Reproducibility
* All output folders are parameterised by threshold, method, and other experiment settings to ensure transparent tracking.
* Intermediate and final datasets, metrics, and labels are cached for rapid experimentation.
* The modular architecture allows easy rerunning of only affected stages if a parameter is changed.

## Contributing
Open to issues and pull requests for improvements, bugfixes, or new experiment modules! Community contributions are welcome.

## License
MIT License

## Further Information
For further details, see the code comments and function docstrings, or reach out via GitHub.
















