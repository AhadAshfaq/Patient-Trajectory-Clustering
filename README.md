# Clustering Patient Lab Trajectories from Electronic Health Records
## Overview
This repository provides a modular data science pipeline for clustering hospital patient lab trajectories using unsupervised machine learning. Designed for working with time series extracted from MIMIC-IV Electronic health records, the project supports time-series preprocessing, multiple distance metrics, and advanced clustering algorithms for medical time series analysis, enabling exploration and identification of meaningful patient subgroups, such as clinical patterns preceding hospital discharge through advanced clustering and comprehensive evaluation.

## Features
### Data Extraction & Preprocessing
* SQL-based cohort selection for pre-discharge (e.g., 7-day) lab data
* Customisable lab feature selection by measurement frequency
* Temporal discretisation (e.g., aggregation into 24-hour bins)
* Versatile missing data handling: Nomralization, Interpolation, Forward/Backward filling, KNN imputation
* Support for both numeric and binary (present/absent) lab datasets

### Multiple Input Representations
* Non-imputed numeric matrices (for pairwise-measured distance computation)
* Non-imputed normalized numeric matrices (for pairwise-measured distance computation)
* Nomralized, interpolated & imputed (complete) numeric time series matrices
* Binary matrices indicating lab presence/absence

### Flexible Distance Metrics
* Euclidean, Manhattan, Mahalanobis, and Cosine distances
* Dynamic Time Warping (DTW: standard (tslearn) and fast (dtaidistance) implementations)
* Binary metrics: Hamming, Jaccard, Dice

### Clustering Algorithms
* KMeans
* KMedoids
* Agglomerative Hierarchical Clustering (supporting all standard linkages and metric types)
* DBSCAN
* Spectral Clustering

### Comprehensive Evaluation
* Silhouette Score
* Calinski–Harabasz Index
* Davies–Bouldin Index
* Fisher’s Exact Test for associating cluster results with target outcomes (with multiple-testing correction)

### Output & Logging
* Experiment results saved to parameterised folders (all relevant thresholds/choices encoded in folder names)
* Intermediate and final artefacts: preprocessed datasets, distance matrices, cluster labels, evaluation scores, and statistical results
* Option to aggregate all output metrics in master CSV files

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
├── /metric_results            # Outputs: parameterised subfolders for all experiments 
├── /plots                     # All generated visualisations 
├── README.md
```

## Installation
### 1. Clone the Repository
```
git clone https://github.com/yourusername/patient-trajectory-clustering.git
cd patient-trajectory-clustering
```
### 2. Install Dependencies
```
pip install -r requirements.txt
```

### 3. Prepare Data
Place the relevant cohort and lab events files in the /data directory.
Make sure you have appropriate permissions for the dataset (e.g., MIMIC-IV).

## Usage
To run the pipeline with the current configuration:
```
python main.py
```

* Adjust experiment settings (lab feature selection, window length, clustering methods, etc.) in hyperparameters.py.
* Modular execution flags allow running only specific pipeline stages or algorithm variants.

## Customisation & Extensibility
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
















