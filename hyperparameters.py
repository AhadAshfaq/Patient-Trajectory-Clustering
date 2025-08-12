'''
             ============================================   HYPERPARAMETER SETTINGS FOR CLUSTERING PIPELINE   ===========================================
'''


"""
This module defines all global experimental settings and configuration flags for the project pipeline. Parameters included here control key aspects of preprocessing, feature extraction, experimental design, and clustering model selection:

- Observation window size (e.g., number of days before discharge)
- Inclusion thresholds for labs/features (by percentage of admissions)
- Cluster evaluation parameter ranges (e.g., k values for clustering)
- Quality thresholds for metric-based model selection
- Execution flags for selectively enabling/disabling each clustering or analysis method

Centralizing these settings ensures reproducibility, transparency, and ease of experimentation. All downstream workflow modules and notebooks import these parameters to drive consistent, reproducible analyses across the codebase. Edit values here to customize or rerun experiments without modifying algorithm logic elsewhere.

"""

# Set the observation window: Number of days before discharge to include in feature extraction and filtering.
days = 7

# Threshold for lab inclusion: Retain labs measured in at least this fraction of admissions (e.g., 0.75 = 75%).
set_percentage = 75/100

# Range of cluster numbers ("k") to evaluate; used in clustering algorithms to find the optimal number of clusters.
K_RANGE = range(2, 16)

# Silhouette score threshold: Only clusters with average Silhouette score at least this value are considered decent quality.
SIL_THR = 0.25

# Run all clustering methods if True; if False, use RUN_FLAGS below for selective activation.
RUN_ALL = False


#   -----------------------------------------
#           CLUSTERING METHOD SELECTION
#   -----------------------------------------

# Set each method flag to True to enable specific clustering algorithms in the main pipeline. Only methods set to 'True' here will be run if RUN_ALL is set to False.

RUN_FLAGS = {

    # K-MEANS Clustering using 'euclidean' metric
    "KMEANS": True,

    # Agglomerative Clustering (Ward) using 'euclidean' metric
    "AGG_EUCLID": True,

    # Agglomerative Clustering using Manhattan Distance matrix (Unimputed vector)
    "AGG_MANHATTAN_UNIMPUTED": True,

    # Agglomerative Clustering using Manhattan Distance matrix (Imputed vector)
    "AGG_MANHATTAN_IMPUTED": True ,

    # Agglomerative Clustering using Cosine Distance matrix (Unimputed vector)
    "AGG_COSINE_UNIMPUTED": True,

    # Agglomerative Clustering using Cosine Distance matrix (Imputed vector)
    "AGG_COSINE_IMPUTED": True,

    # Agglomerative Clustering using Mahalanobis Distance matrix (Unimputed vector)
    "AGG_MAHALANOBIS_UNIMPUTED": True,

    # Agglomerative Clustering using Mahalanobis Distance matrix(Imputed vector)
    "AGG_MAHALANOBIS_IMPUTED": True,

    # Agglomerative Clustering using precomputed Euclidean Distance matrix (Unimputed vector)
    "AGG_EUCLID_UNIMPUTED": True,

    # Agglomerative Clustering using precomputed Euclidean Distance matrix (Imputed vector)
    "AGG_EUCLID_IMPUTED": True,

    # Agglomerative Clustering using DTW matrix (fast, dtaidistance)
    "AGG_DTW_FAST": True,

    # Agglomerative Clustering using DTW matrix (tslearn)
    "AGG_DTW_TS": True,

    # Agglomerative Clustering using Binary Distance Matrices (Hamming, Jaccard, Dice)
    "BINARY": True,

    #  DBSCAN clustering using 'euclidean' metric
    "DBSCAN_EU": True,

    # Spectral Clustering based on Hidden Markov Model derived features
    "SPECTRAL": True,

    # K-Medoids Clustering using 'euclidean' metric
    "KMEDOIDS_EUCLID" : True

}
