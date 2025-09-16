'''
                                 ================================ IMPORTS MODULE =============================                                                       
'''

"""
This module serves as the centralized import hub for all external and core Python packages required by the project. It includes standard libraries (NumPy, pandas), scikit-learn machine learning tools, clustering algorithms, distance computation routines, statistical analysis utilities, time-series and sequence analysis packages, plotting/visualization libraries, and typing helpers.

By consolidating all imports here, the codebase maintains consistency, eases dependency management, and simplifies environment setup and troubleshooting.
All analysis, preprocessing, and modeling modules import from this file to ensure a single source of truth for package requirements.

"""


# ===================== CORE PYTHON & OS UTILITIES =====================
import numpy as np                    # Numerical computations and arrays
import pandas as pd                   # DataFrame operations and data manipulation
from datetime import timedelta        # Time delta calculations for dates/times
import time                           # Timing operations/performance measurement
import os                             # OS-level operations (file paths, environment variables)
import re                             # Standard library for working with regular expressions, useful for extracting information from filenames.
from pathlib import Path              # Flexible, object-oriented handling of file system paths and directory operations
os.environ["OMP_NUM_THREADS"] = "20"  # Suppress multi-threading OpenMP warnings


# ===================== LOGGING =====================
import logging                        # Standard Python event logging system


# ===================== IMPUTATION & CLUSTERING BASICS =====================
from sklearn.impute import KNNImputer      # KNN-based missing value imputation
from sklearn.cluster import KMeans         # KMeans clustering algorithm


# ===================== CLUSTERING METRICS/EVALUATION =====================
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score  # Clustering performance metrics


# ===================== VISUALIZATION =====================
import matplotlib.pyplot as plt     # Plotting and figure generation
import seaborn as sns               # High-level statistical data visualization library built on matplotlib, especially useful for boxplots & distribution plots
import matplotlib.lines as mlines       # Line2D object creation for custom legends & manual legend entries
import matplotlib.patches as mpatches   # Shape drawing and annotation, e.g., rectangles/circles for patch-based legends
import matplotlib.gridspec as gridspec  # GridSpec for flexible subplot grid layout management in figures with precise control over subplot positioning & spacing


# ===================== DISTANCE CALCULATIONS =====================
from sklearn.metrics import pairwise_distances          # General pairwise distances between observations
from scipy.stats import fisher_exact                    # Fisher's exact test for categorical data analysis
from statsmodels.stats.multitest import multipletests   # Multiple testing correction for p-values
from scipy.stats import kruskal, mannwhitneyu           # Kruskal: non-parametric test to compare groups, Mannwhitneyu: non-parametric test to compare two groups


# ===================== ADVANCED DISTANCES =====================
from sklearn.metrics.pairwise import cosine_distances       # Cosine distance for vectorized feature matrices
from tslearn.metrics import cdist_dtw                       # DTW (Dynamic Time Warping) distance for time series


# ===================== HIERARCHICAL CLUSTERING & MATRIX OPS =====================
from sklearn.cluster import AgglomerativeClustering         # Agglomerative (hierarchical) clustering
from numpy.linalg import inv, LinAlgError                   # Linear algebra operations: matrix inversion and error handling
from scipy.linalg import pinv                               # Pseudo-inverse for numerical stability


# ===================== FAST TIME SERIES DISTANCES =====================
from dtaidistance import dtw                                # Highly efficient DTW from dtaidistance package


# ===================== DENSITY-BASED & NEIGHBOR ALGORITHMS =====================
from sklearn.cluster import DBSCAN                          # Density-based clustering (DBSCAN)
from sklearn.neighbors import NearestNeighbors              # Fast neighbor lookup (for DBSCAN epsilon estimation)


# ===================== HIDDEN MARKOV MODELS & SPECTRAL CLUSTERING =====================
from hmmlearn import hmm                                   # Hidden Markov Models for sequence modeling
from sklearn.cluster import SpectralClustering             # Spectral clustering algorithm


# ===================== STRUCTURED DATA & TYPING =====================
from dataclasses import dataclass                          # Data class for structured code
from typing import Iterable, List, Dict                    # Type hints for improved code clarity and static checks


# ===================== K-MEDOIDS CLUSTERING =====================
from sklearn_extra.cluster import KMedoids                 # K-Medoids clustering (partitioning, robust to outliers)
