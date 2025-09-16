'''
              =====================================================   CLUSTERING AND METRICS MODULE   ====================================================

'''


'''
This module provides implementations for clustering, metrics computation, and statistical evaluation (Fisher's test) for different algorithms and distance types (KMeans, Agglomerative, KMedoids, custom/precomputed distances, DBSCAN, and Spectral), supporting a wide range of experiment designs in the pipeline.

Supported clustering methods include:
 - KMeans
 - Agglomerative clustering with various distance metrics (Euclidean, Manhattan, Mahalanobis, Cosine, DTW, Binary)
 - KMedoids clustering using 'euclidean' metric
 - DBSCAN (density-based spatial clustering) using 'euclidean' metric
 - Spectral clustering (including HMM-derived features)
'''

from imports import *
from hyperparameters import set_percentage

# ------------------------------------------
#  DATA CONTAINERS FOR EVALUATION METRICS
# ------------------------------------------

@dataclass
class KMeansMetricsRow:
    """
    Container for clustering metrics of KMeans solutions at a given number of clusters (k):
    - silhouette: average silhouette score (range -1 to 1, higher is better)
    - calinski_harabasz: Calinski-Harabasz index (higher indicates better-defined clusters)
    - davies_bouldin: Davies-Bouldin index (lower indicates better clusters)
    """
    k: int
    silhouette: float
    calinski_harabasz: float
    davies_bouldin: float


@dataclass
class AggloMetricsRow:
    """
    Container for clustering metrics of K‑MEDOIDS clustering solutions at given k:
    Same metric definitions as for KMeans.
    """
    k: int
    silhouette: float
    calinski_harabasz: float
    davies_bouldin: float


@dataclass
class KMedoidsMetricsRow:
    """
    Container for clustering metrics of Agglomerative clustering solutions at given k:
    Same metric definitions as for KMeans.
    """
    k: int
    silhouette: float
    calinski_harabasz: float
    davies_bouldin: float


# Select top cluster counts for each metric, applying a silhouette threshold
def get_top_clusters_with_threshold(
    metrics_df: pd.DataFrame,
    silhouette_threshold: float = 0.25,
    top_n: int = 3,
    logger: logging.Logger | None = None,
) -> Dict[str, List[int]]:

    """
    Select top cluster counts for each metric.
    Typically, columns must be ['k', 'silhouette', 'calinski_harabasz', 'davies_bouldin'].
    Returns lists of top k for each metric, respecting threshold for silhouette.

    """
    log = logger or logging.getLogger("MAIN")

    top_clusters: Dict[str, List[int]] = {}

    # Silhouette with threshold
    sil_df = metrics_df[metrics_df["silhouette"] > silhouette_threshold]
    if sil_df.empty:
        log.info("Silhouette Score: No cluster count (k) returns Silhouette score >= %.2f, therefore returning [].\n", silhouette_threshold)
        top_clusters["silhouette"] = []
    else:
        top_clusters["silhouette"] = sil_df.nlargest(top_n, "silhouette")["k"].tolist()

    # Calinski-Harabasz (higher better)
    top_clusters["calinski_harabasz"] = (
        metrics_df.nlargest(top_n, "calinski_harabasz")["k"].tolist()
    )

    # Davies-Bouldin (lower better)
    top_clusters["davies_bouldin"] = (
        metrics_df.nsmallest(top_n, "davies_bouldin")["k"].tolist()
    )

    return top_clusters


# ------------------------------------------------------
#          KMEANS CLUSTERING AND EVALUATION
# ------------------------------------------------------

# Run KMeans for a range of k and compute cluster evaluation metrics
def run_kmeans_metrics(
    df_features: pd.DataFrame,
    id_vars: Iterable[str],
    k_values: Iterable[int] = range(2, 16),
    random_state: int = 42,
    n_init: int = 10,
    logger: logging.Logger | None = None,
) -> pd.DataFrame:
    """Run KMeans for each value of k, returning a DataFrame of metrics."""
    log = logger or logging.getLogger("MAIN")

    feature_cols = [c for c in df_features.columns if c not in id_vars]
    X = df_features[feature_cols].values

    rows: List[KMeansMetricsRow] = []
    for k in k_values:
        log.info("KMeans clustering for k=%d ...", k)
        km = KMeans(n_clusters=k, random_state=random_state, n_init=n_init)
        labels = km.fit_predict(X)

        rows.append(
            KMeansMetricsRow(
                k=k,
                silhouette=silhouette_score(X, labels),
                calinski_harabasz=calinski_harabasz_score(X, labels),
                davies_bouldin=davies_bouldin_score(X, labels),
            )
        )

    return pd.DataFrame([r.__dict__ for r in rows])

# Run Fisher's exact test to assess cluster/outcome associations for KMeans clusters
def run_fisher_test(
    df_features: pd.DataFrame,
    id_vars: Iterable[str],
    k: int,
    outcome_df: pd.DataFrame,
    outcome_col: str = "target",
    random_state: int = 42,
    n_init: int = 10,
) -> pd.DataFrame:
    """Fisher's test for KMeans clusters at k."""
    feature_cols = [c for c in df_features.columns if c not in id_vars]
    X = df_features[feature_cols].values

    km = KMeans(n_clusters=k, random_state=random_state, n_init=n_init)
    labels = km.fit_predict(X)

    # Merge with outcomes for Fisher's test
    merged = (
        pd.DataFrame({"hadm_id": df_features["hadm_id"].values, "cluster": labels})
        .merge(outcome_df[["hadm_id", outcome_col]], on="hadm_id", how="left")
        .dropna(subset=[outcome_col])
    )

    # Perform Fisher's exact test per cluster vs. rest
    p_vals = []
    clusters = np.unique(labels)
    for cid in clusters:
        in_c = merged["cluster"] == cid
        target_in = merged.loc[in_c, outcome_col]
        target_out = merged.loc[~in_c, outcome_col]

        a = (target_in == 1).sum(); b = (target_out == 1).sum()
        c = (target_in == 0).sum(); d = (target_out == 0).sum()

        _, p = fisher_exact([[a, b], [c, d]], alternative="greater")
        p_vals.append(p)

    # Multiple testing correction (Bonferroni)
    rejected, pvals_corr, _, _ = multipletests(p_vals, method="bonferroni")

    return pd.DataFrame({
        "Cluster": clusters,
        "Raw_p_value": p_vals,
        "Corrected_p_value": pvals_corr,
        "Significant_after_correction": rejected,
    })



#   ==========================================================  AGGLOMERATIVE CLUSTERING & METRICS (VARIOUS DISTANCES)=====================================================


# ------------------------------------------------
#    AGGLOMERATIVE CLUSTERING: Ward (Euclidean)
# ------------------------------------------------

# Ward Agglomerative clustering with Euclidean distance for a range of k, plus metrics calculation
def run_agglomerative_metrics_euclidean(
    df_features: pd.DataFrame,
    id_vars: Iterable[str],
    k_values: Iterable[int] = range(2, 16),
    linkage: str = "ward",
    metric: str = "euclidean",
    logger: logging.Logger | None = None,
) -> pd.DataFrame:
    """Compute metrics for Ward (Euclidean) Agglomerative clustering across k."""
    log = logger or logging.getLogger("MAIN")

    feature_cols = [c for c in df_features.columns if c not in id_vars]
    X = df_features[feature_cols].values

    rows: List[AggloMetricsRow] = []
    for k in k_values:
        log.info("Agglomerative clustering (Ward/Euclidean) for k=%d ...", k)
        model = AgglomerativeClustering(n_clusters=k, linkage=linkage, metric=metric)
        labels = model.fit_predict(X)

        rows.append(
            AggloMetricsRow(
                k=k,
                silhouette=silhouette_score(X, labels),
                calinski_harabasz=calinski_harabasz_score(X, labels),
                davies_bouldin=davies_bouldin_score(X, labels),
            )
        )

    return pd.DataFrame([r.__dict__ for r in rows])

# Fisher's test for outcome association for Ward Agglomerative clusters
def run_fisher_test_agglomerative_euclidean(
    df_features: pd.DataFrame,
    id_vars: Iterable[str],
    k: int,
    outcome_df: pd.DataFrame,
    outcome_col: str = "target",
    linkage: str = "ward",
    metric: str = "euclidean",
) -> pd.DataFrame:
    """Fisher's exact test for Ward Agglomerative clusters."""
    feature_cols = [c for c in df_features.columns if c not in id_vars]
    X = df_features[feature_cols].values

    model = AgglomerativeClustering(n_clusters=k, linkage=linkage, metric=metric)
    labels = model.fit_predict(X)

    merged = (
        pd.DataFrame({"hadm_id": df_features["hadm_id"].values, "cluster": labels})
        .merge(outcome_df[["hadm_id", outcome_col]], on="hadm_id", how="left")
        .dropna(subset=[outcome_col])
    )

    # Calculate Fisher's test across clusters
    p_vals = []
    clusters = np.unique(labels)
    for cid in clusters:
        in_c = merged["cluster"] == cid
        target_in = merged.loc[in_c, outcome_col]
        target_out = merged.loc[~in_c, outcome_col]

        a = (target_in == 1).sum(); b = (target_out == 1).sum()
        c = (target_in == 0).sum(); d = (target_out == 0).sum()

        _, p = fisher_exact([[a, b], [c, d]], alternative="greater")
        p_vals.append(p)

    rejected, pvals_corr, _, _ = multipletests(p_vals, method="bonferroni")

    return pd.DataFrame({
        "Cluster": clusters,
        "Raw_p_value": p_vals,
        "Corrected_p_value": pvals_corr,
        "Significant_after_correction": rejected,
    })


# --------------------------------------------------------------------------
#   AGGLOMERATIVE CLUSTERING: MANHATTAN DISTANCE (using Unimputed vector)
# --------------------------------------------------------------------------

# Computes a custom Manhattan distance matrix between samples, handling missing values by only using overlapping non-NaN features. Uses a global covariance matrix (pairwise-complete) for feature scaling.
def _custom_manhattan_distance(X: np.ndarray) -> np.ndarray:
    """Pairwise Manhattan distance using only overlapping non-NaN features."""
    n = X.shape[0]
    D = np.full((n, n), np.nan)
    for i in range(n):
        xi = X[i]
        for j in range(i + 1, n):
            xj = X[j]
            mask = ~np.isnan(xi) & ~np.isnan(xj)
            if mask.any():
                D[i, j] = D[j, i] = np.abs(xi[mask] - xj[mask]).sum()
    np.fill_diagonal(D, 0.0)
    return D

def _fill_nan_in_distance(D: np.ndarray) -> np.ndarray:
    max_dist = np.nanmax(D)
    D_filled = np.where(np.isnan(D), max_dist + 1, D)
    np.fill_diagonal(D_filled, 0.0)
    return D_filled

# Computes metrics for agglomerative clustering using a precomputed Manhattan distance matrix.
def run_agglomerative_metrics_manhattan(
    df_raw: pd.DataFrame,
    id_vars: Iterable[str],
    k_values: Iterable[int] = range(2, 16),
    df_for_space: pd.DataFrame | None = None,
    logger: logging.Logger | None = None,
) -> tuple[pd.DataFrame, np.ndarray]:
    """
    Manhattan (precomputed) Agglomerative metrics.
    Returns (metrics_df, dist_matrix_filled).
    df_raw: no-imputation matrix (may contain NaNs)
    df_for_space: optional fully-imputed matrix for CH/DB
    """
    log = logger or logging.getLogger("MAIN")

    # Extract feature matrix with possible NaNs
    feature_cols_raw = [c for c in df_raw.columns if c not in id_vars]
    X_raw = df_raw[feature_cols_raw].values

    t0 = time.time()
    PCT_LABEL = f"{int(set_percentage*100)}pct" 

    # Path for the threshold-specific folder
    threshold_folder = os.path.join("distance_matrices", f"{PCT_LABEL}_threshold")
    if not os.path.exists(threshold_folder):
        os.makedirs(threshold_folder)

    # Save/load to this folder
    name = f"manhattan_distance_unimputed_{set_percentage*100:.0f}.npy"
    manh_path = os.path.join(threshold_folder, name)

    if os.path.exists(manh_path):
        log.info("Since Manhattan distance matrix on unimputed data for the given threshold is already computed, therfore loading cached Manhattan distance matrix from path %s\n", manh_path)
        D_filled = np.load(manh_path)
    else:
        log.info("Since Manhattan distance matrix on unimputed data for the given threshold is not stored in 'distance_matrices' folder, therefore Computing Manhattan distance matrix...\n")

        # Compute pairwise manhattan distance matrix, handling missing values if needed
        D = _custom_manhattan_distance(X_raw)
        D_filled = _fill_nan_in_distance(D)
        log.info("Manhattan distance computed in %.1f sec.\n", time.time() - t0)
        os.makedirs("distance_matrices", exist_ok=True)
        np.save(manh_path, D_filled)
        log.info(f"Saved Manhattan distance matrix to {manh_path}\n")

    # If imputed data provided, prepare for CH/DB scores
    X_space = None
    if df_for_space is not None:
        feat_cols_space = [c for c in df_for_space.columns if c not in id_vars]
        X_space = df_for_space[feat_cols_space].values

    rows: List[AggloMetricsRow] = []
    for k in k_values:
        log.info("Agglomerative clustering on unimputed data with Manhattan distance for k=%d ...", k)
        model = AgglomerativeClustering(n_clusters=k, metric="precomputed", linkage="average")
        labels = model.fit_predict(D_filled)

        sil = ch = db = np.nan
        if len(np.unique(labels)) > 1:
            sil = silhouette_score(D_filled, labels, metric="precomputed")
            if X_space is not None:
                ch = calinski_harabasz_score(X_space, labels)
                db = davies_bouldin_score(X_space, labels)

        rows.append(AggloMetricsRow(k=k, silhouette=sil, calinski_harabasz=ch, davies_bouldin=db))

    return pd.DataFrame([r.__dict__ for r in rows]), D_filled

# Performs Fisher's exact test to assess cluster-outcome association after agglomerative clustering using a precomputed Manhattan distance matrix.
def run_fisher_test_agglomerative_manhattan(
    dist_matrix_filled: np.ndarray,
    df_ids: pd.DataFrame,
    k: int,
    outcome_df: pd.DataFrame,
    outcome_col: str = "target",
) -> pd.DataFrame:
    """Fisher's Exact test for Agglomerative (Manhattan) clusters."""
    model = AgglomerativeClustering(n_clusters=k, metric="precomputed", linkage="average")
    labels = model.fit_predict(dist_matrix_filled)

    merged = (
        pd.DataFrame({"hadm_id": df_ids["hadm_id"].values, "cluster": labels})
        .merge(outcome_df[["hadm_id", outcome_col]], on="hadm_id", how="left")
        .dropna(subset=[outcome_col])
    )

    # Fisher's test per cluster
    p_vals = []
    clusters = np.unique(labels)
    for cid in clusters:
        in_c = merged["cluster"] == cid
        target_in = merged.loc[in_c, outcome_col]
        target_out = merged.loc[~in_c, outcome_col]

        a = (target_in == 1).sum(); b = (target_out == 1).sum()
        c = (target_in == 0).sum(); d = (target_out == 0).sum()

        _, p = fisher_exact([[a, b], [c, d]], alternative="greater")
        p_vals.append(p)

    rejected, pvals_corr, _, _ = multipletests(p_vals, method="bonferroni")
    return pd.DataFrame({
        "Cluster": clusters,
        "Raw_p_value": p_vals,
        "Corrected_p_value": pvals_corr,
        "Significant_after_correction": rejected,
    })


# ----------------------------------------------------------------------------------
# AGGLOMERATIVE CLUSTERING: MANHATTAN DISTANCE (using Unimputed Normalized vector)
# ----------------------------------------------------------------------------------

# Computes metrics for agglomerative clustering using a precomputed Manhattan distance matrix.
def run_agglomerative_metrics_manhattan_unimputed_normalized(
    df_raw: pd.DataFrame,
    id_vars: Iterable[str],
    k_values: Iterable[int] = range(2, 16),
    df_for_space: pd.DataFrame | None = None,   
    logger: logging.Logger | None = None,
) -> tuple[pd.DataFrame, np.ndarray]:
    """
    Manhattan (precomputed) Agglomerative metrics using UNIMPUTED NORMALIZED data.
    Returns: (metrics_df, filled_distance_matrix).

    df_raw: normalized but unimputed matrix (NaNs allowed)
    df_for_space: optional fully-imputed matrix for CH/DB metrics evaluation
    """
    log = logger or logging.getLogger("MAIN")

    # Extract features (with possible NaNs)
    feature_cols = [c for c in df_raw.columns if c not in id_vars]
    X_raw = df_raw[feature_cols].values

    t0 = time.time()
    PCT_LABEL = f"{int(set_percentage*100)}pct"

    # Path for the threshold-specific folder
    threshold_folder = os.path.join("distance_matrices", f"{PCT_LABEL}_threshold")
    if not os.path.exists(threshold_folder):
        os.makedirs(threshold_folder)

    # Save/load to this folder
    name = f"manhattan_distance_unimputed_normalized_{set_percentage*100:.0f}.npy"
    manh_path = os.path.join(threshold_folder, name)
    
    # Load from cache if exists, else compute
    if os.path.exists(manh_path):
        log.info("Since Manhattan distance matrix on normalized unimputed data for the given threshold is already computed, therfore loading cached Manhattan distance matrix from path %s\n", manh_path)
        D_filled = np.load(manh_path)
    else:
        log.info("Since Manhattan distance matrix on normalized unimputed data for the given threshold is not stored in 'distance_matrices' folder, therefore Computing Manhattan distance matrix...\n")
        D = _custom_manhattan_distance(X_raw)
        D_filled = _fill_nan_in_distance(D)
        log.info("Manhattan distance computed in %.1f sec", time.time() - t0)
        np.save(manh_path, D_filled)
        log.info(f"Saved Manhattan distance matrix to {manh_path}")

    # If imputed data provided, prepare for CH/DB scores
    X_space = None
    if df_for_space is not None:
        feat_cols_space = [c for c in df_for_space.columns if c not in id_vars]
        X_space = df_for_space[feat_cols_space].values

    # Run clustering and compute metrics for each k
    rows: List[AggloMetricsRow] = []
    for k in k_values:
        log.info("Agglomerative clustering (Manhattan, unimputed normalized) for k=%d", k)
        model = AgglomerativeClustering(n_clusters=k, metric="precomputed", linkage="average")
        labels = model.fit_predict(D_filled)

        sil = ch = db = np.nan
        if len(np.unique(labels)) > 1:
            sil = silhouette_score(D_filled, labels, metric="precomputed")
            if X_space is not None:
                ch = calinski_harabasz_score(X_space, labels)
                db = davies_bouldin_score(X_space, labels)

        rows.append(AggloMetricsRow(k=k, silhouette=sil, calinski_harabasz=ch, davies_bouldin=db))

    return pd.DataFrame([r.__dict__ for r in rows]), D_filled


# --------------------------------------------------------------------------
# AGGLOMERATIVE CLUSTERING: MANHATTAN DISTANCE (using Imputed vector)
# --------------------------------------------------------------------------

# Computes metrics for agglomerative clustering using a precomputed Manhattan distance matrix.
def run_agglomerative_metrics_manhattan_imputed(
    df_raw: pd.DataFrame,
        id_vars: Iterable[str],
        k_values: Iterable[int] = range(2, 16),
        df_for_space: pd.DataFrame | None = None,
        logger: logging.Logger | None = None,
   ) -> tuple[pd.DataFrame, np.ndarray]:
        """Agglomerative Clustering metrics with Manhattan distance on fully imputed data."""
        log = logger or logging.getLogger("MAIN")
     
        # Extract feature matrix with possible NaNs
        feature_cols_raw = [c for c in df_raw.columns if c not in id_vars]
        X_raw = df_raw[feature_cols_raw].values
    
        t0 = time.time()
        PCT_LABEL = f"{int(set_percentage*100)}pct" 
        
        # Path for the threshold-specific folder
        threshold_folder = os.path.join("distance_matrices", f"{PCT_LABEL}_threshold")
        if not os.path.exists(threshold_folder):
            os.makedirs(threshold_folder)

        # Save/load to this folder
        name = f"manhattan_distance_imputed_{100*set_percentage:.0f}.npy"
        manh_path = os.path.join(threshold_folder, name)

        if os.path.exists(manh_path):
            log.info("Since Manhattan distance matrix on imputed data for the given threshold is already computed, therfore loading cached Manhattan distance matrix from path %s\n", manh_path)
            D_filled = np.load(manh_path)
        else:
            log.info("Since Manhattan distance matrix on imputed data for the given threshold is not stored in 'distance_matrices' folder, therefore Computing Manhattan distance matrix...\n")
            # If data is fully imputed, NaNs should theoretically not be present,we may use sklearn's efficient pairwise_distances
            D = pairwise_distances(X_raw, metric='manhattan')
            D_filled = D
            os.makedirs("distance_matrices", exist_ok=True)
            np.save(manh_path, D_filled)
            log.info("Saved Manhattan distance matrix to %s", manh_path)

        # If imputed data provided, prepare for CH/DB scores
        X_space = None
        if df_for_space is not None:
            feat_cols_space = [c for c in df_for_space.columns if c not in id_vars]
            X_space = df_for_space[feat_cols_space].values
    
        rows: List = []
        for k in k_values:
            log.info("Agglomerative clustering on imputed data with Manhattan distance for k=%d", k)
            model = AgglomerativeClustering(n_clusters=k, metric="precomputed", linkage="average")
            labels = model.fit_predict(D_filled)
    
            sil = ch = db = np.nan
            if len(np.unique(labels)) > 1:
                sil = silhouette_score(D_filled, labels, metric="precomputed")
                if X_space is not None:
                    ch = calinski_harabasz_score(X_space, labels)
                    db = davies_bouldin_score(X_space, labels)
            rows.append(dict(k=k, silhouette=sil, calinski_harabasz=ch, davies_bouldin=db))
    
        return pd.DataFrame(rows), D_filled

# Performs Fisher's exact test to assess cluster-outcome association after agglomerative clustering using a precomputed Manhattan distance matrix.
def run_fisher_test_agglomerative_manhattan_imputed(
    dist_matrix_filled: np.ndarray,
    df_ids: pd.DataFrame,
    k: int,
    outcome_df: pd.DataFrame,
    outcome_col: str = "target",
) -> pd.DataFrame:
    """Run Fisher's Exact test on clusters from Manhattan distance agglomerative clustering."""
    model = AgglomerativeClustering(n_clusters=k, metric="precomputed", linkage="average")
    labels = model.fit_predict(dist_matrix_filled)

    merged = (
        pd.DataFrame({"hadm_id": df_ids["hadm_id"].values, "cluster": labels})
        .merge(outcome_df[["hadm_id", outcome_col]], on="hadm_id", how="left")
        .dropna(subset=[outcome_col])
    )

    p_vals = []
    clusters = np.unique(labels)
    for cid in clusters:
        in_c = merged["cluster"] == cid
        target_in = merged.loc[in_c, outcome_col]
        target_out = merged.loc[~in_c, outcome_col]
        a = (target_in == 1).sum(); b = (target_out == 1).sum()
        c = (target_in == 0).sum(); d = (target_out == 0).sum()
        _, p = fisher_exact([[a, b], [c, d]], alternative="greater")
        p_vals.append(p)

    rejected, pvals_corr, _, _ = multipletests(p_vals, method="bonferroni")
    return pd.DataFrame({
        "Cluster": clusters,
        "Raw_p_value": p_vals,
        "Corrected_p_value": pvals_corr,
        "Significant_after_correction": rejected,
    })


# ------------------------------------------------------------------------
# AGGLOMERATIVE CLUSTERING: MAHALANOBIS DISTANCE (using Unimputed vector)
# ------------------------------------------------------------------------

# Computes a custom Mahalanobis distance matrix between samples, handling missing values by only using overlapping non-NaN features. Uses a global covariance matrix (pairwise-complete) for feature scaling.
def _custom_mahalanobis_distance(X: np.ndarray, cov_global: np.ndarray) -> np.ndarray:
    """
    Pairwise Mahalanobis distance using only overlapping non-NaN features.
    cov_global: full covariance matrix (pairwise-complete) over all features.
    """
    n = X.shape[0]
    D = np.full((n, n), np.nan)
    for i in range(n):
        xi = X[i]
        for j in range(i + 1, n):
            xj = X[j]
            mask = ~np.isnan(xi) & ~np.isnan(xj)
            n_overlap = mask.sum()
            if n_overlap == 0:
                continue

            diff = xi[mask] - xj[mask]
            sub_cov = cov_global[mask][:, mask]

            if n_overlap == 1:
                denom = np.sqrt(sub_cov[0, 0]) if sub_cov[0, 0] > 0 else 1.0
                dist = np.abs(diff[0]) / denom
            else:
                try:
                    inv_cov = inv(sub_cov)
                except LinAlgError:
                    inv_cov = pinv(sub_cov)
                dist = np.sqrt(diff @ inv_cov @ diff)

            D[i, j] = D[j, i] = dist

    np.fill_diagonal(D, 0.0)
    return D

# Computes metrics for agglomerative clustering using a precomputed Mahalanobis distance matrix.
def run_agglomerative_metrics_mahalanobis(
    df_raw: pd.DataFrame,
    id_vars: Iterable[str],
    k_values: Iterable[int] = range(2, 16),
    df_for_space: pd.DataFrame | None = None,
    logger: logging.Logger | None = None,
) -> tuple[pd.DataFrame, np.ndarray]:
    """
    Mahalanobis (precomputed) Agglomerative metrics.
    Returns (metrics_df, dist_matrix_filled).
    df_raw: no-imputation matrix (can contain NaNs)
    df_for_space: optional fully-imputed matrix for CH/DB
    """
    log = logger or logging.getLogger("MAIN")

    # Extract feature matrix with possible NaNs
    feat_cols_raw = [c for c in df_raw.columns if c not in id_vars]
    X_raw = df_raw[feat_cols_raw].values

    t0 = time.time()

    PCT_LABEL = f"{int(set_percentage*100)}pct" 
    # Path for the threshold-specific folder
    threshold_folder = os.path.join("distance_matrices", f"{PCT_LABEL}_threshold")
    if not os.path.exists(threshold_folder):
        os.makedirs(threshold_folder)
    # Save/load to this folder
    name = f"Mahalanobis_distance_unimputed_{set_percentage*100:.0f}.npy"
    mahalanobis_path = os.path.join(threshold_folder, name)

    if os.path.exists(mahalanobis_path):
        log.info("Since Mahalanobis distance matrix on unimputed data for the given threshold is already computed, therfore loading cached Mahalanobis distance matrix from path %s\n", mahalanobis_path)
        D_filled = np.load(mahalanobis_path)
    else:
        log.info("Since Mahalanobis distance matrix on unimputed data for the given threshold is not stored in 'distance_matrices' folder, therefore Computing Mahalanobis distance matrix...\n")

        # Compute global covariance and distance matrix
        cov_global = pd.DataFrame(X_raw).cov().values
        D = _custom_mahalanobis_distance(X_raw, cov_global)
        D_filled = _fill_nan_in_distance(D)
        log.info("Mahalanobis distance computed in %.1f sec.", time.time() - t0)
        os.makedirs("distance_matrices", exist_ok=True)
        np.save(mahalanobis_path, D_filled)
        log.info(f"Saved Mahalanobis distance matrix to {mahalanobis_path}")

    # If imputed data provided, prepare for CH/DB scores
    X_space = None
    if df_for_space is not None:
        feat_cols_space = [c for c in df_for_space.columns if c not in id_vars]
        X_space = df_for_space[feat_cols_space].values

    rows: List[AggloMetricsRow] = []
    for k in k_values:
        log.info("Agglomerative clustering on unimputed data with Mahalanobis distance for k=%d ...", k)
        model = AgglomerativeClustering(n_clusters=k, metric="precomputed", linkage="average")
        labels = model.fit_predict(D_filled)

        sil = ch = db = np.nan
        if len(np.unique(labels)) > 1:
            sil = silhouette_score(D_filled, labels, metric="precomputed")
            if X_space is not None:
                ch = calinski_harabasz_score(X_space, labels)
                db = davies_bouldin_score(X_space, labels)

        rows.append(AggloMetricsRow(k=k, silhouette=sil, calinski_harabasz=ch, davies_bouldin=db))

    return pd.DataFrame([r.__dict__ for r in rows]), D_filled

# Performs Fisher's exact test to assess cluster-outcome association after agglomerative clustering using a precomputed Mahalanobis distance matrix.
def run_fisher_test_agglomerative_mahalanobis(
    dist_matrix_filled: np.ndarray,
    df_ids: pd.DataFrame,   # must contain hadm_id in same order as dist rows
    k: int,
    outcome_df: pd.DataFrame,
    outcome_col: str = "target",
) -> pd.DataFrame:
    model = AgglomerativeClustering(n_clusters=k, metric="precomputed", linkage="average")
    labels = model.fit_predict(dist_matrix_filled)

    merged = (
        pd.DataFrame({"hadm_id": df_ids["hadm_id"].values, "cluster": labels})
        .merge(outcome_df[["hadm_id", outcome_col]], on="hadm_id", how="left")
        .dropna(subset=[outcome_col])
    )

    # Fisher's test per cluster
    p_vals = []
    clusters = np.unique(labels)
    for cid in clusters:
        in_c = merged["cluster"] == cid
        target_in = merged.loc[in_c, outcome_col]
        target_out = merged.loc[~in_c, outcome_col]

        a = (target_in == 1).sum(); b = (target_out == 1).sum()
        c = (target_in == 0).sum(); d = (target_out == 0).sum()

        _, p = fisher_exact([[a, b], [c, d]], alternative="greater")
        p_vals.append(p)

    rejected, pvals_corr, _, _ = multipletests(p_vals, method="bonferroni")
    return pd.DataFrame({
        "Cluster": clusters,
        "Raw_p_value": p_vals,
        "Corrected_p_value": pvals_corr,
        "Significant_after_correction": rejected,
    })


# --------------------------------------------------------------------------------------
#  AGGLOMERATIVE CLUSTERING: MAHALANOBIS DISTANCE (using Unimputed Normalized vector)
# --------------------------------------------------------------------------------------

# Computes metrics for agglomerative clustering using a precomputed Mahalanobis distance matrix.
def run_agglomerative_metrics_mahalnobis_unimputed_normalized(
    df_raw: pd.DataFrame,
    id_vars: Iterable,
    k_values: Iterable[int] = range(2, 16),
    df_for_space: pd.DataFrame | None = None,
    logger: logging.Logger | None = None
) -> tuple[pd.DataFrame, np.ndarray]:
    """
    Agglomerative clustering with Mahalanobis distance on unimputed but normalized data.
    Similar to 'run_agglomerative_metrics_mahalanobis' but uses normalized data and caches separately.
    """
    log = logger or logging.getLogger("MAIN")
    
    # Extract feature matrix with possible NaNs
    feat_cols_raw = [c for c in df_raw.columns if c not in id_vars]
    X_raw = df_raw[feat_cols_raw].values

    t0 = time.time()
    PCT_LABEL = f"{int(set_percentage*100)}pct"

    # Path for the threshold-specific folder
    threshold_folder = os.path.join("distance_matrices", f"{PCT_LABEL}_threshold")
    if not os.path.exists(threshold_folder):
        os.makedirs(threshold_folder)

    # Save/load to this folder
    file_name = f"Mahalanobis_distance_unimputed_normalized_{int(set_percentage*100)}.npy"
    mahalanobis_path = os.path.join(threshold_folder, file_name)

    if os.path.exists(mahalanobis_path):
        log.info("Since Mahalanobis distance matrix on normalized unimputed data for the given threshold is already computed, therfore loading cached Mahalanobis distance matrix from path %s\n", mahalanobis_path)
        D_filled = np.load(mahalanobis_path)
    else:
        log.info("Since Mahalanobis distance matrix on normalized unimputed data for the given threshold is not stored in 'distance_matrices' folder, therefore Computing Mahalanobis distance matrix...\n")
        
        # Compute global covariance and distance matrix
        cov_global = pd.DataFrame(X_raw).cov().values
        D = _custom_mahalanobis_distance(X_raw, cov_global)
        D_filled = _fill_nan_in_distance(D)
        log.info("Mahalanobis distance computed in %.1f sec.", time.time() - t0)
        os.makedirs("distance_matrices", exist_ok=True)
        np.save(mahalanobis_path, D_filled)
        log.info(f"Saved Mahalanobis distance matrix to {mahalanobis_path}")

    # If imputed data provided, prepare for CH/DB scores
    X_space = None
    if df_for_space is not None:
        feat_cols_space = [c for c in df_for_space.columns if c not in id_vars]
        X_space = df_for_space[feat_cols_space].values

    rows: List[AggloMetricsRow] = []
    for k in k_values:
        log.info("Mahalanobis Agglomerative clustering using normalized unimputed vector for k=%d ...", k)
        model = AgglomerativeClustering(n_clusters=k, metric="precomputed", linkage="average")
        labels = model.fit_predict(D_filled)

        sil = ch = db = np.nan
        if len(np.unique(labels)) > 1:
            sil = silhouette_score(D_filled, labels, metric="precomputed")
            if X_space is not None:
                ch = calinski_harabasz_score(X_space, labels)
                db = davies_bouldin_score(X_space, labels)

        rows.append(AggloMetricsRow(k=k, silhouette=sil, calinski_harabasz=ch, davies_bouldin=db))

    return pd.DataFrame([r.__dict__ for r in rows]), D_filled


# ------------------------------------------------------------------------
# AGGLOMERATIVE CLUSTERING: MAHALANOBIS DISTANCE (using Imputed vector)
# ------------------------------------------------------------------------

# Computes a custom Mahalanobis distance matrix between samples, handling missing values by only using overlapping non-NaN features. Uses a global covariance matrix (pairwise-complete) for feature scaling.
def _custom_mahalanobis_distance(X: np.ndarray, cov_global: np.ndarray) -> np.ndarray:
    """
    Pairwise Mahalanobis distance using all features (no NaNs assumed here).
    """
    n = X.shape[0]
    D = np.zeros((n, n))
    try:
        inv_cov = inv(cov_global)
    except LinAlgError:
        inv_cov = pinv(cov_global)

    for i in range(n):
        xi = X[i]
        for j in range(i + 1, n):
            xj = X[j]
            diff = xi - xj
            dist = np.sqrt(diff @ inv_cov @ diff)
            D[i, j] = D[j, i] = dist
    np.fill_diagonal(D, 0.0)
    return D

def run_agglomerative_metrics_mahalanobis_imupted(
    df_raw: pd.DataFrame,
    id_vars: Iterable[str],
    k_values: Iterable[int] = range(2, 16),
    df_for_space: pd.DataFrame | None = None,
    logger: logging.Logger | None = None,
) -> tuple[pd.DataFrame, np.ndarray]:
    """
    Agglomerative clustering metrics using precomputed Mahalanobis distance matrix on imputed data.
    """
    log = logger or logging.getLogger("MAIN")

    # Extract feature matrix with possible NaNs
    feat_cols = [c for c in df_raw.columns if c not in id_vars]
    X = df_raw[feat_cols].values

    t0 = time.time()
    PCT_LABEL = f"{int(set_percentage*100)}pct" 

    # Path for the threshold-specific folder
    threshold_folder = os.path.join("distance_matrices", f"{PCT_LABEL}_threshold")
    if not os.path.exists(threshold_folder):
        os.makedirs(threshold_folder)

    # Save/load to this folder
    filename = f"Mahalanobis_distance_unimputed_{set_percentage*100:.0f}.npy"   
    filepath = os.path.join(threshold_folder, filename)

    if os.path.exists(filepath):
        log.info("Since Mahalanobis distance matrix on imputed data for the given threshold is already computed, therfore loading cached Mahalanobis distance matrix from path %s\n", filepath)
        D = np.load(filepath)
    else:
        log.info("Since Mahalanobis distance matrix on imputed data for the given threshold is not stored in 'distance_matrices' folder, therefore Computing Mahalanobis distance matrix...\n")
        cov_global = np.cov(X, rowvar=False)
        D = _custom_mahalanobis_distance(X, cov_global)
        os.makedirs("distance_matrices", exist_ok=True)
        np.save(filepath, D)
        log.info(f"Saved Mahalanobis distance matrix to {filepath}")

    # If imputed data provided, prepare for CH/DB scores
    X_space = None
    if df_for_space is not None:
        feat_cols_space = [c for c in df_for_space.columns if c not in id_vars]
        X_space = df_for_space[feat_cols_space].values

    results = []
    for k in k_values:
        log.info("Agglomerative clustering on imputed data with Mahalanobis distance for k=%d ...", k)
        model = AgglomerativeClustering(n_clusters=k, metric="precomputed", linkage="average")
        labels = model.fit_predict(D)

        sil = ch = db = np.nan
        if len(np.unique(labels)) > 1:
            sil = silhouette_score(D, labels, metric="precomputed")
            if X_space is not None:
                ch = calinski_harabasz_score(X_space, labels)
                db = davies_bouldin_score(X_space, labels)
        results.append(dict(k=k, silhouette=sil, calinski_harabasz=ch, davies_bouldin=db))

    return pd.DataFrame(results), D

def run_fisher_test_agglomerative_mahalanobis_imputed(
    dist_matrix_filled: np.ndarray,
    df_ids: pd.DataFrame,
    k: int,
    outcome_df: pd.DataFrame,
    outcome_col: str = "target",
) -> pd.DataFrame:
    model = AgglomerativeClustering(n_clusters=k, metric="precomputed", linkage="average")
    labels = model.fit_predict(dist_matrix_filled)

    merged = (
        pd.DataFrame({"hadm_id": df_ids["hadm_id"].values, "cluster": labels})
        .merge(outcome_df[["hadm_id", outcome_col]], on="hadm_id", how="left")
        .dropna(subset=[outcome_col])
    )

    p_vals = []
    clusters = np.unique(labels)
    for cid in clusters:
        in_c = merged["cluster"] == cid
        target_in = merged.loc[in_c, outcome_col]
        target_out = merged.loc[~in_c, outcome_col]

        a = (target_in == 1).sum(); b = (target_out == 1).sum()
        c = (target_in == 0).sum(); d = (target_out == 0).sum()

        _, p = fisher_exact([[a, b], [c, d]], alternative="greater")
        p_vals.append(p)

    rejected, pvals_corr, _, _ = multipletests(p_vals, method="bonferroni")
    return pd.DataFrame({
        "Cluster": clusters,
        "Raw_p_value": p_vals,
        "Corrected_p_value": pvals_corr,
        "Significant_after_correction": rejected,
    })


# -------------------------------------------------------------------
# AGGLOMERATIVE CLUSTERING: COSINE DISTANCE (Using Unimputed vector)
# -------------------------------------------------------------------

# Computes a cosine distance matrix accounting for missing data, using only overlapping features for each pair of samples.
def _custom_cosine_distance(X: np.ndarray) -> np.ndarray:
    """
    Cosine distance using only overlapping non-NaN features.
    Returns a symmetric matrix with NaNs where no overlap exists.
    """
    n = X.shape[0]
    D = np.full((n, n), np.nan)
    for i in range(n):
        xi = X[i]
        for j in range(i + 1, n):
            xj = X[j]
            mask = ~np.isnan(xi) & ~np.isnan(xj)
            if not mask.any():
                continue

            vi, vj = xi[mask], xj[mask]
            ni, nj = np.linalg.norm(vi), np.linalg.norm(vj)
            if ni > 0 and nj > 0:
                cos_sim = np.clip(np.dot(vi, vj) / (ni * nj), -1.0, 1.0)
                dist = 1.0 - cos_sim
            else:
                dist = 1.0   # zero norm => treat as max distance
            D[i, j] = D[j, i] = max(0.0, dist)

    np.fill_diagonal(D, 0.0)
    return D


# Computes metrics for agglomerative clustering using a precomputed cosine distance matrix.
def run_agglomerative_metrics_cosine(
    df_raw: pd.DataFrame,
    id_vars: Iterable[str],
    k_values: Iterable[int] = range(2, 16),
    df_for_space: pd.DataFrame | None = None,
    logger: logging.Logger | None = None,
) -> tuple[pd.DataFrame, np.ndarray]:
    """
    Cosine (precomputed) Agglomerative metrics.
    Returns (metrics_df, dist_matrix_filled).
    df_raw: matrix WITH NaNs (no imputation)
    df_for_space: optional imputed matrix for CH/DB
    """
    log = logger or logging.getLogger("MAIN")

    # Extract feature matrix with possible NaNs
    feat_cols_raw = [c for c in df_raw.columns if c not in id_vars]
    X_raw = df_raw[feat_cols_raw].values

    t0 = time.time()
    PCT_LABEL = f"{int(set_percentage*100)}pct" 

    # Path for the threshold-specific folder
    threshold_folder = os.path.join("distance_matrices", f"{PCT_LABEL}_threshold")
    if not os.path.exists(threshold_folder):
        os.makedirs(threshold_folder)

    # Save/load to this folder
    name = f"cosine_distance_unimputed_{set_percentage*100:.0f}.npy"
    cosine_path = os.path.join(threshold_folder, name)

    if os.path.exists(cosine_path):
        log.info("Since Cosine distance matrix on unimputed data for the given threshold is already computed, therfore loading cached Cosine distance matrix from path %s\n", cosine_path)
        D_filled = np.load(cosine_path)
    else:
        log.info("Since Cosine distance matrix on unimputed data for the given threshold is not stored in 'distance_matrices' folder, therefore Computing Cosine distance matrix...\n")
        D = _custom_cosine_distance(X_raw)
        D_filled = _fill_nan_in_distance(D)          # reuse helper
        D_filled = np.abs(D_filled)                  # safety: non-negative
        log.info("Cosine distance computed in %.1f sec.", time.time() - t0)
        os.makedirs("distance_matrices", exist_ok=True)
        np.save(cosine_path, D_filled)
        log.info(f"Saved Cosine distance matrix to {cosine_path}")

    # If imputed data provided, prepare for CH/DB scores
    X_space = None
    if df_for_space is not None:
        feat_cols_space = [c for c in df_for_space.columns if c not in id_vars]
        X_space = df_for_space[feat_cols_space].values

    rows: List[AggloMetricsRow] = []
    for k in k_values:
        log.info("Agglomerative clustering on unimputed data with Cosine distance for k=%d ...", k)
        model = AgglomerativeClustering(n_clusters=k, metric="precomputed", linkage="average")
        labels = model.fit_predict(D_filled)

        sil = ch = db = np.nan
        if len(np.unique(labels)) > 1:
            sil = silhouette_score(D_filled, labels, metric="precomputed")
            if X_space is not None:
                ch = calinski_harabasz_score(X_space, labels)
                db = davies_bouldin_score(X_space, labels)

        rows.append(AggloMetricsRow(k=k, silhouette=sil, calinski_harabasz=ch, davies_bouldin=db))

    return pd.DataFrame([r.__dict__ for r in rows]), D_filled

# Performs Fisher's exact test to assess cluster-outcome association after agglomerative clustering using a cosine distance matrix.
def run_fisher_test_agglomerative_cosine(
    dist_matrix_filled: np.ndarray,
    df_ids: pd.DataFrame,
    k: int,
    outcome_df: pd.DataFrame,
    outcome_col: str = "target",
) -> pd.DataFrame:
    """
    Fisher's test for Agglomerative (Cosine/precomputed) clusters.
    df_ids must contain 'hadm_id' in same order as rows of dist_matrix_filled.
    """
    model = AgglomerativeClustering(n_clusters=k, metric="precomputed", linkage="average")
    labels = model.fit_predict(dist_matrix_filled)

    merged = (
        pd.DataFrame({"hadm_id": df_ids["hadm_id"].values, "cluster": labels})
        .merge(outcome_df[["hadm_id", outcome_col]], on="hadm_id", how="left")
        .dropna(subset=[outcome_col])
    )

    p_vals = []
    clusters = np.unique(labels)
    for cid in clusters:
        in_c = merged["cluster"] == cid
        target_in = merged.loc[in_c, outcome_col]
        target_out = merged.loc[~in_c, outcome_col]

        a = (target_in == 1).sum(); b = (target_out == 1).sum()
        c = (target_in == 0).sum(); d = (target_out == 0).sum()

        _, p = fisher_exact([[a, b], [c, d]], alternative="greater")
        p_vals.append(p)

    rejected, pvals_corr, _, _ = multipletests(p_vals, method="bonferroni")
    return pd.DataFrame({
        "Cluster": clusters,
        "Raw_p_value": p_vals,
        "Corrected_p_value": pvals_corr,
        "Significant_after_correction": rejected,
    })


# -------------------------------------------------------------------------------
# AGGLOMERATIVE CLUSTERING: COSINE DISTANCE (Using Unimputed Normalized vector)
# -------------------------------------------------------------------------------

# Computes a cosine distance matrix accounting for missing data, using only overlapping features for each pair of samples.
def run_agglomerative_metrics_cosine_unimputed_normalized(
    df_raw: pd.DataFrame,
    id_vars: Iterable[str],
    k_values: Iterable[int] = range(2, 16),
    df_for_space: pd.DataFrame | None = None,
    logger: logging.Logger | None = None,
) -> tuple[pd.DataFrame, np.ndarray]:
    """
    Cosine (precomputed) Agglomerative metrics using UNIMPUTED NORMALIZED data.
    Returns (metrics_df, dist_matrix_filled).

    df_raw: normalized but unimputed matrix (can contain NaNs)
    df_for_space: optional fully imputed matrix for CH/DB scores
    """
    log = logger or logging.getLogger("MAIN")

    # Extract feature matrix with possible NaNs
    feat_cols_raw = [c for c in df_raw.columns if c not in id_vars]
    X_raw = df_raw[feat_cols_raw].values

    t0 = time.time()
    PCT_LABEL = f"{int(set_percentage*100)}pct"

    # Path for the threshold-specific folder
    threshold_folder = os.path.join("distance_matrices", f"{PCT_LABEL}_threshold")
    if not os.path.exists(threshold_folder):
        os.makedirs(threshold_folder)

    # Save/load to this folder
    name = f"cosine_distance_unimputed_normalized_{set_percentage*100:.0f}.npy"
    cosine_path = os.path.join(threshold_folder, name)

    if os.path.exists(cosine_path):
        log.info("Since Cosine distance matrix on normalized unimputed data for the given threshold is already computed, therfore loading cached Cosine distance matrix from path %s\n", cosine_path)
        D_filled = np.load(cosine_path)
    else:
        log.info("Since Cosine distance matrix on normalized unimputed data for the given threshold is not stored in 'distance_matrices' folder, therefore Computing Cosine distance matrix...\n")
        D = _custom_cosine_distance(X_raw)
        D_filled = _fill_nan_in_distance(D)
        D_filled = np.abs(D_filled)  # ensure non-negative
        log.info(f"Computed Cosine distance in {time.time() - t0:.1f} sec.")
        np.save(cosine_path, D_filled)
        log.info(f"Saved Cosine distance matrix to {cosine_path}")

    # If imputed data provided, prepare for CH/DB scores
    X_space = None
    if df_for_space is not None:
        feat_cols_space = [c for c in df_for_space.columns if c not in id_vars]
        X_space = df_for_space[feat_cols_space].values

    rows: List[AggloMetricsRow] = []
    for k in k_values:
        log.info(f"Agglomerative clustering on unimputed normalized data with Cosine distance for k={k} ...")
        model = AgglomerativeClustering(n_clusters=k, metric="precomputed", linkage="average")
        labels = model.fit_predict(D_filled)

        sil = ch = db = np.nan
        if len(np.unique(labels)) > 1:
            sil = silhouette_score(D_filled, labels, metric="precomputed")
            if X_space is not None:
                ch = calinski_harabasz_score(X_space, labels)
                db = davies_bouldin_score(X_space, labels)

        rows.append(AggloMetricsRow(k=k, silhouette=sil, calinski_harabasz=ch, davies_bouldin=db))

    return pd.DataFrame([r.__dict__ for r in rows]), D_filled


# -------------------------------------------------------------------
# AGGLOMERATIVE CLUSTERING: COSINE DISTANCE (using Imputed vector)
# -------------------------------------------------------------------

# Computes metrics for agglomerative clustering using a precomputed cosine distance matrix.
def run_agglomerative_metrics_cosine_imputed(
    df_raw: pd.DataFrame,
    id_vars: Iterable[str],
    k_values: Iterable[int] = range(2, 16),
    df_for_space: pd.DataFrame | None = None,
    logger: logging.Logger | None = None,
) -> tuple[pd.DataFrame, np.ndarray]:
    """
    Agglomerative clustering metrics using precomputed Cosine distance matrix.
    df_raw: DataFrame with features (can contain NaNs or be fully imputed).
    df_for_space: optional DataFrame with imputed features for CH/DB scores.
    """
    log = logger or logging.getLogger("MAIN")

    feat_cols_raw = [c for c in df_raw.columns if c not in id_vars]
    X_raw = df_raw[feat_cols_raw].values

    t0 = time.time()
    PCT_LABEL = f"{int(set_percentage*100)}pct" 

    # Path for the threshold-specific folder
    threshold_folder = os.path.join("distance_matrices", f"{PCT_LABEL}_threshold")
    if not os.path.exists(threshold_folder):
        os.makedirs(threshold_folder)

    # Save/load to this folder
    name = f"cosine_distance_imputed_{set_percentage*100:.0f}.npy"
    cosine_path = os.path.join(threshold_folder, name)

    if os.path.exists(cosine_path):
        log.info("Since Cosine distance matrix on imputed data for the given threshold is already computed, therfore loading cached Cosine distance matrix from path %s\n", cosine_path)
        D_filled = np.load(cosine_path)
    else:
        log.info("Since Cosine distance matrix on imputed data for the given threshold is not stored in 'distance_matrices' folder, therefore Computing Cosine distance matrix...\n")
        D = _custom_cosine_distance(X_raw)
        D_filled = _fill_nan_in_distance(D)
        D_filled = np.abs(D_filled)  # ensure non-negative
        os.makedirs("distance_matrices", exist_ok=True)
        np.save(cosine_path, D_filled)
        log.info("Saved Cosine distance matrix (imputed) to %s", cosine_path)

    # If imputed data provided, prepare for CH/DB scores
    X_space = None
    if df_for_space is not None:
        feat_cols_space = [c for c in df_for_space.columns if c not in id_vars]
        X_space = df_for_space[feat_cols_space].values

    rows: List[AggloMetricsRow] = []
    for k in k_values:
        log.info("Agglomerative clustering on imputed data with Cosine distance for k=%d ...", k)
        model = AgglomerativeClustering(n_clusters=k, metric="precomputed", linkage="average")
        labels = model.fit_predict(D_filled)

        sil = ch = db = np.nan
        if len(np.unique(labels)) > 1:
            sil = silhouette_score(D_filled, labels, metric="precomputed")
            if X_space is not None:
                ch = calinski_harabasz_score(X_space, labels)
                db = davies_bouldin_score(X_space, labels)

        rows.append(AggloMetricsRow(k=k, silhouette=sil, calinski_harabasz=ch, davies_bouldin=db))

    return pd.DataFrame([r.__dict__ for r in rows]), D_filled
    
# Performs Fisher's exact test to assess cluster-outcome association after agglomerative clustering using a cosine distance matrix.
def run_fisher_test_agglomerative_cosine_imputed(
    dist_matrix_filled: np.ndarray,
    df_ids: pd.DataFrame,
    k: int,
    outcome_df: pd.DataFrame,
    outcome_col: str = "target",
) -> pd.DataFrame:
    """
    Fisher's test for Agglomerative (Cosine/precomputed) clusters.
    df_ids must contain 'hadm_id' in same order as rows of dist_matrix_filled.
    """
    model = AgglomerativeClustering(n_clusters=k, metric="precomputed", linkage="average")
    labels = model.fit_predict(dist_matrix_filled)

    merged = (
        pd.DataFrame({"hadm_id": df_ids["hadm_id"].values, "cluster": labels})
        .merge(outcome_df[["hadm_id", outcome_col]], on="hadm_id", how="left")
        .dropna(subset=[outcome_col])
    )

    p_vals = []
    clusters = np.unique(labels)
    for cid in clusters:
        in_c = merged["cluster"] == cid
        target_in = merged.loc[in_c, outcome_col]
        target_out = merged.loc[~in_c, outcome_col]

        a = (target_in == 1).sum(); b = (target_out == 1).sum()
        c = (target_in == 0).sum(); d = (target_out == 0).sum()

        _, p = fisher_exact([[a, b], [c, d]], alternative="greater")
        p_vals.append(p)

    rejected, pvals_corr, _, _ = multipletests(p_vals, method="bonferroni")
    return pd.DataFrame({
        "Cluster": clusters,
        "Raw_p_value": p_vals,
        "Corrected_p_value": pvals_corr,
        "Significant_after_correction": rejected,
    })
    

# ------------------------------------------------------------------
#    AGGLOMERATIVE CLUSTERING: EUCLIDEAN  (using Unimputed vector)
# ------------------------------------------------------------------

# Computes a Euclidean distance matrix using only features present for both samples.
def _custom_euclidean_distance(X: np.ndarray) -> np.ndarray:
    """Euclidean distance using only overlapping non-NaN features."""
    n = X.shape[0]
    D = np.full((n, n), np.nan)
    for i in range(n):
        xi = X[i]
        for j in range(i + 1, n):
            xj = X[j]
            mask = ~np.isnan(xi) & ~np.isnan(xj)
            if mask.any():
                D[i, j] = D[j, i] = np.linalg.norm(xi[mask] - xj[mask])
    np.fill_diagonal(D, 0.0)
    return D

# Calculates and evaluates agglomerative clustering with a precomputed Euclidean distance matrix, useful for handling missing data without imputation.
def run_agglomerative_metrics_euclidean_precomputed(
    df_raw: pd.DataFrame,
    id_vars: Iterable[str],
    k_values: Iterable[int] = range(2, 16),
    df_for_space: pd.DataFrame | None = None,
    logger: logging.Logger | None = None,
) -> tuple[pd.DataFrame, np.ndarray]:
    """
    Precomputed Euclidean (average-link) Agglomerative metrics without imputation.
    Returns (metrics_df, dist_matrix_filled).
    """
    log = logger or logging.getLogger("MAIN")

    # Extract feature matrix with possible NaNs
    feat_cols_raw = [c for c in df_raw.columns if c not in id_vars]
    X_raw = df_raw[feat_cols_raw].values

    # log.info("Computing Euclidean distance matrix (overlap-aware)...")
    t0 = time.time()
    PCT_LABEL = f"{int(set_percentage*100)}pct" 

    # Path for the threshold-specific folder
    threshold_folder = os.path.join("distance_matrices", f"{PCT_LABEL}_threshold")
    if not os.path.exists(threshold_folder):
        os.makedirs(threshold_folder)

    # Save/load to this folder
    name = f"euclidean_distance_unimputed_{set_percentage*100:.0f}.npy"
    euc_path = os.path.join(threshold_folder, name)

    if os.path.exists(euc_path):
        log.info("Since Euclidean distance matrix on unimputed data for the given threshold is already computed, therfore loading cached Euclidean distance matrix from path %s\n", euc_path)
        D_filled = np.load(euc_path)
    else:
        log.info("\nSince Euclidean distance matrix on unimputed data for the given threshold is not stored in 'distance_matrices' folder, therefore Computing Euclidean distance matrix...\n")
        D = _custom_euclidean_distance(X_raw)
        D_filled = _fill_nan_in_distance(D)
        log.info("Euclidean distance computed in %.1f sec.", time.time() - t0)
        os.makedirs("distance_matrices", exist_ok=True)
        np.save(euc_path, D_filled)
        log.info(f"Saved Euclidean distance matrix to {euc_path}")

    # If imputed data provided, prepare for CH/DB score
    X_space = None
    if df_for_space is not None:
        feat_cols_space = [c for c in df_for_space.columns if c not in id_vars]
        X_space = df_for_space[feat_cols_space].values

    rows: List[AggloMetricsRow] = []
    for k in k_values:
        log.info("Agglomerative clustering on unimputed data with pre-computed Euclidean distance for k=%d ...", k)
        model = AgglomerativeClustering(n_clusters=k, metric="precomputed", linkage="average")
        labels = model.fit_predict(D_filled)

        sil = ch = db = np.nan
        if len(np.unique(labels)) > 1:
            sil = silhouette_score(D_filled, labels, metric="precomputed")
            if X_space is not None:
                ch = calinski_harabasz_score(X_space, labels)
                db = davies_bouldin_score(X_space, labels)

        rows.append(AggloMetricsRow(k=k, silhouette=sil, calinski_harabasz=ch, davies_bouldin=db))

    return pd.DataFrame([r.__dict__ for r in rows]), D_filled

# Fisher's exact test of cluster-outcome association for agglomerative clusters based on a precomputed Euclidean distance matrix.
def run_fisher_test_agglomerative_euclidean_precomputed(
    dist_matrix_filled: np.ndarray,
    df_ids: pd.DataFrame,
    k: int,
    outcome_df: pd.DataFrame,
    outcome_col: str = "target",
) -> pd.DataFrame:
    """Fisher's test for Agglomerative (Euclidean/precomputed) clusters."""
    model = AgglomerativeClustering(n_clusters=k, metric="precomputed", linkage="average")
    labels = model.fit_predict(dist_matrix_filled)

    merged = (
        pd.DataFrame({"hadm_id": df_ids["hadm_id"].values, "cluster": labels})
        .merge(outcome_df[["hadm_id", outcome_col]], on="hadm_id", how="left")
        .dropna(subset=[outcome_col])
    )

    p_vals = []
    clusters = np.unique(labels)
    for cid in clusters:
        in_c = merged["cluster"] == cid
        target_in = merged.loc[in_c, outcome_col]
        target_out = merged.loc[~in_c, outcome_col]

        a = (target_in == 1).sum(); b = (target_out == 1).sum()
        c = (target_in == 0).sum(); d = (target_out == 0).sum()

        _, p = fisher_exact([[a, b], [c, d]], alternative="greater")
        p_vals.append(p)

    rejected, pvals_corr, _, _ = multipletests(p_vals, method="bonferroni")
    return pd.DataFrame({
        "Cluster": clusters,
        "Raw_p_value": p_vals,
        "Corrected_p_value": pvals_corr,
        "Significant_after_correction": rejected,
    })


# ----------------------------------------------------------------------------
#   AGGLOMERATIVE CLUSTERING: EUCLIDEAN  (using Unimputed Normalized vector)
# ----------------------------------------------------------------------------

# Computes a Euclidean distance matrix using only features present for both samples.
def run_agglomerative_euclidean_unimputed_normalized(
    df_raw: 'pd.DataFrame',
    id_vars: Iterable[str],
    k_values: Iterable[int] = range(2, 16),
    df_for_space: 'pd.DataFrame | None' = None,
    logger: 'logging.Logger | None' = None,
) -> tuple['pd.DataFrame', np.ndarray]:
    """
    Agglomerative clustering with custom Euclidean distance on unimputed normalized data.
    Computes the distance matrix with _custom_euclidean_distance considering missing data.
    Caches the distance matrix for reuse.
    """
    log = logger or logging.getLogger("MAIN")

    # Extract feature matrix with possible NaNs
    feat_cols = [c for c in df_raw.columns if c not in id_vars]
    X = df_raw[feat_cols].values  # numpy array
    
    t0 = time.time()
    PCT_LABEL = f"{int(set_percentage*100)}pct"

    # Path for the threshold-specific folder
    threshold_folder = os.path.join("distance_matrices", f"{PCT_LABEL}_threshold")
    os.makedirs(threshold_folder, exist_ok=True)
    
    # Save/load to this folder
    dist_filename = f"euclidean_distance_unimputed_normalized_{int(set_percentage*100)}.npy"
    dist_path = os.path.join(threshold_folder, dist_filename)
             
    if os.path.exists(dist_path):
        log.info("Since Euclidean distance matrix on normalized unimputed data for the given threshold is already computed, therfore loading cached Euclidean distance matrix from path %s\n", dist_path)
        D_filled = np.load(dist_path)
    else:
        log.info("\nSince Euclidean distance matrix on normalized unimputed data for the given threshold is not stored in 'distance_matrices' folder, therefore Computing Euclidean distance matrix...\n")
        D = _custom_euclidean_distance(X)
        D_filled = _fill_nan_in_distance(D)
        log.info("Euclidean distance computed in %.1f sec.", time.time() - t0)
        os.makedirs("distance_matrices", exist_ok=True)
        np.save(dist_path, D_filled)
        log.info(f"Saved Euclidean distance matrix to {dist_path}")

    # If imputed data provided, prepare for CH/DB score
    X_space = None
    if df_for_space is not None:
        feat_cols_space = [c for c in df_for_space.columns if c not in id_vars]
        X_space = df_for_space[feat_cols_space].values

    rows: List[AggloMetricsRow] = []
    for k in k_values:
        log.info("Euclidean Agglomerative clustering using normalized unimputed vector for k=%d ...", k)
        model = AgglomerativeClustering(n_clusters=k, metric="precomputed", linkage="average")
        labels = model.fit_predict(D_filled)

        sil = ch = db = np.nan
        if len(np.unique(labels)) > 1:
            sil = silhouette_score(D_filled, labels, metric="precomputed")
            if X_space is not None:
                ch = calinski_harabasz_score(X_space, labels)
                db = davies_bouldin_score(X_space, labels)

        rows.append(AggloMetricsRow(k=k, silhouette=sil, calinski_harabasz=ch, davies_bouldin=db))

    return pd.DataFrame([r.__dict__ for r in rows]), D_filled


# -----------------------------------------------------------------
#    AGGLOMERATIVE CLUSTERING: EUCLIDEAN  (using Imputed vector)
# -----------------------------------------------------------------
# Computes a Euclidean distance matrix using only features present for both samples.
def run_agglomerative_metrics_euclidean_imputed(
    df_raw: pd.DataFrame,
    id_vars: Iterable[str],
    k_values: Iterable[int] = range(2, 16),
    df_for_space: pd.DataFrame | None = None,
    logger: logging.Logger | None = None,
) -> tuple[pd.DataFrame, np.ndarray]:
    """
    Agglomerative clustering using a precomputed Euclidean distance matrix,
    computed on fully imputed data.
    Returns (metrics_df, dist_matrix).
    """
    log = logger or logging.getLogger("MAIN")

    # Extract feature matrix with possible NaNs
    feat_cols_raw = [c for c in df_raw.columns if c not in id_vars]
    X_raw = df_raw[feat_cols_raw].values

    t0 = time.time()
    PCT_LABEL = f"{int(set_percentage*100)}pct" 

    # Path for the threshold-specific folder
    threshold_folder = os.path.join("distance_matrices", f"{PCT_LABEL}_threshold")
    if not os.path.exists(threshold_folder):
        os.makedirs(threshold_folder)

    # Save/load to this folder
    name = f"euclidean_distance_imputed_{set_percentage*100:.0f}.npy"
    euc_path = os.path.join(threshold_folder, name)

    if os.path.exists(euc_path):
        log.info("Since Euclidean distance matrix on imputed data for the given threshold is already computed, therfore loading cached Euclidean distance matrix from path %s\n", euc_path)
        D_filled = np.load(euc_path)
    else:
        log.info("Since Euclidean distance matrix on imputed data for the given threshold is not stored in 'distance_matrices' folder, therefore Computing Euclidean distance matrix...\n")
        D = pairwise_distances(X_raw, metric='euclidean')
        D_filled = D
        os.makedirs("distance_matrices", exist_ok=True)
        np.save(euc_path, D_filled)
        log.info(f"Saved Euclidean distance matrix (imputed) to {euc_path}")

    # If imputed data provided, prepare for CH/DB scores
    X_space = None
    if df_for_space is not None:
        feat_cols_space = [c for c in df_for_space.columns if c not in id_vars]
        X_space = df_for_space[feat_cols_space].values

    rows: List[AggloMetricsRow] = []
    for k in k_values:
        log.info("Agglomerative clustering on imputed data with pre-computed Euclidean distance for k=%d ...", k)
        model = AgglomerativeClustering(n_clusters=k, metric="precomputed", linkage="average")
        labels = model.fit_predict(D_filled)

        sil = ch = db = np.nan
        if len(np.unique(labels)) > 1:
            sil = silhouette_score(D_filled, labels, metric="precomputed")
            if X_space is not None:
                ch = calinski_harabasz_score(X_space, labels)
                db = davies_bouldin_score(X_space, labels)

        rows.append(AggloMetricsRow(k=k, silhouette=sil, calinski_harabasz=ch, davies_bouldin=db))

    return pd.DataFrame([r.__dict__ for r in rows]), D_filled
    
# Fisher's exact test of cluster-outcome association for agglomerative clusters based on a precomputed Euclidean distance matrix.
def run_fisher_test_agglomerative_euclidean_imputed(
    dist_matrix_filled: np.ndarray,
    df_ids: pd.DataFrame,
    k: int,
    outcome_df: pd.DataFrame,
    outcome_col: str = "target",
) -> pd.DataFrame:
    """Fisher's test for Agglomerative (Euclidean/precomputed) clusters."""
    model = AgglomerativeClustering(n_clusters=k, metric="precomputed", linkage="average")
    labels = model.fit_predict(dist_matrix_filled)

    merged = (
        pd.DataFrame({"hadm_id": df_ids["hadm_id"].values, "cluster": labels})
        .merge(outcome_df[["hadm_id", outcome_col]], on="hadm_id", how="left")
        .dropna(subset=[outcome_col])
    )

    p_vals = []
    clusters = np.unique(labels)
    for cid in clusters:
        in_c = merged["cluster"] == cid
        target_in = merged.loc[in_c, outcome_col]
        target_out = merged.loc[~in_c, outcome_col]

        a = (target_in == 1).sum(); b = (target_out == 1).sum()
        c = (target_in == 0).sum(); d = (target_out == 0).sum()

        _, p = fisher_exact([[a, b], [c, d]], alternative="greater")
        p_vals.append(p)

    rejected, pvals_corr, _, _ = multipletests(p_vals, method="bonferroni")
    return pd.DataFrame({
        "Cluster": clusters,
        "Raw_p_value": p_vals,
        "Corrected_p_value": pvals_corr,
        "Significant_after_correction": rejected,
    })


# -----------------------------------------------------------------
#      AGGLOMERATIVE CLUSTERING: DTW Matrix using dtaidistance
# -----------------------------------------------------------------

# Prepares a list of flattened series for DTW distance computation, assuming all features are ordered by (day, then lab).
def _reshape_to_series(df_features: pd.DataFrame, id_vars: Iterable[str], n_timesteps: int) -> List[np.ndarray]:
    """
    Flattens each sample's multivariate time series to a 1D vector for DTW.
    Assumes feature columns are ordered by time (e.g., ..._day1, ..._day2, ...).
    """
    feat_cols = [c for c in df_features.columns if c not in id_vars]
    n_features = len(feat_cols) // n_timesteps
    X = df_features[feat_cols].values.reshape(-1, n_timesteps, n_features)
    return [x.flatten() for x in X]

# Runs agglomerative clustering on distances computed by fast DTW (dtaidistance), returning metrics and the pairwise DTW distance matrix.
def run_agglomerative_metrics_dtw(
    df_features: pd.DataFrame,
    id_vars: Iterable[str],
    n_timesteps: int,
    k_values: Iterable[int] = range(2, 16),
    df_for_space: pd.DataFrame | None = None,
    logger: logging.Logger | None = None,
) -> tuple[pd.DataFrame, np.ndarray]:
    """
    DTW distance (fast) via dtaidistance.distance_matrix_fast on flattened series.
    Returns (metrics_df, dtw_matrix).
    df_features should be imputed/normalized (no NaNs) so DTW is meaningful.
    df_for_space is used for CH/DB (default: df_features).
    """
    log = logger or logging.getLogger("MAIN")

    # Build list of 1D series
    series_list = _reshape_to_series(df_features, id_vars, n_timesteps)

    t0 = time.time()
    pct = int(set_percentage*100)

    PCT_LABEL = f"{int(set_percentage*100)}pct" 
    # Path for the threshold-specific folder
    threshold_folder = os.path.join("distance_matrices", f"{PCT_LABEL}_threshold")
    if not os.path.exists(threshold_folder):
        os.makedirs(threshold_folder)

    # Save/load to this folder
    fname = f"dtw_distance_fast_{pct}.npy"
    dtw_path = os.path.join(threshold_folder, fname)

    if os.path.exists(dtw_path):
        log.info("Since DTW matrix with dtaidistance (fast) matrix for the given threshold is already computed, therfore loading cached DTW distance matrix from path %s\n", dtw_path)
        dtw_matrix = np.load(dtw_path)

    else:
        log.info("Since DTW matrix with dtaidistance (fast) matrix for the given threshold is not stored in 'distance_matrices' folder, therefore computing DTW matrix...\n")
        dtw_matrix = dtw.distance_matrix_fast(series_list, parallel=True, use_c=True)
        np.fill_diagonal(dtw_matrix, 0,0)
        os.makedirs("distance_matrices", exist_ok=True)
        np.save(dtw_path, dtw_matrix)
        logger.info("Saved DTW matrix fast to %s",dtw_path)

    log.info("DTW matrix computed in %.1f sec.", time.time() - t0)

    # Feature space for CH/DB
    if df_for_space is None:
        df_for_space = df_features
    feat_cols_space = [c for c in df_for_space.columns if c not in id_vars]
    X_space = df_for_space[feat_cols_space].values

    rows: List[AggloMetricsRow] = []
    for k in k_values:
        log.info("Agglomerative clustering (DTW matrix) for k=%d ...", k)
        model = AgglomerativeClustering(n_clusters=k, metric="precomputed", linkage="average")
        labels = model.fit_predict(dtw_matrix)

        sil = ch = db = np.nan
        if len(np.unique(labels)) > 1:
            sil = silhouette_score(dtw_matrix, labels, metric="precomputed")
            try:
                ch = calinski_harabasz_score(X_space, labels)
                db = davies_bouldin_score(X_space, labels)
            except Exception as e:
                log.warning("CH/DB failed for k=%d: %s", k, e)

        rows.append(AggloMetricsRow(k=k, silhouette=sil, calinski_harabasz=ch, davies_bouldin=db))

    return pd.DataFrame([r.__dict__ for r in rows]), dtw_matrix

# Fisher's exact test for clusters produced by agglomerative clustering over a DTW distance matrix.
def run_fisher_test_agglomerative_dtw(
    dtw_matrix: np.ndarray,
    df_ids: pd.DataFrame,     # must have hadm_id in the same order
    k: int,
    outcome_df: pd.DataFrame,
    outcome_col: str = "target",
) -> pd.DataFrame:
    model = AgglomerativeClustering(n_clusters=k, metric="precomputed", linkage="average")
    labels = model.fit_predict(dtw_matrix)

    merged = (
        pd.DataFrame({"hadm_id": df_ids["hadm_id"].values, "cluster": labels})
        .merge(outcome_df[["hadm_id", outcome_col]], on="hadm_id", how="left")
        .dropna(subset=[outcome_col])
    )

    p_vals = []
    clusters = np.unique(labels)
    for cid in clusters:
        in_c = merged["cluster"] == cid
        target_in = merged.loc[in_c, outcome_col]
        target_out = merged.loc[~in_c, outcome_col]

        a = (target_in == 1).sum(); b = (target_out == 1).sum()
        c = (target_in == 0).sum(); d = (target_out == 0).sum()

        _, p = fisher_exact([[a, b], [c, d]], alternative="greater")
        p_vals.append(p)

    rejected, pvals_corr, _, _ = multipletests(p_vals, method="bonferroni")
    return pd.DataFrame({
        "Cluster": clusters,
        "Raw_p_value": p_vals,
        "Corrected_p_value": pvals_corr,
        "Significant_after_correction": rejected,
    })
# -------------------------------------------------------------
#     AGGLOMERATIVE CLUSTERING: DTW using tslearn.cdist_dtw
# -------------------------------------------------------------

# Computes DTW distances for multivariate time series using tslearn, then performs agglomerative clustering and cluster metric evaluation.
def run_agglomerative_metrics_dtw_tslearn(
    df_features: pd.DataFrame,
    id_vars: Iterable[str],
    n_timesteps: int,
    k_values: Iterable[int] = range(2, 16),
    df_for_space: pd.DataFrame | None = None,
    logger: logging.Logger | None = None,
) -> tuple[pd.DataFrame, np.ndarray]:
    """
    DTW distance via tslearn.metrics.cdist_dtw on multivariate series.
    df_features: imputed & ordered (no NaNs).
    n_timesteps: length of the window (e.g., 7).
    Returns (metrics_df, dtw_matrix).
    """
    log = logger or logging.getLogger("MAIN")

    # reshape (n_samples, n_timesteps, n_features)
    feat_cols = [c for c in df_features.columns if c not in id_vars]
    n_features = len(feat_cols) // n_timesteps
    X3d = df_features[feat_cols].values.reshape(-1, n_timesteps, n_features)

    t0 = time.time()
    pct = int(set_percentage*100)

    PCT_LABEL = f"{int(set_percentage*100)}pct" 
    # Path for the threshold-specific folder
    threshold_folder = os.path.join("distance_matrices", f"{PCT_LABEL}_threshold")
    if not os.path.exists(threshold_folder):
        os.makedirs(threshold_folder)

    # Save/load to this folder
    fname = f"dtw_distance_tslearn_{pct}.npy"
    dtw_path = os.path.join(threshold_folder, fname)

    if os.path.exists(dtw_path):
        log.info("Since DTW matrix using tslearn.metrics.cdist_dtw for the given threshold is already computed, therfore loading cached DTW distance matrix from path %s\n", dtw_path)
        dtw_matrix = np.load(dtw_path)

    else:
        log.info("Since DTW matrix (computed using tslearn.metrics.cdist_dtw) for the given threshold is not stored in 'distance_matrices' folder, therefore computing DTW matrix and saving it for future...\n")
        dtw_matrix = cdist_dtw(X3d, n_jobs=-1)
        np.fill_diagonal(dtw_matrix, 0.0)
        os.makedirs("distance_matrices", exist_ok=True)
        np.save(dtw_path, dtw_matrix)
        logger.info("Saved DTW matrix fast to %s",dtw_path)

    log.info("DTW matrix computed in %.1f sec.", time.time() - t0)

    if df_for_space is None:
        df_for_space = df_features
    feat_cols_space = [c for c in df_for_space.columns if c not in id_vars]
    X_space = df_for_space[feat_cols_space].values

    rows: List[AggloMetricsRow] = []
    for k in k_values:
        log.info("Agglomerative clustering (DTW matrix) for k=%d ...", k)
        model = AgglomerativeClustering(n_clusters=k, metric="precomputed", linkage="average")
        labels = model.fit_predict(dtw_matrix)

        sil = ch = db = np.nan
        if len(np.unique(labels)) > 1:
            sil = silhouette_score(dtw_matrix, labels, metric="precomputed")
            try:
                ch = calinski_harabasz_score(X_space, labels)
                db = davies_bouldin_score(X_space, labels)
            except Exception as e:
                log.warning("CH/DB failed for k=%d: %s", k, e)

        rows.append(AggloMetricsRow(k=k, silhouette=sil, calinski_harabasz=ch, davies_bouldin=db))

    return pd.DataFrame([r.__dict__ for r in rows]), dtw_matrix

# Fisher's test for clusters based on tslearn DTW matrix
def run_fisher_test_agglomerative_dtw_tslearn(
    dtw_matrix: np.ndarray,
    df_ids: pd.DataFrame,
    k: int,
    outcome_df: pd.DataFrame,
    outcome_col: str = "target",
) -> pd.DataFrame:
    """Fisher's exact test for clusters from tslearn-DTW matrix."""
    model = AgglomerativeClustering(n_clusters=k, metric="precomputed", linkage="average")
    labels = model.fit_predict(dtw_matrix)

    merged = (
        pd.DataFrame({"hadm_id": df_ids["hadm_id"].values, "cluster": labels})
        .merge(outcome_df[["hadm_id", outcome_col]], on="hadm_id", how="left")
        .dropna(subset=[outcome_col])
    )

    p_vals = []
    clusters = np.unique(labels)
    for cid in clusters:
        in_c = merged["cluster"] == cid
        target_in = merged.loc[in_c, outcome_col]
        target_out = merged.loc[~in_c, outcome_col]

        a = (target_in == 1).sum(); b = (target_out == 1).sum()
        c = (target_in == 0).sum(); d = (target_out == 0).sum()

        _, p = fisher_exact([[a, b], [c, d]], alternative="greater")
        p_vals.append(p)

    rejected, pvals_corr, _, _ = multipletests(p_vals, method="bonferroni")
    return pd.DataFrame({
        "Cluster": clusters,
        "Raw_p_value": p_vals,
        "Corrected_p_value": pvals_corr,
        "Significant_after_correction": rejected,
    })


# -----------------------------------------------------------------------------
#    AGGLOMERATIVE CLUSTERING : BINARY DISTANCES ( HAMMING / JACCARD / DICE)
# -----------------------------------------------------------------------------

# Pivot long-format filtered labs into a binary (0/1) matrix: rows are admissions, columns are itemids (lab tests). Entry is 1 if a lab was measured for that admission, otherwise 0.
def build_binary_dataset(
    filtered_df: pd.DataFrame,
    id_col: str = "hadm_id",
    item_col: str = "itemid",
) -> pd.DataFrame:
    """Pivot to binary matrix (1 if lab measured)."""
    bin_df = filtered_df.copy()
    bin_df["measured"] = 1
    binary_dataset = bin_df.pivot_table(
        index=id_col,
        columns=item_col,
        values="measured",
        aggfunc="max",
        fill_value=0
    ).reset_index()
    return binary_dataset

# Compute Dice distance for a boolean feature matrix
def _dice_distance(X_bool: np.ndarray) -> np.ndarray:
    """Dice distance for boolean matrix."""
    n = X_bool.shape[0]
    D = np.zeros((n, n))
    for i in range(n):
        Ai = X_bool[i]
        for j in range(i, n):
            Bj = X_bool[j]
            inter = np.sum(Ai & Bj)
            size_sum = Ai.sum() + Bj.sum()
            dist = 0.0 if size_sum == 0 else 1 - (2 * inter / size_sum)
            D[i, j] = D[j, i] = dist
    return D

# Compute Hamming, Jaccard, and Dice distance matrices for a binary dataset. Returns both numeric and boolean versions of the feature array, and a dict of distance matrices.
def compute_binary_distance_matrices(binary_dataset: pd.DataFrame, id_col: str = "hadm_id"):
    """
    Returns:
      feats_np      : numeric 0/1 array
      feats_bool_np : boolean array
      mats          : dict {name: distance_matrix}
    """
    feats = binary_dataset.drop(columns=[id_col])
    feats_bool = feats.astype(bool)

    hamming_dist = pairwise_distances(feats.values, metric="hamming")
    jaccard_dist = pairwise_distances(feats_bool.values, metric="jaccard")
    dice_dist = _dice_distance(feats_bool.values)

    mats = {
        "Hamming": hamming_dist,
        "Jaccard": jaccard_dist,
        "Dice": dice_dist,
    }
    return feats.values, feats_bool.values, mats


def prepare_binary_distance_matrices(filtered_common_labs_percent, set_percentage, id_col="hadm_id", item_col="itemid", folder="distance_matrices", logger=None):
    
    log = logger or logging.getLogger("MAIN")
    pct = int(set_percentage * 100)
    base = f"binary_{pct}"
    
    # Create threshold-specific subfolder
    threshold_folder = os.path.join(folder, f"{pct}pct_threshold")
    os.makedirs(threshold_folder, exist_ok=True)
    
    feat_path = os.path.join(threshold_folder, f"{base}_feats.npy")
    feat_bool_path = os.path.join(threshold_folder, f"{base}_feats_bool.npy")
    dist_path = os.path.join(threshold_folder, f"{base}_distance.npy")
    binary_dataset_path = os.path.join(threshold_folder, f"{base}_binary_dataset.csv")
    
    if all(os.path.exists(p) for p in (feat_path, feat_bool_path, dist_path)):
        log.info("Found cached binary matrices (Hamming, Jaccard & Dice) in '%s'.", threshold_folder)
        bin_feats_np = np.load(feat_path, allow_pickle=True)
        bin_feats_bool_np = np.load(feat_bool_path, allow_pickle=True)
        dist_mats_bin = np.load(dist_path, allow_pickle=True).item()
        if os.path.exists(binary_dataset_path):
            binary_dataset = pd.read_csv(binary_dataset_path)
        else:
            binary_dataset = build_binary_dataset(filtered_common_labs_percent, id_col=id_col, item_col=item_col)
            binary_dataset.to_csv(binary_dataset_path, index=False)
            log.info(f"Recomputed and saved binary dataset CSV at {binary_dataset_path}")
    else:
        log.info("Cached binary matrices ((Hamming, Jaccard & Dice) ) NOT found in '%s'. Computing and saving for future runs.", threshold_folder)
        binary_dataset = build_binary_dataset(filtered_common_labs_percent, id_col=id_col, item_col=item_col)
        bin_feats_np, bin_feats_bool_np, dist_mats_bin = compute_binary_distance_matrices(binary_dataset, id_col=id_col)
        np.save(feat_path, bin_feats_np)
        np.save(feat_bool_path, bin_feats_bool_np)
        np.save(dist_path, dist_mats_bin)
        binary_dataset.to_csv(binary_dataset_path, index=False)
        log.info("Saved binary matrices and features to '%s'.", threshold_folder)
    
    return bin_feats_np, bin_feats_bool_np, dist_mats_bin, binary_dataset

# Run agglomerative clustering and compute clustering metrics for each binary distance type. Returns a dict keyed by distance name, each containing a metrics dataframe for k in range.
def run_agglomerative_metrics_binary(
    dist_mats: Dict[str, np.ndarray],
    feature_space_np: np.ndarray,
    k_values: Iterable[int] = range(2, 16),
    logger: logging.Logger | None = None,
) -> Dict[str, pd.DataFrame]:
    """
    Run clustering & metrics for each binary distance matrix.
    Returns dict {name: metrics_df} where metrics_df columns = ['k', 'silhouette', 'calinski_harabasz', 'davies_bouldin'].
    """
    log = logger or logging.getLogger("MAIN")
    results: Dict[str, pd.DataFrame] = {}

    for name, D in dist_mats.items():
        log.info("Processing %s distance matrix ...", name)
        rows: List[AggloMetricsRow] = []
        for k in k_values:
            model = AgglomerativeClustering(n_clusters=k, metric="precomputed", linkage="average")
            labels = model.fit_predict(D)

            sil = ch = db = np.nan
            if len(np.unique(labels)) > 1:
                sil = silhouette_score(D, labels, metric="precomputed")
                try:
                    ch = calinski_harabasz_score(feature_space_np, labels)
                    db = davies_bouldin_score(feature_space_np, labels)
                except Exception as e:
                    log.warning("Metric computation error (%s, k=%d): %s", name, k, e)

            rows.append(AggloMetricsRow(k=k, silhouette=sil, calinski_harabasz=ch, davies_bouldin=db))

        results[name] = pd.DataFrame([r.__dict__ for r in rows])

    return results

# Run Fisher's exact test to assess the association between clusters (from binary distance agglomerative clustering) and outcome
def run_fisher_test_agglomerative_binary(
    dist_matrix: np.ndarray,
    df_ids: pd.DataFrame,            # must have hadm_id
    k: int,
    outcome_df: pd.DataFrame,
    outcome_col: str = "target",
) -> pd.DataFrame:
    """Fisher's exact test for a single binary distance matrix & k."""
    model = AgglomerativeClustering(n_clusters=k, metric="precomputed", linkage="average")
    labels = model.fit_predict(dist_matrix)

    merged = (
        pd.DataFrame({"hadm_id": df_ids["hadm_id"].values, "cluster": labels})
        .merge(outcome_df[["hadm_id", outcome_col]], on="hadm_id", how="left")
        .dropna(subset=[outcome_col])
    )

    p_vals = []
    clusters = np.unique(labels)
    for cid in clusters:
        in_c = merged["cluster"] == cid
        target_in = merged.loc[in_c, outcome_col]
        target_out = merged.loc[~in_c, outcome_col]

        a = (target_in == 1).sum(); b = (target_out == 1).sum()
        c = (target_in == 0).sum(); d = (target_out == 0).sum()

        _, p = fisher_exact([[a, b], [c, d]], alternative="greater")
        p_vals.append(p)

    rejected, pvals_corr, _, _ = multipletests(p_vals, method="bonferroni")
    return pd.DataFrame({
        "Cluster": clusters,
        "Raw_p_value": p_vals,
        "Corrected_p_value": pvals_corr,
        "Significant_after_correction": rejected,
    })


# ----------------------------------------------
#   DBSCAN CLUSTERING using 'euclidean' metric
# ----------------------------------------------

# Compute sorted k-nearest neighbor distances for each sample, useful for plotting the elbow and selecting DBSCAN epsilon parameter.
def kdist_sorted(X: np.ndarray, k: int = 5) -> np.ndarray:
    """Return the sorted k-th NN distance for each sample (for elbow plot)."""
    nbrs = NearestNeighbors(n_neighbors=k).fit(X)
    dists = nbrs.kneighbors(X)[0][:, -1]
    return np.sort(dists)

# Run DBSCAN clustering with Euclidean distance, evaluate noise/excluded points, and report silhouette/Calinski-Harabasz/Davies-Bouldin scores on non-noise clusters.
def run_dbscan_euclidean(
    df_features: pd.DataFrame,
    id_vars: Iterable[str],
    eps: float,
    min_samples: int = 5,
    outcome_df: pd.DataFrame | None = None,
    outcome_col: str = "target",
) -> tuple[pd.DataFrame, np.ndarray, dict]:
    """
    Run DBSCAN (euclidean) and compute silhouette/CH/DB (excluding noise).
    Returns:
        metrics_df  : one-row DataFrame with scores & counts
        labels      : full cluster labels (noise = -1)
        fisher_dict : {'results': DataFrame} if outcome_df given else {}
    """
    feature_cols = [c for c in df_features.columns if c not in id_vars]
    X = df_features[feature_cols].values

    db = DBSCAN(eps=eps, min_samples=min_samples, metric="euclidean")
    labels = db.fit_predict(X)

    valid_mask = labels != -1
    if valid_mask.sum() > 0 and np.unique(labels[valid_mask]).size > 1:
        sil = silhouette_score(X[valid_mask], labels[valid_mask])
        ch = calinski_harabasz_score(X[valid_mask], labels[valid_mask])
        dbi = davies_bouldin_score(X[valid_mask], labels[valid_mask])
    else:
        sil = ch = dbi = np.nan

    metrics_df = pd.DataFrame({
        "eps": [eps],
        "min_samples": [min_samples],
        "silhouette": [sil],
        "calinski_harabasz": [ch],
        "davies_bouldin": [dbi],
        "n_clusters": [np.unique(labels[labels != -1]).size],
        "n_noise": [(labels == -1).sum()]
    })

    fisher_dict: dict = {}
    if outcome_df is not None:
        merged = (
            pd.DataFrame({"hadm_id": df_features["hadm_id"].values, "cluster": labels})
            .merge(outcome_df[["hadm_id", outcome_col]], on="hadm_id", how="left")
            .dropna(subset=[outcome_col])
        )
        clust_ids = np.unique(labels)
        clust_ids = clust_ids[clust_ids != -1]

        p_vals = []
        for cid in clust_ids:
            in_c = merged["cluster"] == cid
            t_in = merged.loc[in_c, outcome_col]
            t_out = merged.loc[~in_c, outcome_col]

            a = (t_in == 1).sum();  b = (t_out == 1).sum()
            c = (t_in == 0).sum();  d = (t_out == 0).sum()

            _, p = fisher_exact([[a, b], [c, d]], alternative="greater")
            p_vals.append(p)

        rejected, p_corr, _, _ = multipletests(p_vals, method="bonferroni")
        fisher_res = pd.DataFrame({
            "Cluster": clust_ids,
            "Raw_p_value": p_vals,
            "Corrected_p_value": p_corr,
            "Significant_after_correction": rejected
        }).sort_values("Corrected_p_value")
        fisher_dict["results"] = fisher_res

    return metrics_df, labels, fisher_dict


# -------------------------------------------------
#    SPECTRAL CLUSTERING (HMM-derived features)
# -------------------------------------------------

# Given time-series feature matrix, fits a Gaussian HMM to all data and generates new features as mean posterior state probabilities per sequence.
def build_hmm_state_features(
    df_features: pd.DataFrame,
    id_vars: Iterable[str],
    n_timesteps: int,
    n_hmm_states: int = 3,
    covariance_type: str = "diag",
    n_iter: int = 100,
    random_state: int = 42,
    logger: logging.Logger | None = None,
) -> tuple[np.ndarray, pd.DataFrame]:
    """
    Fit one GaussianHMM on ALL sequences and use mean posterior state probs per sample as features.
    Returns (feature_matrix, df_ids)
    """
    log = logger or logging.getLogger("MAIN")

    feat_cols = [c for c in df_features.columns if c not in id_vars]
    n_features = len(feat_cols) // n_timesteps
    X3d = df_features[feat_cols].values.reshape(-1, n_timesteps, n_features)
    n_samples = X3d.shape[0]

    # Train HMM on concatenated sequences
    X_combined = X3d.reshape(-1, n_features)
    log.info("Fitting GaussianHMM: states=%d, timesteps=%d, features=%d", n_hmm_states, n_timesteps, n_features)

    model = hmm.GaussianHMM(
        n_components=n_hmm_states,
        covariance_type=covariance_type,
        n_iter=n_iter,
        random_state=random_state,
        verbose=False
    )
    model.fit(X_combined)

    # Posterior means per sequence
    feats = []
    for seq in X3d:
        _, post = model.score_samples(seq)
        feats.append(post.mean(axis=0))
    feats = np.array(feats)

    df_ids = df_features[list(id_vars)].reset_index(drop=True)
    return feats, df_ids

# Run spectral clustering for a range of k and return clustering evaluation metrics for each k
def run_spectral_metrics(
    feature_matrix: np.ndarray,
    k_values: Iterable[int] = range(2, 16),
    affinity: str = "nearest_neighbors",
    random_state: int = 42,
    logger: logging.Logger | None = None,
) -> pd.DataFrame:
    """Compute silhouette / CH / DB over k for Spectral clustering on given features."""
    log = logger or logging.getLogger("MAIN")

    rows: List[AggloMetricsRow] = []
    for k in k_values:
        log.info("Spectral clustering with k=%d ...", k)
        spec = SpectralClustering(n_clusters=k, affinity=affinity, random_state=random_state)
        labels = spec.fit_predict(feature_matrix)

        sil = ch = db = np.nan
        if np.unique(labels).size > 1:
            sil = silhouette_score(feature_matrix, labels)
            ch = calinski_harabasz_score(feature_matrix, labels)
            db = davies_bouldin_score(feature_matrix, labels)

        rows.append(AggloMetricsRow(k=k, silhouette=sil, calinski_harabasz=ch, davies_bouldin=db))

    return pd.DataFrame([r.__dict__ for r in rows])

# Fisher's test for clusters from spectral clustering's label assignments
def run_fisher_test_spectral(
    labels: np.ndarray,
    df_ids: pd.DataFrame,
    outcome_df: pd.DataFrame,
    outcome_col: str = "target",
) -> pd.DataFrame:
    """Fisher's exact test for Spectral clustering labels."""
    merged = (
        pd.DataFrame({"hadm_id": df_ids["hadm_id"].values, "cluster": labels})
        .merge(outcome_df[["hadm_id", outcome_col]], on="hadm_id", how="left")
        .dropna(subset=[outcome_col])
    )

    p_vals = []
    clusters = np.unique(labels)
    for cid in clusters:
        in_c = merged["cluster"] == cid
        t_in = merged.loc[in_c, outcome_col]
        t_out = merged.loc[~in_c, outcome_col]

        a = (t_in == 1).sum(); b = (t_out == 1).sum()
        c = (t_in == 0).sum(); d = (t_out == 0).sum()

        _, p = fisher_exact([[a, b], [c, d]], alternative="greater")
        p_vals.append(p)

    rejected, p_corr, _, _ = multipletests(p_vals, method="bonferroni")
    return pd.DataFrame({
        "Cluster": clusters,
        "Raw_p_value": p_vals,
        "Corrected_p_value": p_corr,
        "Significant_after_correction": rejected
    })


# ===========================================================
#   K-Medoids Clustering using Euclidean Distance matrix
# ===========================================================

def run_kmedoids_metrics(
    df_features: pd.DataFrame,
    id_vars: Iterable[str],
    k_values: Iterable[int] = range(2, 16),
    metric: str = "euclidean",
    precomputed: bool = False,
    dist_matrix: np.ndarray | None = None,
    random_state: int = 42,
    logger: logging.Logger | None = None,
) -> pd.DataFrame:
    """
    Run KMedoids for each k and return a metrics DataFrame.
    If `precomputed` is True, you must pass a square `dist_matrix`.
    Otherwise it will call KMedoids(metric=metric) on the feature matrix.
    """
    log = logger or logging.getLogger("MAIN")

    # pick X or D
    if precomputed:
        if dist_matrix is None:
            raise ValueError("dist_matrix must be provided when precomputed=True")
        D = dist_matrix
        use_pre = True
    else:
        # build feature matrix
        feature_cols = [c for c in df_features.columns if c not in id_vars]
        X = df_features[feature_cols].values
        use_pre = False

    rows: List[KMedoidsMetricsRow] = []
    for k in k_values:
        log.info("KMedoids (%s) clustering with k=%d ...", metric, k)
        if use_pre:
            model = KMedoids(n_clusters=k,
                             metric="precomputed",
                             random_state=random_state)
            labels = model.fit_predict(D)
            # silhouette over the precomputed matrix:
            sil = silhouette_score(D, labels, metric="precomputed") if len(np.unique(labels))>1 else np.nan
            # for CH/DB use the X from df_for_space:
        else:
            model = KMedoids(n_clusters=k,
                             metric=metric,
                             random_state=random_state)
            labels = model.fit_predict(X)
            sil = silhouette_score(X, labels) if len(np.unique(labels))>1 else np.nan

        # calinski & davies always on X
        if use_pre:
            feature_cols = [c for c in df_features.columns if c not in id_vars]
            X_ch = df_features[feature_cols].values
        else:
            X_ch = X

        if len(np.unique(labels))>1:
            ch = calinski_harabasz_score(X_ch, labels)
            db = davies_bouldin_score(X_ch, labels)
        else:
            ch = db = np.nan

        rows.append(
            KMedoidsMetricsRow(
                k=k,
                silhouette=sil,
                calinski_harabasz=ch,
                davies_bouldin=db,
            )
        )

    return pd.DataFrame([r.__dict__ for r in rows])

def run_fisher_test_kmedoids(
    df_features: pd.DataFrame,
    id_vars: Iterable[str],
    k: int,
    outcome_df: pd.DataFrame,
    outcome_col: str = "target",
    metric: str = "euclidean",
    precomputed: bool = False,
    dist_matrix: np.ndarray | None = None,
    random_state: int = 42,
) -> pd.DataFrame:
    """
    Fisher’s test for KMedoids clusters.
    If precomputed=True, must pass dist_matrix; else uses metric on df_features.
    """
    if precomputed:
        if dist_matrix is None:
            raise ValueError("need dist_matrix for precomputed")
        model = KMedoids(n_clusters=k, metric="precomputed", random_state=random_state)
        labels = model.fit_predict(dist_matrix)
    else:
        feature_cols = [c for c in df_features.columns if c not in id_vars]
        X = df_features[feature_cols].values
        model = KMedoids(n_clusters=k, metric=metric, random_state=random_state)
        labels = model.fit_predict(X)

    merged = (
        pd.DataFrame({"hadm_id": df_features["hadm_id"].values, "cluster": labels})
        .merge(outcome_df[["hadm_id", outcome_col]], on="hadm_id", how="left")
        .dropna(subset=[outcome_col])
    )

    p_vals, clusters = [], np.unique(labels)
    for cid in clusters:
        in_c = merged["cluster"] == cid
        a = (merged.loc[in_c, outcome_col] == 1).sum()
        b = (merged.loc[~in_c, outcome_col] == 1).sum()
        c = (merged.loc[in_c, outcome_col] == 0).sum()
        d = (merged.loc[~in_c, outcome_col] == 0).sum()
        _, p = fisher_exact([[a, b], [c, d]], alternative="greater")
        p_vals.append(p)

    rej, p_corr, *_ = multipletests(p_vals, method="bonferroni")
    return pd.DataFrame({
        "Cluster": clusters,
        "Raw_p_value": p_vals,
        "Corrected_p_value": p_corr,
        "Significant_after_correction": rej

    })

