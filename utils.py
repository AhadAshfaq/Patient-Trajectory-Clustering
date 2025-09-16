'''
                =========================================================   UTILITIES MODULE   =======================================================

'''


"""
This module provides core utility functions to support the entire data preprocessing and feature engineering pipeline. It includes routines for:

- Data loading, merging, and time-window filtering
- Data cleaning and lab event filtering based on configurable cohort/admission thresholds
- Temporal discretization (binning labs per day/time step)
- Feature matrix construction, normalization, and various missing data imputation strategies
- Data transformation between long and wide formats for time-series analysis
- Helper functions for plotting and saving clustering metrics and distance matrices
- Utility routines for advanced distance matrix computation (handling missing values)
- Modular wrappers for executing statistical analyses and iterative evaluation loops

The `utils` module ensures standardized, reproducible, and robust processing of raw clinical data for use in downstream machine learning, clustering, and statistical analyses. It is designed to be reusable across multiple experiment designs with a focus on clarity, collaboration, and extensibility.

"""

from imports import *
import utils
from clustering import *


logger = logging.getLogger("UTILS")

# Configure logging for the pipeline with the specified verbosity level
def configure_logging(level=logging.INFO):
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - [%(name)s] - %(levelname)s - %(message)s')

# Folder name where all intermediate and raw data pickle files will be saved
output_folder = 'preprocessing_data'

# Create the output folder if it does not exist to avoid file save errors
def ensure_folder_exists():
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

# Save the raw input dataframes 'cohort_data' and 'labevents_data' as pickled files, this speeds up future runs by avoiding repeated costly CSV loading and parsing.
def save_raw_dataframes(cohort_data, labevents_data):
    ensure_folder_exists()
    cohort_data.to_pickle(os.path.join(output_folder, 'cohort_data.pkl'))
    labevents_data.to_pickle(os.path.join(output_folder, 'labevents_data.pkl'))

# Check if the raw data pickle files already exist in the output folder. If both files exist, return True to indicate caching can be used otherwise print info message and return False to trigger CSV loading and saving
def check_existing_raw_data():
    paths = [
        os.path.join(output_folder, 'cohort_data.pkl'),
        os.path.join(output_folder, 'labevents_data.pkl')
    ]
    if all(os.path.exists(p) for p in paths):
        logger.info(f"Raw cohort and labevents files already stored in '{output_folder}'. Using cached versions.\n")
        return True
    else:
        logger.info(f"Raw cohort and labevents files not found in '{output_folder}'. Loading from CSV and saving for future runs.\n")
        return False

# Check if all the required preprocessed intermediate files for a given label exist in the folder, provides informative print messages about whether preprocessing needs to run or cached files will be used.
def check_existing_preprocessed_files(label="", required_files=None):
    if required_files is None:
        required_files = [
            f'filtered_common_labs_{label}.pkl',
            f'unimputed_vector_{label}.pkl',
            f'feature_matrix_{label}.pkl',
            f'unimputed_normalized_vector_{label}.pkl',
            f'imputed_vector_{label}.pkl'
        ]
    if not os.path.exists(output_folder):
        logger.info(f"Folder '{output_folder}' not found. Running preprocessing for the first time.")

        return False
    else:
        all_files_exist = all(os.path.exists(os.path.join(output_folder, f)) for f in required_files)
        if all_files_exist:
            logger.info(f"All required files for the selected threshold, 'filtered_common_labs', 'unimputed_vector', 'feature_matrix', unimputed_normalized_vector and 'imputed_vector' are found in '{output_folder}' folder. Therefore, we are going to use already stored files for further processing.\n")
            return True
        else:
            logger.info(f"Folder '{output_folder}' exists but not all required files are found. Running preprocessing.")
            return False

# Save the main intermediate preprocessing outputs as pickled DataFrames, appending an optional label to the filenames to distinguish between different runs/settings.
def save_preprocessing_outputs(filtered_common_labs_percent, pivot_percent, feature_matrix_df, unimputed_normalized, imputed_preprocessed, label=""):
    ensure_folder_exists()
    filtered_common_labs_percent.to_pickle(os.path.join(output_folder, f'filtered_common_labs_{label}.pkl'))
    pivot_percent.to_pickle(os.path.join(output_folder, f'unimputed_vector_{label}.pkl'))
    feature_matrix_df.to_pickle(os.path.join(output_folder, f'feature_matrix_{label}.pkl'))
    unimputed_normalized.to_pickle(os.path.join(output_folder, f'unimputed_normalized_vector_{label}.pkl'))
    imputed_preprocessed.to_pickle(os.path.join(output_folder, f'imputed_vector_{label}.pkl'))

# Load CSV data from the given path, parsing the specified date column with an optionally to accept a row limit
def loading_data(path, col_name, num_row=None):
    if num_row is not None:
        return pd.read_csv(path, parse_dates=[col_name], nrows=num_row)
    else:
        return pd.read_csv(path, parse_dates=[col_name])

# Merge lab events data with cohort information to bring in the discharge time for each admission
def merge_data(data1, data2):
     merged = pd.merge(
        data1,
        data2[['subject_id', 'hadm_id', 'dischtime']],
        on=['subject_id', 'hadm_id'],
        how='inner'
    )
     return merged

# Select lab events that occurred within the specified number of days before discharge for further analysis
def labs_within_n_days_of_discharge(merged, days):

    # Calculating the number of "days" ("days" from hyperparameters.py) before discharge
    merged['days_before_discharge'] = (merged['dischtime'] - merged['charttime']).dt.days

    # Filtering for the labs within last selected days ("days" from hyperparameters.py) before discharge.
    filtered_labs = merged[
        (merged['days_before_discharge'] >= 0) &
        (merged['days_before_discharge'] < days)
        ].copy()

    # Selecting required columns
    result = filtered_labs[
        ['itemid', 'valuenum', 'charttime', 'hadm_id', 'subject_id', 'dischtime', 'days_before_discharge']]
    return result

# Filter lab events to retain only labs observed in at least a given percentage of admissions and with an average of at least two measurements per admission
def filtering_labs_by_percentage(result, percentage,days):

    # Calculating total admissions (hadm_id) and checking how many unique values are there in labs (item_ids)
    unique_labs = result['itemid'].nunique()
    total_admissions = result['hadm_id'].nunique()

    logger.info("Total unique item_ids (lab tests) are %d and Total admissions (patients) are %d.", unique_labs, total_admissions)

    # Counting admissions per itemid
    itemid_counts = result.groupby('itemid')['hadm_id'].nunique().reset_index()
    itemid_counts.rename(columns={'hadm_id': 'admission_count'}, inplace=True)

    # Calculating the selected threshold ("set_percentage" from hyperparameters.py) of the total admissions,
    threshold = round(percentage * total_admissions)
    logger.info(f"Threshold ({percentage * 100}% of the total admissions): {threshold}")

    # Getting common itemids that are conduted by selected threshold ("set_percentage" from hyperparameters.py) of the patients
    selected_itemids = itemid_counts[itemid_counts['admission_count'] >= threshold]['itemid']

    # Summary
    logger.info(
    f"{len(selected_itemids)} common lab tests out of {itemid_counts.shape[0]} total have been performed in at least {percentage * 100:.0f}% of admissions.\n")

    # logger.info(f"Filtered lab's ids are ({len(selected_itemids)} lab tests): \n {list(selected_itemids)}\n")
    logger.info(
    f"IDs of included labs ({len(selected_itemids)} selected): \n{list(selected_itemids)}\n")

    '''
        Now we'll filter the selected common lab tests that are measured ≥2 times per admission on average
    '''

    # Filtering the result for required labs only
    filtered_common_labs = result[result['itemid'].isin(selected_itemids)].copy()

    # Counting number of measurements per itemid per admission
    lab_counts = filtered_common_labs.groupby(['itemid', 'hadm_id']).size().reset_index(name='count')

    # For each itemid, calculating the average number of measurements per admission
    avg_counts = lab_counts.groupby('itemid')['count'].mean().reset_index(name='avg_per_admission')

    # Selecting itemids with average >= 2
    selected_itemids_2plus = avg_counts[avg_counts['avg_per_admission'] >= 2]['itemid']

    logger.info(
    f" After applying the additional filter, i.e mean measurements ≥2 per admission (labs that are measured ≥2 times per admission on average): {len(selected_itemids_2plus)} lab tests remain\n" )

    logger.info(
        f"Final IDs of {len(selected_itemids_2plus)} retained labs to create a {days}-dimensional vector: \n{list(selected_itemids_2plus)}\n" )

    # Filtering to keep only these labs
    filtered_common_labs = filtered_common_labs[filtered_common_labs['itemid'].isin(selected_itemids_2plus)].copy()

    return filtered_common_labs, unique_labs, total_admissions


# Temporal Discretization (24h Bins) - Discretize lab events temporally into day bins and pivot to a wide-format feature matrix with lab-day columns
def descritization(filtered_common_labs, days):

    # Days should be counted forward from the start of the observation window. This means day 1 is the earliest day in the window, day 7 is the latest.
    filtered_common_labs['day_bin'] = days - filtered_common_labs['days_before_discharge']

    # Aggregating by mean
    agg = (
        filtered_common_labs
        .groupby(['hadm_id', 'itemid', 'day_bin'])['valuenum']
        .mean()
        .reset_index()
    )

    # Creating feature-day column names
    agg['feature_day'] = agg['itemid'].astype(str) + '_day' + agg['day_bin'].astype(str)

    # Creating a vector for each admission and each selected lab feature
    pivot = agg.pivot_table(
        index='hadm_id',
        columns='feature_day',
        values='valuenum'
    )

    # Adding subject_id to vectors to make it more understandable
    pivot = pivot.reset_index().merge(
        filtered_common_labs[['hadm_id', 'subject_id']].drop_duplicates(),
        on='hadm_id',
        how='left'
    )

    '''
    We should be getting a vector with a length of selected labs(selected_itemids_2plus) * number of days (7) in the observation window. Actual length may be smaller due to missing or infrequent lab-day pairs. Some labs may not be measured on every day for enough patients, so low-coverage features are excluded for robustness and data quality.
    '''
    logger.info(
        f"{days}-dimensional vector with a length of {len(pivot.columns) - 2} (including two identifiers 'hadm_id' & 'subject_id' ) per admission is:\n {(pivot.head())}")  # subtracting "2" from vector length here as we have two identifiers aswell named "hadm_id" & "subject_id"

    return pivot

# Construct the Feature matrix (we are going to create Feature matrix using unimputed, unimputed normalized or imputed vector depending on the type of clustering. Here in this case, creating from the wide-form pivoted data i.e unimputed vector as a sample), excluding identifier columns.
def get_feature_matrix(pivot):
    '''
     "Feature Matrix" is a 2D NumPy array where each Row represents a single admission (hadm_id) & each Column represents a lab measurement at a specific day in the format 'itemid_day' (e.g., 50861_day1, 50861_day2, 50861_day3, etc.).
    '''
    logger.info(
        "Constructing the Feature matrix (it is 2D NumPy array where each row corresponds to an admission (hadm_id) and each column to a specific lab–day measurement) that will serve as the basis for custom pairwise distance calculations to compute different types of Clustering using Distance matrix.\n")

    feature_cols = [col for col in pivot.columns if col not in ['hadm_id', 'subject_id']]
    feature_matrix = pivot[feature_cols].values

    # Converting pivot into long format for proper time-series processing to impute within each admission-lab series
    id_vars = ['hadm_id', 'subject_id']
    feature_cols = [col for col in pivot.columns if col not in id_vars]
    feature_matrix = pivot[feature_cols].values
    return feature_matrix, feature_cols, id_vars


# Transform the pivoted data to long format for per-lab Normalization and Interpolation processing
def create_long_format(pivot, feature_cols, id_vars):

    # Converting into long format: hadm_id, subject_id, itemid_day, valuenum
    long_df = pivot.melt(id_vars=id_vars, value_vars=feature_cols,
                         var_name='itemid_day', value_name='valuenum')

    # Extracting itemid and day from column name
    long_df[['itemid', 'day']] = long_df['itemid_day'].str.extract(r'(\d+)_day(\d+)')
    long_df['itemid'] = long_df['itemid'].astype(int)
    long_df['day'] = long_df['day'].astype(int)

    # Computing lab-wise statistics for normalization (Using observed values only)
    lab_stats = long_df.groupby('itemid')['valuenum'].agg(['mean', 'std']).reset_index()
    lab_stats.columns = ['itemid', 'lab_mean', 'lab_std']

    # Merging stats back to main dataframe
    long_df = long_df.merge(lab_stats, on='itemid', how='left')

    # Normalizing before imputation (per lab)
    long_df['valuenum_normalized'] = (long_df['valuenum'] - long_df['lab_mean']) / long_df['lab_std']

    return long_df


def prepare_unimputed_normalized(long_df: pd.DataFrame, id_vars: list = ['hadm_id', 'subject_id']) -> pd.DataFrame:
    """
    Convert long-format unimputed normalized data to a wide-format DataFrame ready for clustering
    """
    # Pivot normalized values to wide format
    wide_df = long_df.pivot(index='hadm_id', columns='itemid_day', values='valuenum_normalized').reset_index()

    # Add identifiers like 'hadm_id' and 'subject_id' for clarity
    if 'subject_id' in long_df.columns:
        ids_df = long_df[['hadm_id', 'subject_id']].drop_duplicates()
        wide_df = wide_df.merge(ids_df, on='hadm_id', how='left')

    # Reorder columns with ids first, then feature columns sorted by lab id and day for consistency
    def sort_key(col_name):
        try:
            # For columns like '1234_day5'
            parts = col_name.split('_day')
            lab_id = int(parts[0])
            day = int(parts[1])
            return (lab_id, day)
        except Exception:
            # For id columns
            return (-1, -1)

    feature_cols = [col for col in wide_df.columns if col not in id_vars]
    feature_cols_sorted = sorted(feature_cols, key=sort_key)
    cols_ordered = id_vars + feature_cols_sorted
    wide_df = wide_df[cols_ordered]

    return wide_df


# Interpolate missing values in the time series per lab per admission using linear interpolation
def impute_timeseries(group):

    # Sorting by day to maintain temporal order
    group = group.sort_values('day')
    '''
      While applying Linear interpolation (only between existing values), we'll use "limit_direction= both" which already fills the missing values
      before and after known values. Computing ffill and bfill after Linear interpolation does not change the result, so those steps are redundant.
    '''

    # Linear interpolation (only between existing values)
    group['valuenum_imputed'] = group['valuenum_normalized'].interpolate(method='linear', limit_direction='both')

    return group

# Convert the interpolated long-format data back into wide format with lab-day feature columns for each admission
def get_imputed_df(long_df, id_vars):

    # Applying to each admission-lab group
    imputed_df = long_df.groupby(['hadm_id', 'itemid']).apply(impute_timeseries).reset_index(drop=True)

    # Converting Pivot back to wide format
    imputed_df['itemid_day'] = imputed_df['itemid'].astype(str) + '_day' + imputed_df['day'].astype(str)
    wide_df = imputed_df.pivot_table(index=id_vars, columns='itemid_day', values='valuenum_imputed').reset_index()

    return imputed_df, wide_df


# Apply KNN imputation to further fill missing values and sort features by lab and day for consistency
def knn_impute_and_sort_features(wide_df, id_vars,days):

    # Identifying feature columns (all except identifiers)
    feature_cols = [col for col in wide_df.columns if col not in id_vars]

    # Performing KNN imputation if any missing values exist
    if wide_df[feature_cols].isnull().values.any():
        imputer = KNNImputer(n_neighbors=5)
        wide_df[feature_cols] = imputer.fit_transform(wide_df[feature_cols])

    df = wide_df.copy()

    def sort_key(col_name: str):
        itemid_str, day_str = col_name.split('_day')
        return (int(itemid_str), int(day_str))

    # Sorting feature columns by itemid and day
    sorted_feature_cols = sorted(feature_cols, key=sort_key)

    # Reordering DataFrame: identifiers first, then sorted features
    preprocessed_df = df[id_vars + sorted_feature_cols]

    logger.info(
        f"After applying Normalization, Interpolation & KNN Imputation, constructed {days}-dimensional vector with a length  of {len(preprocessed_df.columns) - 2} (including two identifiers 'hadm_id' & 'subject_id' ) per admission is:\n {(preprocessed_df.head())}\n ")  # subtracting "2" from vector length here as we have two identifiers aswell named "hadm_id" & "subject_id"

    return preprocessed_df

# Create and save plots of clustering evaluation metrics (Silhouette, Calinski-Harabasz, Davies-Bouldin) for documentation and review
def plot_and_save_kmeans_metrics(metrics_df, filename, logger=None, folder="metric_results"):
    """
    Plots Silhouette, Calinski-Harabasz, Davies-Bouldin vs number of clusters,
    saves plot to given filename inside folder (default: 'metric_results').
    Creates folder if it does not exist.
    """
    import matplotlib.pyplot as plt
    import os

    # Ensure output directory exists
    if not os.path.exists(folder):
        os.makedirs(folder)
        if logger is not None:
            logger.info(f"Created directory '{folder}' for saving metrics.")

    # Compose full path
    save_path = os.path.join(folder, filename)
    k_vals = metrics_df['k'].tolist()

    plt.figure(figsize=(15, 5))

    # Silhouette score
    plt.subplot(1, 3, 1)
    plt.plot(k_vals, metrics_df['silhouette'], marker='o', color='red')
    plt.title('Silhouette Score')
    plt.xlabel('k')
    plt.grid(True)

    # Calinski-Harabasz index
    plt.subplot(1, 3, 2)
    plt.plot(k_vals, metrics_df['calinski_harabasz'], marker='o', color='blue')
    plt.title('Calinski-Harabasz Index')
    plt.xlabel('k')
    plt.grid(True)

    # Davies-Bouldin index
    plt.subplot(1, 3, 3)
    plt.plot(k_vals, metrics_df['davies_bouldin'], marker='o', color='green')
    plt.title('Davies-Bouldin Index')
    plt.xlabel('k')
    plt.grid(True)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()

    if logger is not None:
        logger.info(f"\nClustering metrics plot named '{filename}' saved to the path {save_path}\n")
    else:
        print(f"\nClustering metrics plot named '{filename}' saved to the path {save_path}\n")

    logger.info(f"\nClustering result for all three used metrics:\n{metrics_df}\n")
    return save_path

# Plot k-distance graphs to help determine the optimal epsilon parameter for DBSCAN clustering
def plot_and_save_kdist(distances: np.ndarray, filename: str, k_label: str = "k-NN distance"):
    plt.figure(figsize=(6, 4))
    plt.plot(distances)
    plt.xlabel("Samples")
    plt.ylabel(k_label)
    plt.title("Elbow Method for Optimal ε")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(filename, dpi=300)
    plt.close()

# Plot k-distance graphs for a precomputed distance matrix to inform DBSCAN parameter choice
def plot_and_save_kdist_precomputed(distances, filename, k_label="k-NN distance"):
    plt.figure(figsize=(6,4))
    plt.plot(distances)
    plt.xlabel("Samples"); plt.ylabel(k_label)
    plt.title("k-Distance Graph (precomputed)")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(filename, dpi=300)
    plt.close()

# Iterate over selected cluster numbers and run Fisher's exact test for each, logging the results
def run_fisher_loop(
    top_clusters: dict,
    fisher_fn,
    base_kwargs: dict,
    logger: logging.Logger,
    method_label: str
) -> dict:
    """
    Iterating over top_clusters dict and run fisher_fn for each k.
    base_kwargs: common kwargs for fisher_fn (we'll add 'k' each time).
    Returns dict { "<metric>_k=<k>": result_df }
    """
    results = {}
    logger.info("Running Fisher’s Exact Test on attained Top-3 clusters...\n")
    for metric_name, ks in top_clusters.items():
        if not ks:
            logger.info("Since no cluster counts identified for %s (%s), therefore Fisher's exact test will not be performed for this metric.\n", metric_name, method_label)
            continue
        for k in ks:
            kwargs = base_kwargs.copy()
            kwargs["k"] = k
            res = fisher_fn(**kwargs)
            key = f"{metric_name}_k={k}"
            results[key] = res
            logger.info("Fisher results for %s k=%d:\n%s\n", metric_name, k, res)
    return results

# Run Fisher's exact test using precomputed distance matrices and corresponding IDs for each selected cluster number
def run_fisher_loop_precomputed(
    top_clusters: dict,
    fisher_fn,
    base_kwargs: dict,
    dist_matrix,
    df_ids: pd.DataFrame,
    logger: logging.Logger,
    method_label: str,
    dist_arg_name: str = "dist_matrix_filled",  # name expected by fisher_fn
    ids_arg_name: str = "df_ids"
) -> dict:
    """
    Same idea as run_fisher_loop, but injects a precomputed distance matrix
    and an id DataFrame each time.
    """
    results = {}
    logger.info("Running Fisher’s Exact Test on attained Top-3 clusters...\n")
    for metric_name, ks in top_clusters.items():
        if not ks:
            logger.info("Since no cluster counts identified for %s (%s), therefore Fisher's exact test will not be performed for this metric.\n", metric_name, method_label)
            continue
        for k in ks:
            kwargs = base_kwargs.copy()
            kwargs["k"] = k
            kwargs[dist_arg_name] = dist_matrix
            kwargs[ids_arg_name] = df_ids
            res = fisher_fn(**kwargs)
            key = f"{metric_name}_k={k}"
            results[key] = res
            logger.info("%s Fisher results for %s k=%d:\n%s\n", method_label, metric_name, k, res)
    return results


# Run DBSCAN clustering with Euclidean distance metric, generate k-distance elbow plot, cluster, compute metrics, and perform Fisher’s exact test
def run_dbscan_block_euclidean(
    df_features: pd.DataFrame,
    id_vars: list[str],
    k_for_elbow: int,
    eps: float,
    min_samples: int,
    outcome_df: pd.DataFrame,
    outcome_col: str,
    logger: logging.Logger,
    prefix: str = "DBSCAN-Euclidean",
    output_folder: str = "metric_results"
):
    """
    1) Build feature matrix X
    2) Compute k-distance for elbow method and save plot in output_folder
    3) Run DBSCAN clustering and compute metrics, Fisher's test
    Returns (metrics_df, labels, fisher_dict)
    """

    # Extract feature columns (exclude id_vars)
    feature_cols = [c for c in df_features.columns if c not in id_vars]
    X = df_features[feature_cols].values

    # Compute k-distance for k-for-elbow nearest neighbors using sklearn's NearestNeighbors
    from sklearn.neighbors import NearestNeighbors
    nbrs = NearestNeighbors(n_neighbors=k_for_elbow).fit(X)
    distances, _ = nbrs.kneighbors(X)
    kdist = np.sort(distances[:, k_for_elbow - 1])

    # Ensure that the output folder for saving plots and results exists
    os.makedirs(output_folder, exist_ok=True)

    # Compose full path for saving the k-distance plot
    plot_filename = os.path.join(output_folder, f"{prefix.lower()}_k_distance_graph.png")

    # Plot and save k-distance graph
    plt.figure()
    plt.plot(kdist)
    plt.xlabel(f"{k_for_elbow}-NN distance")
    plt.ylabel("Distance")
    plt.title(f"{prefix} - k-distance Plot")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(plot_filename)
    plt.close()
    logger.info(f"Saved k-distance graph to {plot_filename}")

    # Run DBSCAN clustering and metrics calculation
    metrics_df, labels, fisher_dict = run_dbscan_euclidean(
        df_features=df_features,
        id_vars=id_vars,
        eps=eps,
        min_samples=min_samples,
        outcome_df=outcome_df,
        outcome_col=outcome_col,

    )
    # Log the clustering metrics and Fisher test results for tracking
    logger.info(f"{prefix} metrics:\n{metrics_df}")
    if fisher_dict and "results" in fisher_dict:
        logger.info(f"{prefix} Fisher results:\n{fisher_dict['results']}")

    # Return the metrics, cluster labels, and Fisher's test results
    return metrics_df, labels, fisher_dict


# For each selected cluster number, perform spectral clustering, then run Fisher's test on the resulting groupings
def run_fisher_loop_spectral(
    top_clusters: dict,
    feature_matrix: np.ndarray,
    ids_df: pd.DataFrame,
    outcome_df: pd.DataFrame,
    outcome_col: str,
    logger: logging.Logger,
    method_label: str = "Spectral",
    affinity: str = "nearest_neighbors",
    random_state: int = 42,
) -> dict:
    """
    For each k in top_clusters, fit SpectralClustering, then run Fisher.
    Returns { "<metric>_k=<k>": result_df }
    """
    results = {}
    logger.info("Running Fisher’s Exact Test on attained Top-3 clusters...\n")
    for metric_name, ks in top_clusters.items():
        if not ks:
            logger.info("Since no cluster counts identified for %s (%s), therefore Fisher's exact test will not be performed for this metric.\n", metric_name, method_label)
            continue
        for k in ks:
            spec = SpectralClustering(n_clusters=k, affinity=affinity, random_state=random_state)
            labels = spec.fit_predict(feature_matrix)
            res = run_fisher_test_spectral(labels, ids_df, outcome_df, outcome_col)
            key = f"{metric_name}_k={k}"
            results[key] = res
            logger.info("Fisher results for %s k=%d:\n%s\n", metric_name, k, res)
    return results


def spectral_metrics_block(
    feature_matrix: np.ndarray,
    k_values: range,
    affinity: str,
    plot_filename: str,
    sil_threshold: float,
    logger: logging.Logger,
    folder: str = "metric_results"
):
    """Run spectral metrics, plot curves, pick top clusters."""
    metrics_df = run_spectral_metrics(
        feature_matrix=feature_matrix,
        k_values=k_values,
        affinity=affinity,
        logger=logger
    )
    # Pass the folder parameter instead of hardcoding
    utils.plot_and_save_kmeans_metrics(metrics_df, filename=plot_filename, logger=logger, folder=folder)
    top_clust = get_top_clusters_with_threshold(metrics_df, sil_threshold, logger=logger)
    return metrics_df, top_clust



def should_run(name: str, run_all: bool, flags: dict[str, bool]) -> bool:
    return run_all or flags.get(name, False)

#saving distance matrices
def save_distance_matrix(matrix: np.ndarray,
                         name: str,
                         out_dir: str = "distance_matrices") -> str:
    """
    Save a distance matrix as a NumPy .npy file.

    Args:
        matrix: 2D NumPy array of distances (after any fill/abs).
        name:   Base filename (no “.npy”); should describe this metric.
        out_dir:Directory where .npy files will be written (created if needed).

    Returns:
        Full path to the saved .npy file.
    """
    os.makedirs(out_dir, exist_ok=True)
    filepath = os.path.join(out_dir, f"{name}.npy")
    np.save(filepath, matrix)
    return filepath




