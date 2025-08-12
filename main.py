'''
                ======================================================   MAIN PIPELINE MODULE   ==================================================
'''

"""
This is the primary pipeline script for the project. It orchestrates the end-to-end analysis workflow, serving as the central entry point where all
key processes, configuration parameters, and workflow logic are put together.

Key responsibilities include:
- Importing project configuration settings (hyperparameters), utility routines, and clustering and statistical analysis modules.
- Loading, merging, and caching clinical cohort and lab event datasets to optimize performance.
- Preprocessing data by filtering common labs based on coverage thresholds, performing temporal discretization, normalization, interpolation,
  and imputation to produce feature vectors in multiple formats (unimputed pivot vector, numeric arrays, fully imputed feature vector).
- Managing a modular, flag-driven execution framework that dynamically invokes diverse clustering algorithms (e.g., KMeans, Agglomerative with
  various distance metrics, Binary distances, DBSCAN, Spectral, KMedoids) based on configurable analysis flags.
- Computing, caching, and loading precomputed distance matrices and intermediate outputs for computational efficiency and reproducibility.
- Evaluating clustering results with multiple metrics (Silhouette score, Calinski-Harabasz, Davies-Bouldin indices) and conducting statistical
  significance testing (e.g., Fisher's Exact Test) on cluster associations with clinical outcomes.
- Saving all key outputs—feature matrices, clustering metrics plots, distance matrices, and test results—in structured folders to maintain
  project organization.
- Logging all major processing steps, runtime metrics, and results systematically to support transparency, auditability, and collaborative
  development.
- Supporting interactive iterative development through module reloading to apply code updates without restarting the environment.

This design enables reproducible, transparent, and extensible experiments with standardized data processing, multiple clustering techniques,
and rigorous evaluation, forming a robust foundation for clinical data-driven phenotyping and analysis.
"""



#   ================================================================= IMPORTS AND CONFIGURATION ============================================================================

from hyperparameters import *         # Import all analysis hyperparameters and pipeline flags
import utils                          # Import all utility functions (data loading, filtering, etc.)
from imports import *                 # Import packages (pandas, numpy, etc.)
from clustering import *              # Import clustering method implementations


utils.configure_logging()             # Set up standardized logging format for the pipeline
logger = logging.getLogger('MAIN')


#   ====================================================================== DATA LOADING ====================================================================================

def load_data():
    """
    Load or cache cohort and labevents dataframes.
    Cached .pkl files are loaded if present, otherwise original CSVs are loaded and cached for the future.
    """

    start = time.time()
    if utils.check_existing_raw_data():
        cohort_data = pd.read_pickle(os.path.join(utils.output_folder, 'cohort_data.pkl'))
        labevents_data = pd.read_pickle(os.path.join(utils.output_folder, 'labevents_data.pkl'))

    else:
        # These paths should be adapted if data is relocated
        data_path_cohort = "C:/Ahad/Project/data/cohort1_target.csv"         # Adjust path as needed
        data_path_labevent = "C:/Ahad/Project/data/labevents.csv"            # Adjust path as needed
        cohort_data = utils.loading_data(data_path_cohort,  'dischtime')
        labevents_data = utils.loading_data(data_path_labevent, 'charttime')
        utils.save_raw_dataframes(cohort_data, labevents_data)
        elapsed_time = time.time() - start
        logger.info("Execution time to load the data: %d minutes and %d seconds",
                    int(elapsed_time // 60), int(elapsed_time % 60))

    # Quick logging of the data head for sanity check 
    logger.info(f"Cohort data:\n{cohort_data.head()}\n")
    logger.info(f"Lab events data:\n{labevents_data.head()}\n")

    return cohort_data, labevents_data


#    ====================================================================== DATA PREPROCESSING =========================================================================

def preprocess(cohort_data, labevents_data, set_percentage, days, PCT_LABEL):
    """
    Preprocess, cache, and return the data needed for clustering.
    Handles both cached and fresh computation, and saves results for future runs.
    """

    if utils.check_existing_preprocessed_files(label=PCT_LABEL):

        # Load from cache for speed/reproducibility
        filtered_common_labs_percent = pd.read_pickle(os.path.join(utils.output_folder, f'filtered_common_labs_{PCT_LABEL}.pkl'))
        pivot_percent = pd.read_pickle(os.path.join(utils.output_folder, f'unimputed_vector_{PCT_LABEL}.pkl'))
        feature_matrix_percent = pd.read_pickle(os.path.join(utils.output_folder, f'feature_matrix_{PCT_LABEL}.pkl'))
        imputed_preprocessed = pd.read_pickle(os.path.join(utils.output_folder, f'imputed_vector_{PCT_LABEL}.pkl'))
        id_vars = ['hadm_id', 'subject_id']

    else:

        # Run full preprocessing pipeline if results are not found in cache
        merged = utils.merge_data(labevents_data, cohort_data)
        logger.info("Data merged successfully")

        result = utils.labs_within_n_days_of_discharge(merged, days)
        logger.info("Filtered lab results are successfully taken within the last %d days.\n", days)

        filtered_common_labs_percent, unique_labs, total_admissions = utils.filtering_labs_by_percentage(result, set_percentage, days)
        pivot_percent = utils.descritization(filtered_common_labs_percent, days)
        feature_matrix_percent, feature_cols_percent, id_vars = utils.get_feature_matrix(pivot_percent)
        normalized_long_df_percent = utils.create_long_format(pivot_percent, feature_cols_percent, id_vars)
        imputed_df_percent, wide_df_percent = utils.get_imputed_df(normalized_long_df_percent, id_vars)
        imputed_preprocessed = utils.knn_impute_and_sort_features(wide_df_percent, id_vars, days)
        feature_matrix_df = pd.DataFrame(feature_matrix_percent, columns=feature_cols_percent)

        utils.save_preprocessing_outputs(
            filtered_common_labs_percent,
            pivot_percent,
            feature_matrix_df,
            imputed_preprocessed,
            label=PCT_LABEL
        )
        
    return filtered_common_labs_percent, pivot_percent, feature_matrix_percent, imputed_preprocessed, id_vars


#    ==================================================================== CLUSTERING AND EVALUATION =======================================================================

def run_clustering(filtered_common_labs_percent, pivot_percent, feature_matrix_percent, imputed_preprocessed, id_vars, cohort_data, PCT_LABEL):

    """
    Executes all clustering algorithms and evaluation logic based on analysis flags.
    Handles caching of metrics, plotting, and statistical significance tests for clusters.
    Metric plots are always saved in folder 'metric_results'.
    """

# ==================================================
#     K-MEANS Clustering using 'euclidean' metric
# ==================================================

    # Run standard KMeans clustering using 'euclidean' metric
    if utils.should_run("KMEANS", RUN_ALL, RUN_FLAGS):
        metrics_df_km = run_kmeans_metrics(imputed_preprocessed, id_vars, K_RANGE, logger=logger)

        # Create per-method/per-threshold output directory
        Kmeans_dir = os.path.join("metric_results", f"{PCT_LABEL}_threshold", "Kmeans_clustering_imputed")
        os.makedirs(Kmeans_dir, exist_ok=True)

        # The metrics are saved as PNG and CSV for documentation and later review
        utils.plot_and_save_kmeans_metrics(
            metrics_df_km,
            filename="Kmeans_metrics_graph.png",
            logger=logger,
            folder=Kmeans_dir
        )

        metrics_df_km.to_csv(
            os.path.join(Kmeans_dir, "Kmeans_metrics_score.csv"),
            index=False
        )
        # Top clusters per metric (Silhouette score, Calinski-Harabasz index and Davies-Bouldin index)
        top_clust_km = get_top_clusters_with_threshold(metrics_df_km, SIL_THR, logger=logger)

        # Run Fisher's Exact test for each selected cluster count
        all_fisher_km = utils.run_fisher_loop(
            top_clusters=top_clust_km,
            fisher_fn=run_fisher_test,
            base_kwargs=dict(
                df_features=imputed_preprocessed,
                id_vars=id_vars,
                outcome_df=cohort_data,
                outcome_col='target',
                random_state=42,
                n_init=10
            ),
            logger=logger,
            method_label=f"KMeans-{PCT_LABEL}"
        )

        # Save Fisher's test results CSVs
        for k, fisher_df in all_fisher_km.items():
            fisher_csv_path = os.path.join(
                Kmeans_dir,
                f"Kmeans_fisher_test_{k}.csv"
            )
            fisher_df.to_csv(fisher_csv_path, index=False)
    else:
        metrics_df_km = top_clust_km = all_fisher_km = None


# ==============================================================
#    Agglomerative Clustering (Ward) using 'euclidean' metric
# ==============================================================

    # Run Agglomerative clustering (Ward/euclidean) using 'euclidean' metric
    if utils.should_run("AGG_EUCLID", RUN_ALL, RUN_FLAGS):
        metrics_df_agg_eu = run_agglomerative_metrics_euclidean(imputed_preprocessed, id_vars, K_RANGE, logger=logger)

        # Create per-method/per-threshold output directory
        agg_ward_dir = os.path.join("metric_results", f"{PCT_LABEL}_threshold", "Agglomerative_clustering_ward_imputed")
        os.makedirs(agg_ward_dir, exist_ok=True)

        # The metrics are saved as PNG and CSV for documentation and later review
        utils.plot_and_save_kmeans_metrics(
            metrics_df_agg_eu,
            filename="Agglomerative_ward_metrics_graph.png",
            logger=logger,
            folder=agg_ward_dir
        )

        metrics_df_agg_eu.to_csv(
            os.path.join(agg_ward_dir, "Agglomerative_ward_metrics_score.csv"),
            index=False
        )

        # Top clusters per metric (Silhouette score, Calinski-Harabasz index and Davies-Bouldin index)
        top_clust_agg_eu = get_top_clusters_with_threshold(metrics_df_agg_eu, SIL_THR, logger=logger)

        # Run Fisher's Exact test for each selected cluster count
        all_fisher_agg_eu = utils.run_fisher_loop(
            top_clusters=top_clust_agg_eu,
            fisher_fn=run_fisher_test_agglomerative_euclidean,
            base_kwargs=dict(
                df_features=imputed_preprocessed,
                id_vars=id_vars,
                outcome_df=cohort_data,
                outcome_col="target",
                linkage="ward",
                metric="euclidean"
            ),
            logger=logger,
            method_label=f"Agglo-Euclidean-{PCT_LABEL}"
        )

        # Save Fisher's test results CSVs
        for k, fisher_df in all_fisher_agg_eu.items():
            fisher_csv_path = os.path.join(
                agg_ward_dir,
                f"Agglomerative_ward_fisher_test_{k}.csv"
            )
            fisher_df.to_csv(fisher_csv_path, index=False)
    else:
        metrics_df_agg_eu = top_clust_agg_eu = all_fisher_agg_eu = None


# ==================================================================================
#    Agglomerative Clustering using Manhattan Distance matrix (Unimputed vector)
# ==================================================================================

    # Run Agglomerative clustering with precomputed Manhattan distance matrix
    if utils.should_run("AGG_MANHATTAN_UNIMPUTED", RUN_ALL, RUN_FLAGS):
        metrics_df_agg_man, dist_man = run_agglomerative_metrics_manhattan(
            df_raw=pivot_percent,
            id_vars=id_vars,
            k_values=K_RANGE,
            df_for_space=imputed_preprocessed,
            logger=logger
        )

        # Create per-method/per-threshold output directory
        manhattan_dir = os.path.join("metric_results", f"{PCT_LABEL}_threshold", "Agglomerative_manhattan_clustering_unimputed")
        os.makedirs(manhattan_dir, exist_ok=True)

        # The metrics are saved as PNG and CSV for documentation and later review
        utils.plot_and_save_kmeans_metrics(
            metrics_df_agg_man,
            filename="Manhattan_unimputed_metrics_graph.png",
            logger=logger,
            folder=manhattan_dir
        )

        metrics_df_agg_man.to_csv(
            os.path.join(manhattan_dir, "Manhattan_unimputed_metrics_score.csv"),
            index=False
        )

        # Top clusters per metric (Silhouette score, Calinski-Harabasz index and Davies-Bouldin index)
        top_clust_agg_man = get_top_clusters_with_threshold(metrics_df_agg_man, SIL_THR, logger=logger)

        # Run Fisher's Exact test for each selected cluster count
        all_fisher_agg_man = utils.run_fisher_loop_precomputed(
            top_clusters=top_clust_agg_man,
            fisher_fn=run_fisher_test_agglomerative_manhattan,
            base_kwargs=dict(outcome_df=cohort_data, outcome_col="target"),
            dist_matrix=dist_man,
            df_ids=pivot_percent[id_vars],
            logger=logger,
            method_label=f"Agglo-Manhattan-{PCT_LABEL}"
        )

        # Save Fisher's test results CSVs
        for k, fisher_df in all_fisher_agg_man.items():
            fisher_csv_path = os.path.join(
                manhattan_dir,
                f"Manhattan_unimputed_fisher_test_{k}.csv"
            )
            fisher_df.to_csv(fisher_csv_path, index=False)
    else:
        metrics_df_agg_man = dist_man = top_clust_agg_man = all_fisher_agg_man = None


# # ==================================================================================
# #    Agglomerative Clustering using Manhattan Distance matrix (Imputed vector)
# # ==================================================================================

    # Run Agglomerative clustering with precomputed Manhattan distance matrix using imputed vector
    if utils.should_run("AGG_MANHATTAN_IMPUTED", RUN_ALL, RUN_FLAGS):

        # Create per-method/per-threshold output directory
        manhattan_dir = os.path.join("metric_results", f"{PCT_LABEL}_threshold", "Agglomerative_manhattan_clustering_imputed")
        os.makedirs(manhattan_dir, exist_ok=True)

        metrics_df_agg_man_imputed, dist_man = run_agglomerative_metrics_manhattan_imputed(
            df_raw=imputed_preprocessed,
            id_vars=id_vars,
            k_values=K_RANGE,
            df_for_space=imputed_preprocessed,
            logger=logger
        )

        # The metrics are saved as PNG and CSV for documentation and later review
        utils.plot_and_save_kmeans_metrics(
            metrics_df_agg_man_imputed,
            filename="Manhattan_imputed_metrics_graph.png",
            logger=logger,
            folder=manhattan_dir
        )
        metrics_df_agg_man_imputed.to_csv(os.path.join(manhattan_dir, "Manhattan_imputed_metrics_score.csv"), index=False)

        # Top clusters per metric (Silhouette score, Calinski-Harabasz index and Davies-Bouldin index)
        top_clust_agg_man = get_top_clusters_with_threshold(metrics_df_agg_man_imputed, SIL_THR, logger=logger)

        # Run Fisher's Exact test for each selected cluster count
        all_fisher_agg_man = utils.run_fisher_loop_precomputed(
            top_clusters=top_clust_agg_man,
            fisher_fn=run_fisher_test_agglomerative_manhattan_imputed,
            base_kwargs=dict(outcome_df=cohort_data, outcome_col="target"),
            dist_matrix=dist_man,
            df_ids=imputed_preprocessed[id_vars],
            logger=logger,
            method_label=f"Agglo-Manhattan-Imputed-{PCT_LABEL}"
        )

        # Save Fisher's test results CSVs
        for k, fisher_df in all_fisher_agg_man.items():
            fisher_csv_path = os.path.join(manhattan_dir, f"Manhattan_imputed_fisher_test_{k}.csv")
            fisher_df.to_csv(fisher_csv_path, index=False)
    else:
        metrics_df_agg_man_imputed = dist_man = top_clust_agg_man = all_fisher_agg_man = None


# =================================================================================
#    Agglomerative Clustering using Mahalanobis Distance matrix (Unimputed vector)
# =================================================================================

    # Run Agglomerative clustering with precomputed Mahalanobis distance matrix using unimputed vector
    if utils.should_run("AGG_MAHALANOBIS_UNIMPUTED", RUN_ALL, RUN_FLAGS):

        # Create per-method/per-threshold output directory
        mahalanobis_dir = os.path.join("metric_results", f"{PCT_LABEL}_threshold", "Agglomerative_mahalanobis_clustering_unimputed")
        os.makedirs(mahalanobis_dir, exist_ok=True)

        metrics_df_agg_mah, dist_mah = run_agglomerative_metrics_mahalanobis(
            df_raw=pivot_percent,
            id_vars=id_vars,
            k_values=K_RANGE,
            df_for_space=imputed_preprocessed,
            logger=logger
        )

        # The metrics are saved as PNG and CSV for documentation and later review
        utils.plot_and_save_kmeans_metrics(
            metrics_df_agg_mah,
            filename="Mahalanobis_unimputed_metrics_graph.png",
            logger=logger,
            folder=mahalanobis_dir
        )
        metrics_df_agg_mah.to_csv(os.path.join(mahalanobis_dir, "Mahalanobis_unimputed_metrics_score.csv"), index=False)

        # Top clusters per metric (Silhouette score, Calinski-Harabasz index and Davies-Bouldin index)
        top_clust_agg_mah = get_top_clusters_with_threshold(metrics_df_agg_mah, SIL_THR, logger=logger)

        # Run Fisher's Exact test for each selected cluster count
        all_fisher_agg_mah = utils.run_fisher_loop_precomputed(
            top_clusters=top_clust_agg_mah,
            fisher_fn=run_fisher_test_agglomerative_mahalanobis,
            base_kwargs=dict(outcome_df=cohort_data, outcome_col="target"),
            dist_matrix=dist_mah,
            df_ids=pivot_percent[id_vars],
            logger=logger,
            method_label=f"Agglo-Mahalanobis-{PCT_LABEL}"
        )

         # Save Fisher's test results CSVs
        for k, fisher_df in all_fisher_agg_mah.items():
            fisher_csv_path = os.path.join(mahalanobis_dir, f"Mahalanobis_unimputed_fisher_test_{k}.csv")
            fisher_df.to_csv(fisher_csv_path, index=False)
    else:
        metrics_df_agg_mah = dist_mah = top_clust_agg_mah = all_fisher_agg_mah = None


# ================================================================================
#    Agglomerative Clustering using Mahalanobis Distance matrix (Imputed vector)
# ================================================================================

    # Run Agglomerative clustering with precomputed Mahalanobis distance matrix using imputed vector
    if utils.should_run("AGG_MAHALANOBIS_IMPUTED", RUN_ALL, RUN_FLAGS):

        # Create per-method/per-threshold output directory
        mahalanobis_imp_dir = os.path.join("metric_results", f"{PCT_LABEL}_threshold", "Agglomerative_mahalanobis_clustering_imputed")
        os.makedirs(mahalanobis_imp_dir, exist_ok=True)

        metrics_df_agg_mah_imputed, dist_mah = run_agglomerative_metrics_mahalanobis_imupted(
            df_raw=imputed_preprocessed,
            id_vars=id_vars,
            k_values=K_RANGE,
            df_for_space=imputed_preprocessed,
            logger=logger
        )

        # The metrics are saved as PNG and CSV for documentation and later review
        utils.plot_and_save_kmeans_metrics(
            metrics_df_agg_mah_imputed,
            filename="Mahalanobis_imputed_metrics_graph.png",
            logger=logger,
            folder=mahalanobis_imp_dir
        )
        metrics_df_agg_mah_imputed.to_csv(os.path.join(mahalanobis_imp_dir, "Mahalanobis_imputed_metrics_score.csv"), index=False)

        # Top clusters per metric (Silhouette score, Calinski-Harabasz index and Davies-Bouldin index)
        top_clust_agg_mah = get_top_clusters_with_threshold(metrics_df_agg_mah_imputed, SIL_THR, logger=logger)

        # Run Fisher's Exact test for each selected cluster count
        all_fisher_agg_mah = utils.run_fisher_loop_precomputed(
            top_clusters=top_clust_agg_mah,
            fisher_fn=run_fisher_test_agglomerative_mahalanobis_imputed,
            base_kwargs=dict(outcome_df=cohort_data, outcome_col="target"),
            dist_matrix=dist_mah,
            df_ids=imputed_preprocessed[id_vars],
            logger=logger,
            method_label=f"Agglo-Mahalanobis-Imputed-{PCT_LABEL}"
        )

        # Save Fisher's test results CSVs
        for k, fisher_df in all_fisher_agg_mah.items():
            fisher_csv_path = os.path.join(mahalanobis_imp_dir, f"Mahalanobis_imputed_fisher_test_{k}.csv")
            fisher_df.to_csv(fisher_csv_path, index=False)
    else:
        metrics_df_agg_mah_imputed = dist_mah = top_clust_agg_mah = all_fisher_agg_mah = None


# ===============================================================================
#    Agglomerative Clustering using Cosine Distance matrix (Unimputed vector)
# ===============================================================================

    # Run Agglomerative clustering with precomputed Cosine distance matrix using unimputed vector
    if utils.should_run("AGG_COSINE_UNIMPUTED", RUN_ALL, RUN_FLAGS):

        # Create per-method/per-threshold output directory
        cosine_dir = os.path.join("metric_results", f"{PCT_LABEL}_threshold", "Agglomertaive_cosine_clustering_unimputed")
        os.makedirs(cosine_dir, exist_ok=True)

        metrics_df_agg_cos, dist_cos = run_agglomerative_metrics_cosine(
            df_raw=pivot_percent,
            id_vars=id_vars,
            k_values=K_RANGE,
            df_for_space=imputed_preprocessed,
            logger=logger
        )

        # The metrics are saved as PNG and CSV for documentation and later review
        utils.plot_and_save_kmeans_metrics(
            metrics_df_agg_cos,
            filename="Cosine_unimputed_metrics_graph.png",
            logger=logger,
            folder=cosine_dir
        )
        metrics_df_agg_cos.to_csv(os.path.join(cosine_dir, "Cosine_unimputed_metrics_score.csv"), index=False)

        # Top clusters per metric (Silhouette score, Calinski-Harabasz index and Davies-Bouldin index)
        top_clust_agg_cos = get_top_clusters_with_threshold(metrics_df_agg_cos, SIL_THR, logger=logger)

        # Run Fisher's Exact test for each selected cluster count
        all_fisher_agg_cos = utils.run_fisher_loop_precomputed(
            top_clusters=top_clust_agg_cos,
            fisher_fn=run_fisher_test_agglomerative_cosine,
            base_kwargs=dict(outcome_df=cohort_data, outcome_col="target"),
            dist_matrix=dist_cos,
            df_ids=pivot_percent[id_vars],
            logger=logger,
            method_label=f"Agglo-Cosine-{PCT_LABEL}"
        )

        # Save Fisher's test results CSVs
        for k, fisher_df in all_fisher_agg_cos.items():
            fisher_csv_path = os.path.join(cosine_dir, f"Cosine_unimputed_fisher_test_{k}.csv")
            fisher_df.to_csv(fisher_csv_path, index=False)
    else:
        metrics_df_agg_cos = dist_cos = top_clust_agg_cos = all_fisher_agg_cos = None


# ==============================================================================
#    Agglomerative Clustering using Cosine Distance matrix (Imputed vector)
# ==============================================================================

    # Run Agglomerative clustering with precomputed Cosine distance matrix using imputed data
    if utils.should_run("AGG_COSINE_IMPUTED", RUN_ALL, RUN_FLAGS):

        # Create per-method/per-threshold output directory
        cosine_imp_dir = os.path.join("metric_results", f"{PCT_LABEL}_threshold", "Agglomerative_cosine_clustering_imputed")
        os.makedirs(cosine_imp_dir, exist_ok=True)

        metrics_df_agg_cos, dist_cos = run_agglomerative_metrics_cosine_imputed(
            df_raw=imputed_preprocessed,
            id_vars=id_vars,
            k_values=K_RANGE,
            df_for_space=imputed_preprocessed,
            logger=logger
        )

        # The metrics are saved as PNG and CSV for documentation and later review
        utils.plot_and_save_kmeans_metrics(
            metrics_df_agg_cos,
            filename="Cosine_imputed_metrics_graph.png",
            logger=logger,
            folder=cosine_imp_dir
        )
        metrics_df_agg_cos.to_csv(os.path.join(cosine_imp_dir, "Cosine_imputed_metrics_score.csv"), index=False)

        # Top clusters per metric (Silhouette score, Calinski-Harabasz index and Davies-Bouldin index)
        top_clust_agg_cos = get_top_clusters_with_threshold(metrics_df_agg_cos, SIL_THR, logger=logger)

        # Run Fisher's Exact test for each selected cluster count
        all_fisher_agg_cos = utils.run_fisher_loop_precomputed(
            top_clusters=top_clust_agg_cos,
            fisher_fn=run_fisher_test_agglomerative_cosine_imputed,
            base_kwargs=dict(outcome_df=cohort_data, outcome_col="target"),
            dist_matrix=dist_cos,
            df_ids=imputed_preprocessed[id_vars],
            logger=logger,
            method_label=f"Agglo-Cosine-Imputed-{PCT_LABEL}"
        )

        # Save Fisher's test results CSVs
        for k, fisher_df in all_fisher_agg_cos.items():
            fisher_csv_path = os.path.join(cosine_imp_dir, f"Cosine_imputed_fisher_test_{k}.csv")
            fisher_df.to_csv(fisher_csv_path, index=False)
    else:
        metrics_df_agg_cos = dist_cos = top_clust_agg_cos = all_fisher_agg_cos = None



# ================================================================================
#    Agglomerative Clustering using Euclidean Distance matrix (Unimputed vector)
# ================================================================================

    # Run Agglomerative clustering with precomputed Euclidean distance matrix using unimputed vector
    if utils.should_run("AGG_EUCLID_UNIMPUTED", RUN_ALL, RUN_FLAGS):

        # Create per-method/per-threshold output directory
        euclid_dir = os.path.join("metric_results", f"{PCT_LABEL}_threshold", "Agglomerative_euclidean_clustering_unimputed")
        os.makedirs(euclid_dir, exist_ok=True)

        metrics_df_agg_eu_pre, dist_eu_pre = run_agglomerative_metrics_euclidean_precomputed(
            df_raw=pivot_percent,
            id_vars=id_vars,
            k_values=K_RANGE,
            df_for_space=imputed_preprocessed,
            logger=logger
        )

        # The metrics are saved as PNG and CSV for documentation and later review
        utils.plot_and_save_kmeans_metrics(
            metrics_df_agg_eu_pre,
            filename="Euclidean_unimputed_metrics_graph.png",
            logger=logger,
            folder=euclid_dir
        )
        metrics_df_agg_eu_pre.to_csv(os.path.join(euclid_dir, "Euclidean_unimputed_metrics_score.csv"), index=False)

        # Top clusters per metric (Silhouette score, Calinski-Harabasz index and Davies-Bouldin index)
        top_clust_agg_eu_pre = get_top_clusters_with_threshold(metrics_df_agg_eu_pre, SIL_THR, logger=logger)

        # Run Fisher's Exact test for each selected cluster count
        all_fisher_agg_eu_pre = utils.run_fisher_loop_precomputed(
            top_clusters=top_clust_agg_eu_pre,
            fisher_fn=run_fisher_test_agglomerative_euclidean_precomputed,
            base_kwargs=dict(outcome_df=cohort_data, outcome_col="target"),
            dist_matrix=dist_eu_pre,
            df_ids=pivot_percent[id_vars],
            logger=logger,
            method_label=f"Agglo-Euclid-precomputed-{PCT_LABEL}"
        )

        # Save Fisher's test results CSVs
        for k, fisher_df in all_fisher_agg_eu_pre.items():
            fisher_csv_path = os.path.join(euclid_dir, f"Euclidean_unimputed_fisher_test_{k}.csv")
            fisher_df.to_csv(fisher_csv_path, index=False)
    else:
        metrics_df_agg_eu_pre = dist_eu_pre = top_clust_agg_eu_pre = all_fisher_agg_eu_pre = None


# ================================================================================
#    Agglomerative Clustering using Euclidean Distance matrix (Imputed vector)
# ================================================================================

    # Run Agglomerative clustering with precomputed Euclidean distance matrix using imputed vector
    if utils.should_run("AGG_EUCLID_IMPUTED", RUN_ALL, RUN_FLAGS):

        # Create per-method/per-threshold output directory
        euclid_imp_dir = os.path.join("metric_results", f"{PCT_LABEL}_threshold", "Agglomerative_euclidean_clustering_imputed")
        os.makedirs(euclid_imp_dir, exist_ok=True)

        metrics_df_agg_eu_imputed, dist_eu_imputed = run_agglomerative_metrics_euclidean_imputed(
            df_raw=imputed_preprocessed,
            id_vars=id_vars,
            k_values=K_RANGE,
            df_for_space=imputed_preprocessed,
            logger=logger
        )

        # The metrics are saved as PNG and CSV for documentation and later review
        utils.plot_and_save_kmeans_metrics(
            metrics_df_agg_eu_imputed,
            filename="Euclidean_imputed_metrics_graph.png",
            logger=logger,
            folder=euclid_imp_dir
        )
        metrics_df_agg_eu_imputed.to_csv(os.path.join(euclid_imp_dir, "Euclidean_imputed_metrics_score.csv"), index=False)

        # Top clusters per metric (Silhouette score, Calinski-Harabasz index and Davies-Bouldin index)
        top_clust_agg_eu_imputed = get_top_clusters_with_threshold(metrics_df_agg_eu_imputed, SIL_THR, logger=logger)

        # Run Fisher's Exact test for each selected cluster count
        all_fisher_agg_eu_imputed = utils.run_fisher_loop_precomputed(
            top_clusters=top_clust_agg_eu_imputed,
            fisher_fn=run_fisher_test_agglomerative_euclidean_imputed,
            base_kwargs=dict(outcome_df=cohort_data, outcome_col="target"),
            dist_matrix=dist_eu_imputed,
            df_ids=imputed_preprocessed[id_vars],
            logger=logger,
            method_label=f"Agglo-Euclidean-Imputed-{PCT_LABEL}"
        )

        # Save Fisher's test results CSVs
        for k, fisher_df in all_fisher_agg_eu_imputed.items():
            fisher_csv_path = os.path.join(euclid_imp_dir, f"Euclidean_imputed_fisher_test_{k}.csv")
            fisher_df.to_csv(fisher_csv_path, index=False)
    else:
        metrics_df_agg_eu_imputed = dist_eu_imputed = top_clust_agg_eu_imputed = all_fisher_agg_eu_imputed = None


# ====================================================================
#    Agglomerative Clustering using DTW matrix (fast, dtaidistance)
# ====================================================================

    # Run Agglomerative clustering with precomputed DTW distance matrix using dtaidistance
    if utils.should_run("AGG_DTW_FAST", RUN_ALL, RUN_FLAGS):

        # Create per-method/per-threshold output directory
        dtw_dir = os.path.join("metric_results", f"{PCT_LABEL}_threshold", "Agglomerative_clustering_DTW_Fast_imputed")
        os.makedirs(dtw_dir, exist_ok=True)

        metrics_df_agg_dtw, dtw_mat = run_agglomerative_metrics_dtw(
            df_features=imputed_preprocessed,
            id_vars=id_vars,
            n_timesteps=days,
            k_values=K_RANGE,
            df_for_space=imputed_preprocessed,
            logger=logger
        )

        # The metrics are saved as PNG and CSV for documentation and later review
        utils.plot_and_save_kmeans_metrics(
            metrics_df_agg_dtw,
            filename="DTW_Fast_metrics_graph.png",
            logger=logger,
            folder=dtw_dir
        )
        metrics_df_agg_dtw.to_csv(os.path.join(dtw_dir, "DTW_Fast_metrics_score.csv"), index=False)

        # Top clusters per metric (Silhouette score, Calinski-Harabasz index and Davies-Bouldin index)
        top_clust_agg_dtw = get_top_clusters_with_threshold(metrics_df_agg_dtw, SIL_THR, logger=logger)

        # Run Fisher's Exact test for each selected cluster count
        all_fisher_agg_dtw = utils.run_fisher_loop_precomputed(
            top_clusters=top_clust_agg_dtw,
            fisher_fn=run_fisher_test_agglomerative_dtw,
            base_kwargs=dict(outcome_df=cohort_data, outcome_col="target"),
            dist_matrix=dtw_mat,
            df_ids=imputed_preprocessed[id_vars],
            logger=logger,
            method_label=f"Agglo-DTW-{PCT_LABEL}",
            dist_arg_name="dtw_matrix",
            ids_arg_name="df_ids"
        )

        # Save Fisher's test results CSVs
        for k, fisher_df in all_fisher_agg_dtw.items():
            fisher_csv_path = os.path.join(dtw_dir, f"DTW_Fast_fisher_test_{k}.csv")
            fisher_df.to_csv(fisher_csv_path, index=False)
    else:
        metrics_df_agg_dtw = dtw_mat = top_clust_agg_dtw = all_fisher_agg_dtw = None


# =========================================================
#    Agglomerative Clustering using DTW matrix (tslearn)
# =========================================================

    # Run Agglomerative clustering with precomputed DTW distance matrix using tslearn.cdist_dtw
    if utils.should_run("AGG_DTW_TS", RUN_ALL, RUN_FLAGS):

        # Create per-method/per-threshold output directory
        dtw_ts_dir = os.path.join("metric_results", f"{PCT_LABEL}_threshold", "Agglomerative_clustering_DTW_tslearn_imputed")
        os.makedirs(dtw_ts_dir, exist_ok=True)

        metrics_df_agg_dtw_ts, dtw_ts = run_agglomerative_metrics_dtw_tslearn(
            df_features=imputed_preprocessed,
            id_vars=id_vars,
            n_timesteps=days,
            k_values=K_RANGE,
            df_for_space=imputed_preprocessed,
            logger=logger
        )

        # The metrics are saved as PNG and CSV for documentation and later review
        utils.plot_and_save_kmeans_metrics(
            metrics_df_agg_dtw_ts,
            filename="DTW_tslearn_metrics_graph.png",
            logger=logger,
            folder=dtw_ts_dir
        )
        metrics_df_agg_dtw_ts.to_csv(os.path.join(dtw_ts_dir, "DTW_tslearn_metrics_score.csv"), index=False)

        # Top clusters per metric (Silhouette score, Calinski-Harabasz index and Davies-Bouldin index)
        top_clust_agg_dtw_ts = get_top_clusters_with_threshold(metrics_df_agg_dtw_ts, SIL_THR, logger=logger)

        # Run Fisher's Exact test for each selected cluster count
        all_fisher_agg_dtw_ts = utils.run_fisher_loop_precomputed(
            top_clusters=top_clust_agg_dtw_ts,
            fisher_fn=run_fisher_test_agglomerative_dtw_tslearn,
            base_kwargs=dict(outcome_df=cohort_data, outcome_col="target"),
            dist_matrix=dtw_ts,
            df_ids=imputed_preprocessed[id_vars],
            logger=logger,
            method_label=f"Agglo-DTW-tslearn-{PCT_LABEL}",
            dist_arg_name="dtw_matrix",
            ids_arg_name="df_ids"
        )

        # Save Fisher's test results CSVs
        for k, fisher_df in all_fisher_agg_dtw_ts.items():
            fisher_csv_path = os.path.join(dtw_ts_dir, f"DTW_tslearn_fisher_test_{k}.csv")
            fisher_df.to_csv(fisher_csv_path, index=False)
    else:
        metrics_df_agg_dtw_ts = dtw_ts = top_clust_agg_dtw_ts = all_fisher_agg_dtw_ts = None


# ======================================================================================
#     Agglomerative Clustering using Binary Distance Matrices (Hamming/Jaccard/Dice)
# ======================================================================================

    # Run Agglomerative clustering with precomputed Binary matrices (Hamming/Jaccard/Dice) using imputed vector
    if utils.should_run("BINARY", RUN_ALL, RUN_FLAGS):

        # Create per-method/per-threshold output directory
        binary_dir = os.path.join("metric_results", f"{PCT_LABEL}_threshold", "Agglomerative_binary_clustering")
        os.makedirs(binary_dir, exist_ok=True)

        # Prepare binary feature matrices and precompute distance matrices for Hamming, Jaccard, and Dice
        bin_feats_np, bin_feats_bool_np, dist_mats_bin, binary_dataset = prepare_binary_distance_matrices(
            filtered_common_labs_percent,
            set_percentage,
            id_col="hadm_id",
            item_col="itemid",
            logger=logger
        )

        # Run agglomerative clustering with each binary distance matrix and collect evaluation metrics. Returns a dict with a DataFrame of metrics for each distance type
        metrics_dfs_bin = run_agglomerative_metrics_binary(
            dist_mats=dist_mats_bin,
            feature_space_np=bin_feats_np,
            k_values=K_RANGE,
            logger=logger
        )

        # For each binary distance type (Hamming, Jaccard, Dice), the metrics are saved as PNG and CSV for documentation and later review
        for name, df in metrics_dfs_bin.items():
            utils.plot_and_save_kmeans_metrics(
                df,
                filename=f"metrics_graph_binary_{name.lower()}.png",
                logger=logger,
                folder=binary_dir
            )
            df.to_csv(os.path.join(binary_dir, f"metrics_score_binary_{name.lower()}.csv"), index=False)

        all_fisher_agg_bin = {}

        # For each distance type, determine top cluster counts by metrics and run Fisher's exact test
        for name, df in metrics_dfs_bin.items():
            top_clust_bin = get_top_clusters_with_threshold(df, SIL_THR, logger=logger)
            logger.info("Top clusters for %s: %s", name, top_clust_bin)

            # For each chosen k, run Fisher's test to evaluate association between clusters and outcome
            res_dict = utils.run_fisher_loop_precomputed(
                top_clusters=top_clust_bin,
                fisher_fn=run_fisher_test_agglomerative_binary,
                base_kwargs=dict(outcome_df=cohort_data, outcome_col="target"),
                dist_matrix=dist_mats_bin[name],
                df_ids=binary_dataset[["hadm_id"]],
                logger=logger,
                method_label=f"Binary-{name}-{PCT_LABEL}",
                dist_arg_name="dist_matrix",
                ids_arg_name="df_ids"
            )
            # Collect Fisher result DataFrames per distance/k
            all_fisher_agg_bin.update({f"{name}_{k}": v for k, v in res_dict.items()})

            # Save each Fisher's test result as a CSV file, named by method, distance, and k
            for k, fisher_df in res_dict.items():
                fisher_csv_path = os.path.join(binary_dir, f"fisher_test_binary_{name.lower()}_{k}.csv")
                fisher_df.to_csv(fisher_csv_path, index=False)
    else:
        bin_feats_np = bin_feats_bool_np = dist_mats_bin = binary_dataset = metrics_dfs_bin = all_fisher_agg_bin = None


# ==============================================
#   DBSCAN clustering using 'euclidean' metric
# ==============================================

    # Run DBSCAN clustering using 'euclidean' metric & imupted vector
    if utils.should_run("DBSCAN_EU", RUN_ALL, RUN_FLAGS):

        # Create per-method/per-threshold output directory
        dbscan_dir = os.path.join("metric_results", f"{PCT_LABEL}_threshold", "DBSCAN_clustering_euclidean_imputed")

        # Ensure the output directory exists (create if not)
        os.makedirs(dbscan_dir, exist_ok=True)

        # Run the DBSCAN clustering block tailored for Euclidean metric using the imputed feature matrix. Pass all required parameters including clustering hyperparameters and paths
        dbscan_metrics_eu, dbscan_labels_eu, fisher_eu_dbscan = utils.run_dbscan_block_euclidean(
            df_features=imputed_preprocessed,
            id_vars=id_vars,
            k_for_elbow=5,
            eps=20,
            min_samples=5,
            outcome_df=cohort_data,
            outcome_col="target",
            logger=logger,
            prefix=f"DBSCAN-Euclidean-{PCT_LABEL}",
            output_folder=dbscan_dir
                )

        # If clustering metrics are produced (not None), save metrics CSV file in the output folder
        if dbscan_metrics_eu is not None:
            # Note: metric plots are handled inside utils.run_dbscan_block_euclidean
            dbscan_metrics_eu.to_csv(os.path.join(dbscan_dir, "DBSCAN_metrics_score.csv"), index=False)

        # If Fisher’s test results are produced (not None), save each cluster count’s Fisher result as CSV
        if fisher_eu_dbscan is not None:
            for k, fisher_df in fisher_eu_dbscan.items():
                fisher_csv_path = os.path.join(dbscan_dir, f"DBSCAN_fisher_test_{k}.csv")
                fisher_df.to_csv(fisher_csv_path, index=False)
    else:
        dbscan_metrics_eu = dbscan_labels_eu = fisher_eu_dbscan = None


# =========================
#    Spectral Clustering
# =========================

     # Run HMM-based feature representation and Spectral clustering using imputed vector
    if utils.should_run("SPECTRAL", RUN_ALL, RUN_FLAGS):

        # Create per-method/per-threshold output directory
        spectral_dir = os.path.join("metric_results", f"{PCT_LABEL}_threshold", "Spectral_clustering_imputed")
        os.makedirs(spectral_dir, exist_ok=True)

        hmm_feats, ids_df = build_hmm_state_features(
            df_features=imputed_preprocessed,
            id_vars=id_vars,
            n_timesteps=days,
            n_hmm_states=3,
            logger=logger
        )


        # The metrics are saved as PNG and CSV for documentation and later review
        metrics_df_spec, top_clust_spec = utils.spectral_metrics_block(
            feature_matrix=hmm_feats,
            k_values=K_RANGE,
            affinity="nearest_neighbors",
            plot_filename=f"Spectral_metrics_graph.png",
            sil_threshold=SIL_THR,
            logger=logger,
            folder=os.path.join("metric_results", f"{PCT_LABEL}_threshold", "spectral_clustering_imputed")  # pass folder here
        )

        metrics_csv_path = os.path.join(spectral_dir, f"Spectral_metrics_score.csv")
        metrics_df_spec.to_csv(metrics_csv_path, index=False)

        # Run Fisher's Exact test for each selected cluster count
        all_fisher_spec = utils.run_fisher_loop_spectral(
            top_clusters=top_clust_spec,
            feature_matrix=hmm_feats,
            ids_df=ids_df,
            outcome_df=cohort_data,
            outcome_col="target",
            logger=logger,
            method_label=f"Spectral-{PCT_LABEL}"
        )

        # Save Fisher's test results CSVs
        for k, fisher_df in all_fisher_spec.items():
            fisher_csv_path = os.path.join(spectral_dir, f"Spectral_fisher_test_{k}.csv")
            fisher_df.to_csv(fisher_csv_path, index=False)
    else:
        hmm_feats = ids_df = metrics_df_spec = top_clust_spec = all_fisher_spec = None


# ==================================================
#    K‑MEDOIDS Clustering using 'euclidean' metric
# ==================================================

    # Run K‑MEDOIDS clustering using Euclidean metric
    if utils.should_run("KMEDOIDS_EUCLID", RUN_ALL, RUN_FLAGS):

        # Create per-method/per-threshold output directory
        kmed_dir = os.path.join("metric_results", f"{PCT_LABEL}_threshold", " K_MEDOIDS_clustering_imputed")
        os.makedirs(kmed_dir, exist_ok=True)

        metrics_kmed_eu = run_kmedoids_metrics(
            imputed_preprocessed,
            id_vars,
            k_values=K_RANGE,
            metric="euclidean",
            precomputed=False,
            logger=logger
        )

        # The metrics are saved as PNG and CSV for documentation and later review
        utils.plot_and_save_kmeans_metrics(
            metrics_kmed_eu,
            filename="K_MEDOIDS_metrics_graph.png",
            logger=logger,
            folder=kmed_dir
        )
        metrics_kmed_eu.to_csv(os.path.join(kmed_dir, "K_MEDOIDS_metrics_score.csv"), index=False)

        # Top clusters per metric (Silhouette score, Calinski-Harabasz index and Davies-Bouldin index)
        top_kmed_eu = get_top_clusters_with_threshold(metrics_kmed_eu, SIL_THR, logger=logger)

        # Run Fisher's Exact test for each selected cluster count
        all_fisher_kmed_eu = utils.run_fisher_loop(
            top_clusters=top_kmed_eu,
            fisher_fn=run_fisher_test_kmedoids,
            base_kwargs=dict(
                df_features=imputed_preprocessed,
                id_vars=id_vars,
                outcome_df=cohort_data,
                outcome_col="target",
                metric="euclidean",
                precomputed=False,
                random_state=42
            ),
            logger=logger,
            method_label=f"KMedoids-EU-{PCT_LABEL}"
        )

        # Save Fisher's test results CSVs
        for k, fisher_df in all_fisher_kmed_eu.items():
            fisher_csv_path = os.path.join(kmed_dir, f"K_MEDOIDS_fisher_test_{k}.csv")
            fisher_df.to_csv(fisher_csv_path, index=False)
    else:
        metrics_kmed_eu = top_kmed_eu = all_fisher_kmed_eu = None

'''
Python’s "importlib" built-in library enables us to reload an updated module, such as hyperparameters.py, without restarting the kernel. This allows
us to immediately apply and test changes made in configuration or utility files directly within our main script or notebook, streamlining iterative
development and experimentation.

'''

import importlib
import utils
import hyperparameters
import clustering
importlib.reload(hyperparameters)      # Reload the hyperparameters module to get latest changes
importlib.reload(utils)                # Reload utils.py to reflect any function updates
importlib.reload(clustering)           # Reload clustering functions to incorporate any changes in clustering.py code
from clustering import *               # Import all updated clustering functions into the current namespace
from hyperparameters import *          # Import all updated hyperparameter settings and flags

'''
  Print the current lab inclusion threshold percentage (set_percentage) after reload. This confirms what coverage threshold for lab features
  the pipeline will use in this run.
'''
print(f"Threshold: {set_percentage*100} %")  # Getting status of current threshold value after reloading utils.py module

'''
  Print the dictionary RUN_FLAGS controlling which clustering methods will be executed. Each key corresponds to a clustering algorithm,
  with True/False indicating whether it will run. This helps verify that the pipeline will only run the selected clustering blocks as per
  the current configuration
'''
print(f"Flags set: {RUN_FLAGS}")       # Print which clustering blocks will run ('True') after reloading utils.py module


#   =========================================================================== MAIN DRIVER ================================================================================

'''
Description of the data files:
- cohort_data: DataFrame containing patient admission metadata and clinical outcomes, loaded from cached pickle files or raw data.

- labevents_data: DataFrame with individual laboratory event records and timestamps, linked to patient admissions,loaded from cached pickle files or raw data.

- filtered_common_labs_percent: DataFrame of lab events filtered by coverage threshold, retaining frequently measured labs; used as input for Agglomerative Clustering 
                                with Binary Distance Matrices.

- pivot_percent (Unimputed vector): Wide-format unimputed vector (admissions × lab-day features) representing raw measurements before Normalization/Linear Interpolation/
                                    KNN Imputation; used for clustering methods based on pre-computed distance matrices such as Agglomerative Clustering with Manhattan,
                                    Cosine, Mahalanobis and Euclidean metrics.

- feature_matrix_percent: 2D NumPy array derived from pivot_percent vector, serving as input to compute distance matrices for selected algorithms.

- imputed_preprocessed (imputed vector): Fully preprocessed vector with normalized,interpolated and imputed values, serving as primary input for clustering algorithms
                                         like KMeans, KMedoids (Euclidean),Agglomerative Clustering using DTW matrix, DBSCAN, and Spectral Clustering.

- id_vars: List of columns (e.g., ['hadm_id', 'subject_id']) serving as unique admission identifiers throughout the pipeline.

- PCT_LABEL: String identifier reflecting the lab coverage percentage threshold, used consistently for caching, file naming, and reproducibility.

'''

def main():

    '''
    Generate a label reflecting the feature inclusion threshold (e.g., "75pct") for consistent caching and output naming.
    '''
    PCT_LABEL = f"{int(set_percentage*100)}pct"

    '''
    Execute the data loading step, retrieving or caching patient cohort and lab events datasets.
    '''
    cohort_data, labevents_data = load_data()

    '''
    Perform data preprocessing, including filtering, pivoting, and Normalization/Imputation/Interpolation.
    The results include filtered lab events, unimputed and imputed feature matrices, and identifier columns.
    '''
    filtered_common_labs_percent, pivot_percent, feature_matrix_percent, imputed_preprocessed, id_vars = preprocess(
        cohort_data, labevents_data, set_percentage, days, PCT_LABEL)

    '''
    Run the clustering algorithms and evaluation pipeline, passing all preprocessed data and configuration.
    This orchestrates multiple clustering methods, metric calculations, and statistical significance testing.
    Note: The function accepts several data versions to accommodate method-specific requirements.
    '''
    run_clustering(filtered_common_labs_percent, pivot_percent, feature_matrix_percent, imputed_preprocessed, id_vars, cohort_data, PCT_LABEL)

if __name__ == "__main__":
    '''
    Entry point to start the entire analysis pipeline end-to-end
    '''
    main()



