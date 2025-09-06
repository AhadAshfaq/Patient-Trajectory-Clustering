import pandas as pd
from pathlib import Path
import seaborn as sns
import matplotlib.pyplot as plt
import re
import os

def extract_k(filename: str):
    match = re.search(r'k=(\d+)', filename)
    return int(match.group(1)) if match else None

def extract_metric(filename: str):
    fname_lower = filename.lower()
    metrics_found = [m for m in ['silhouette', 'calinski_harabasz', 'davies_bouldin'] if m in fname_lower]
    return metrics_found[0] if metrics_found else None

def get_data_type(name: str) -> str:
    if "unimputed_normalized" in name:
        return "unimputed_normalized"
    elif "unimputed" in name:
        return "unimputed"
    elif "binary" in name:
        return "binary"
    elif "imputed" in name:
        return "imputed"
    else:
        return "unknown"

DISTANCES = [
    "manhattan",
    "euclidean",
    "cosine",
    "dtw_tslearn",
    "dtw_fast",
    "mahalanobis",
    "ward",
    "jaccard",
    "hamming",
    "dice"
]

def extract_distance(cluster_name: str, file_name: str) -> str:
    cname = cluster_name.lower()
    fname = file_name.lower()

    for d in DISTANCES:
        if d in cname or d in fname:
            return d
    return "euclidean" # euclidean if not specified

ALGORITHMS = [
    "agglomerative",
    "k_medoids",
    "kmeans",
    "spectral",
    "dbscan"
]

def extract_algorithm(cluster_name: str) -> str:
    cname = cluster_name.lower().strip()

    for a in ALGORITHMS:
        if a in cname:
            return a
    return "unknown"

root = Path("metric_results")
all_dfs = []

# concatenate all metrics files
for pc_folder in root.glob("*pct_threshold"):
    pc_name = pc_folder.name  # e.g. "50pc_threshold"

    for cluster_folder in pc_folder.iterdir():
        if cluster_folder.is_dir():
            cluster_name = cluster_folder.name  # e.g. "kmeans_raw"

            # Look for CSV files containing "metrics"
            for f in cluster_folder.glob("*metrics*.csv"):
                df = pd.read_csv(f)  # adjust sep= if needed
                df["pct_threshold"] = pc_name
                df["cluster_type"] = cluster_name
                df["file_name"] = f.name
                df.rename(columns={"Unnamed: 0":"k"}, inplace=True)
                all_dfs.append(df)

all_df = pd.concat(all_dfs, ignore_index=True)

all_df

# extract relevant information
final_df = all_df.copy()
final_df["pct_threshold"] = final_df["pct_threshold"].str.replace("pct_threshold", "%")
final_df["data_type"] = final_df["cluster_type"].apply(lambda x: get_data_type(x))
final_df["algorithm"] = final_df["cluster_type"].apply(lambda x: extract_algorithm(x))
final_df["distance"] = final_df.apply(lambda x: extract_distance(x.cluster_type, x.file_name), axis=1)
# reorder columns
final_df = final_df[['k', 'silhouette', 'calinski_harabasz', 'davies_bouldin', 'data_type', 'algorithm', 'distance', 'pct_threshold']]

final_df

final_df = final_df.melt(
    id_vars=['k', 'data_type', 'algorithm', 'distance', 'pct_threshold'],
    value_vars=['silhouette', 'calinski_harabasz', 'davies_bouldin'],
    var_name='metric',
    value_name='score'
)
# Define folder and file names
folder_name = "cluster analysis"
metrics_file_name = "all_metrics_dataframe.csv"
file_path = os.path.join(folder_name, metrics_file_name)

# Create the folder if it does not exist
os.makedirs(folder_name, exist_ok=True)

# Save the dataframe as CSV (overwrite if exists)
final_df.to_csv(file_path, index=False)

print(f"Saved DataFrame to: {file_path}")

final_df



"""#### CLUSTERING RESULTS 1

Clustering is typically applied on imputed data with clustering methods that employ euclidean distance.
First, we should show these results.

We obtain best Silhouette score in agglomerative clustering, which motivates our further exploration of this clustering algorithm across data types and distances.

Interestingly, results do not vary a lot across percentage thresholds.

**issues** how to report results for DBSCAN?
"""

sub1 = final_df[(final_df["data_type"]=="imputed")&(final_df["distance"]=="euclidean")].dropna()

g = sns.FacetGrid(
    sub1,
    row="pct_threshold",
    col="metric",
    hue="algorithm",
    margin_titles=True,
    sharey=False,
    height=2.5,
    aspect=1
)

g.map(sns.lineplot, "k", "score", marker="o")

# add legend & adjust
g.add_legend(
    title="Algorithm",
    bbox_to_anchor=(0.5, -0.05),
    loc="upper center",
    ncol=len(sub1["algorithm"].unique())
)
plt.tight_layout()
plt.show()

"""#### CLUSTERING RESULTS 2

Here we show how, for a standard data type (imputed series), the performance of the agglomerative algorithm changes with respect to different distances. It is shown that DTW (also in its optimised version) computes a better distance for clustering.

Again, results do not vary a lot across percentage thresholds, suggesting that clustering can be done effectively on a reduced set of most frequent features.

**issues** here I saved agglomerative ward as a separate distance (to separate it from the simple Euclidean) but ward is actually a linkage strategy. what is the difference between [agglomerative ward] and [agglomerative euclidean]? what distance and linkage method is used in these two cases?
"""

sub2 = final_df[(final_df["data_type"]=="imputed")&(final_df["algorithm"]=="agglomerative")].dropna()

g = sns.FacetGrid(
    sub2,
    row="pct_threshold",
    col="metric",
    hue="distance",
    margin_titles=True,
    sharey=False,
    height=2.5,
    aspect=1
)

g.map(sns.lineplot, "k", "score", marker="o")

# add legend & adjust
g.add_legend(
    title="Distance",
    bbox_to_anchor=(0.5, -0.05),
    loc="upper center",
    ncol=len(sub1["algorithm"].unique())
)
plt.tight_layout()
plt.show()

"""#### CLUSTERING RESULTS 3

Here we show the effect of preprocessing on clustering. Clustering on imputed data typically separates samples better than unimputed, however clustering on binary yields better results, indicating that whether labs are measured or not in a given admissions separates them in clusters.
"""

sub3 = final_df[ (final_df["algorithm"]=="agglomerative")
                 & (final_df["pct_threshold"]=="50%")
                 & (final_df["data_type"].isin(["imputed","unimputed_normalized", "binary"]))
                 & (final_df["distance"].isin(["euclidean","cosine","mahalanobis","manhattan","jaccard","hamming","dice"]))
                ].dropna()

g = sns.relplot(
    data=sub3,
    x="k",
    y="score",
    hue="data_type",
    style="distance",
    kind="line",
    col="metric",
    row="pct_threshold",
    marker="o",
    height=4,
    aspect=0.5,
    facet_kws={"margin_titles": True, "sharey": False}
)

g.set(ylim=(0, None))

# move legend to bottom
g._legend.set_bbox_to_anchor((1.3, 0.5))

plt.tight_layout()
plt.show()

# here it is easier to appreciate the differences between imputed and unimputed

sub3 = final_df[(final_df["pct_threshold"]=="50%") & (final_df["algorithm"]=="agglomerative")
                 & (final_df["data_type"].isin(["imputed","unimputed_normalized"]))
                 & (final_df["distance"].isin(["euclidean","cosine","mahalanobis","manhattan"]))
                ].dropna()

g = sns.FacetGrid(
    sub3,
    row="distance",
    col="metric",
    hue="data_type",
    margin_titles=True,
    sharey=False,
    height=2,
    aspect=1.2
)

g.map(sns.lineplot, "k", "score", marker="o")

g.set(ylim=(0, None))

# add legend & adjust
g.add_legend(
    title="Algorithm",
    bbox_to_anchor=(0.5, -0.05),
    loc="upper center",
    ncol=len(sub1["algorithm"].unique())
)
plt.tight_layout()
plt.show()

all_fts = []

# concatenate all fisher test files
for pc_folder in root.glob("*pct_threshold"):
    pc_name = pc_folder.name  # e.g. "50pc_threshold"

    for cluster_folder in pc_folder.iterdir():
        if cluster_folder.is_dir():
            cluster_name = cluster_folder.name  # e.g. "kmeans_raw"

            # Look for CSV files containing "metrics"
            for f in cluster_folder.glob("*fisher_test*.csv"):
                df = pd.read_csv(f)  # adjust sep= if needed
                df["pct_threshold"] = pc_name
                df["cluster_type"] = cluster_name
                df["file_name"] = f.name
                df["k"] = extract_k(f.name)
                df["metric_used"] = extract_metric(f.name)
                df.rename(columns={"Unnamed: 0":"Cluster"}, inplace=True)
                all_fts.append(df)

all_ft = pd.concat(all_fts, ignore_index=True)

all_ft

# extract relevant information
final_ft = all_ft.copy()
final_ft["pct_threshold"] = final_ft["pct_threshold"].str.replace("pct_threshold", "%")
final_ft["data_type"] = final_ft["cluster_type"].apply(lambda x: get_data_type(x))
final_ft["algorithm"] = final_ft["cluster_type"].apply(lambda x: extract_algorithm(x))
final_ft["distance"] = final_ft.apply(lambda x: extract_distance(x.cluster_type, x.file_name), axis=1)

final_ft

final_ft = final_ft[['Cluster', 'Corrected_p_value', 'Significant_after_correction',
                     'pct_threshold', 'k', 'metric_used', 'data_type', 'algorithm', 'distance']]
final_ft = final_ft.merge(final_df, on=['k', 'data_type', 'algorithm', 'distance', 'pct_threshold']) #'metric'

# Define folder and file names
folder_name = "cluster analysis"
file_name = "final_dataframe.csv"
file_path = os.path.join(folder_name, file_name)

# Create the folder if it does not exist
os.makedirs(folder_name, exist_ok=True)

# Save the dataframe as CSV (overwrite if exists)
final_ft.to_csv(file_path, index=False)

print(f"Saved DataFrame to: {file_path}")

final_ft.head(20)

final_ft[(final_ft["Significant_after_correction"])
        & (final_ft["data_type"]!="binary")
        & (final_ft["metric"]=="silhouette")
        ].sort_values("score", ascending = False).head(20)

final_ft[(final_ft["Significant_after_correction"])
        & (final_ft["data_type"]=="binary")
        & (final_ft["metric"]=="silhouette")
        ].sort_values("score", ascending = False).head(20)

"""#### CLUSTER ANALYSIS

Although many distances and preprocessing strategies were tested on agglomerative clustering, typically the best resulting clusters are not sinificantly enriched for the clinical outcome under consideration. The only clustering algorithm that achieves good clustering performance and identifies significant clusters is spectral clustering. These clusters should be further investigated.

[This is also due to the fact that in agglomerative clustering the best number of k is small (2 to 5) while in spectral clustering is large (10 to 15) so we generate smaller clusters]

Later, we can also For binary data, agglomerative clustering on dice, hamming or jaccard yielded significant clusters. We should choose one (possibly dice because it yields the best sihouette score) and analyse the significant clusters.

#### HOW TO ANALYSE CLUSTERS

We want to know what lab measurements are most different across different clusters
1. Start from the raw data (not discretised, imputed or normalised). For each admission, compute some aggregating function (mean, std, min_value, max_value).
2. For each admission, consider the cluster assignment (cluster number from 1 to N), and the result of the fisher test for each cluster (T/F)
3. Now, for each variable (lab test) plot the distribution (as boxplot) of these statistics across clusters and cluster significance (by grouping the clusters with the same significance value). You can do this for the most frequent N variables (10 or 20, depending on what results you obtain in the top features)
4. Apply Kruskal-Wallis test to see if differences are significant across clusters - use the kruskal function from scipy.stats
5. Apply Mann Whitney test to see if differences are significant between the two groups of clusters (those that are significant according to the Fisher's test, and those that are not) - use the mannwhitneyu function from scipy.stats

#### IDEAS for LIMITATIONS and FUTURE WORK
clustering results tend to be similar across different threshold of features selection, and also between imputed and unimputed > it would be interesting to see if similar clusters are generated.

here admissions of the same patients are considered independently althought admissions > methods should be modified to account for admissions from same patient

clustering results seems to be better over binary matrices > it would be interesting to cluster over combined imputed and binary distances that account for both lab measured and respective values.
"""



"""# Cluster Interpretation: Investigating Lab Measurement Differences Across Clusters
This section performs a detailed interpretation of clustering results, focusing on how raw laboratory measurements differ across patient clusters identified by spectral clustering at specified thresholds.

The core objectives and steps are:

Aggregate raw lab (filtered_common_labs from preprocessing_data directory) data (mean, standard deviation, min, max) for each patient admission and lab test, prior to discretization or imputation.

Merge these aggregated lab measurements with cluster assignments and cluster-level clinical significance information derived from Fisher's exact tests (from Spectral_clustering_imputed directory). This links patient lab profiles to cluster membership and outcome relevance.

Visualize distributions of lab values across clusters and mark clusters identified as clinically significant, using boxplots for the top-N (20) most frequent lab tests.

Perform statistical hypothesis tests to identify:

Labs with significant differences across clusters (using Kruskal-Wallis tests).

Labs with significant differences between clusters significant for clinical outcome versus those that are not (using Mann-Whitney U tests).

Output figures and summary statistics to facilitate understanding of biologically meaningful clusters and key differentiating lab variables.
"""

from imports import *  # Standard imports for numpy, pandas, seaborn, etc.
logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s:%(name)s:%(message)s')
logger = logging.getLogger("ClusterAnalysis")

def load_cluster_assignments(csv_path):
    """
    Load per-admission cluster assignments from a CSV.
    Args:
        csv_path (str): Path to the cluster assignments CSV. Must contain 'hadm_id' and 'cluster_label'.
    Returns:
        pd.DataFrame: DataFrame with 'hadm_id' and 'cluster_label'.
    """
    df = pd.read_csv(csv_path)
    if 'hadm_id' not in df.columns:
        raise ValueError("Cluster assignments CSV must have 'hadm_id' column")
    return df

def load_fisher_results(csv_path):
    """
    Load Fisher test results, including cluster significance, from a CSV.
    Args:
        csv_path (str): Path to Fisher results CSV. Must contain 'Cluster' and 'Significant_after_correction'.
    Returns:
        pd.DataFrame: DataFrame with per-cluster significance.
    """
    df = pd.read_csv(csv_path)
    if 'Significant_after_correction' not in df.columns:
        raise ValueError("Fisher results CSV must have 'Significant_after_correction' column")
    return df

def aggregate_lab_stats(lab_df):
    """
    Aggregate raw lab values per admission and lab test into summary statistics.
    Args:
        lab_df (pd.DataFrame): DataFrame of raw labs with columns ['hadm_id', 'itemid', 'valuenum'].
    Returns:
        pd.DataFrame: Aggregated DataFrame with mean, std, min, max for each ('hadm_id', 'itemid').
    """
    logger.info("Aggregating lab measurements per admission and lab...")
    agg = lab_df.groupby(['hadm_id', 'itemid'])['valuenum'].agg(['mean', 'std', 'min', 'max']).reset_index()
    logger.info(f"Aggregated lab stats: {agg.shape[0]} rows\n{agg.head(5)}\n")
    return agg

def merge_cluster_lab_data(lab_agg_df, cluster_assign_df, fisher_results_df):
    """
    Merge lab stats with cluster assignments and cluster significance.
    Args:
        lab_agg_df (pd.DataFrame): Output of aggregate_lab_stats.
        cluster_assign_df (pd.DataFrame): Cluster assignments per admission.
        fisher_results_df (pd.DataFrame): Per-cluster Fisher significance.
    Returns:
        pd.DataFrame: Merged DataFrame for plotting and statistical testing.
    """
    logger.info("Merging lab aggregates with cluster assignments and Fisher significance...")
    merged = lab_agg_df.merge(cluster_assign_df, on='hadm_id')
    merged = merged.merge(
        fisher_results_df[['Cluster', 'Significant_after_correction']],
        left_on='cluster_label', right_on='Cluster', how='left'
    )
    merged['Significant_after_correction'] = merged['Significant_after_correction'].fillna(False)
    logger.info(f"Merged DataFrame: {merged.shape[0]} rows\n{merged.head(5)}\n")
    return merged

def plot_lab_distributions(df, output_dir, top_n):
    """
    For each of the N most frequent labs, plot a boxplot of mean lab value by cluster and cluster significance.
    Args:
        df (pd.DataFrame): Output from merge_cluster_lab_data.
        output_dir (str): Directory to save boxplots.
        top_n (int): Number of labs to plot.
    """
    logger.info(f"Plotting boxplots for top {top_n} labs...")
    os.makedirs(output_dir, exist_ok=True)
    top_labs = df['itemid'].value_counts().nlargest(top_n).index.tolist()
    logger.info(f"Top labs selected: {top_labs}\n")

    for lab in top_labs:
        plt.figure(figsize=(10,6))
        sub = df[df['itemid'] == lab]
        sns.boxplot(
            data=sub,
            x='cluster_label',
            y='mean',
            hue='Significant_after_correction',
            palette="Set2"
        )
        plt.title(f"Lab {lab} - Mean by Cluster and Significance")
        plt.xlabel("Cluster")
        plt.ylabel("Mean Value")
        plt.legend(title="Cluster Significant")
        plt.tight_layout()
        filename = os.path.join(output_dir, f"lab_{lab}_mean_boxplot.png")
        plt.savefig(filename)
        plt.close()
        logger.info(f"Saved boxplot: {filename}")

def run_stat_tests(df, output_dir):
    """
    Perform Kruskal-Wallis test across clusters and Mann-Whitney U between significant/non-significant clusters for each lab.
    Saves results as a summary CSV.
    Args:
        df (pd.DataFrame): Output from merge_cluster_lab_data.
        output_dir (str): Directory to save statistical result CSV.
    Returns:
        pd.DataFrame: DataFrame with test results for each lab.
    """
    logger.info("Running statistical tests for each lab...")
    os.makedirs(output_dir, exist_ok=True)
    results = []
    lab_ids = df['itemid'].unique()

    for lab in lab_ids:
        sub = df[df['itemid'] == lab]
        groups = [group['mean'].dropna() for _, group in sub.groupby('cluster_label')]
        if len(groups) <= 1:
            continue
        kw_stat, kw_p = kruskal(*groups)
        sig_vals = sub[sub['Significant_after_correction']]['mean'].dropna()
        non_sig_vals = sub[~sub['Significant_after_correction']]['mean'].dropna()
        mw_p = None
        if len(sig_vals) > 0 and len(non_sig_vals) > 0:
            _, mw_p = mannwhitneyu(sig_vals, non_sig_vals)
        results.append({
            'lab_id': lab,
            'kruskal_p': kw_p,
            'mannwhitney_p': mw_p,
            'mean_sig': sig_vals.mean() if not sig_vals.empty else None,
            'mean_non_sig': non_sig_vals.mean() if not non_sig_vals.empty else None
        })

    results_df = pd.DataFrame(results)
    results_path = os.path.join(output_dir, 'statistical_test_results.csv')
    results_df.to_csv(results_path, index=False)
    logger.info(f"Saved statistical test results to {results_path}\n")
    return results_df

def run_full_analysis(lab_pkl_path, cluster_csv_path, fisher_csv_path, output_base_dir, PCT_LABEL, top_n_labs):
    """
    Complete workflow to produce cluster interpretation:
    - Load all required files,
    - Aggregate labs,
    - Join cluster/fisher labels,
    - Plot distributions,
    - Perform statistical testing.
    Args:
        lab_pkl_path (str): Path to raw filtered labs pickle.
        cluster_csv_path (str): Path to cluster assignments CSV.
        fisher_csv_path (str): Path to Fisher results CSV.
        output_base_dir (str): Directory for plots/statistics.
        PCT_LABEL (str): Threshold labeling string (for logging).
        top_n_labs (int): Number of labs to analyze in depth.
    Returns:
        (merged, stats_df): Tuple of resulting DataFrames for further review.
    """
    logger.info(f"Running full cluster analysis for threshold {PCT_LABEL} and result output in {output_base_dir}")
    clusters = load_cluster_assignments(cluster_csv_path)
    fisher = load_fisher_results(fisher_csv_path)
    lab_df = pd.read_pickle(lab_pkl_path)
    lab_agg = aggregate_lab_stats(lab_df)
    merged = merge_cluster_lab_data(lab_agg, clusters, fisher)
    plot_dir = os.path.join(output_base_dir, f'plots')
    plot_lab_distributions(merged, plot_dir, top_n=top_n_labs)
    stats_dir = os.path.join(output_base_dir, f'statistics')
    stat_results = run_stat_tests(merged, stats_dir)
    return merged, stat_results

if __name__ == "__main__":
    # Example configuration for spectral cluster analysis with 1% threshold and k=15
    threshold = 1                  # Select one of the [1,50,75]
    PCT_LABEL = f"{threshold}pct"
    metric='calinski_harabasz'     # Select one of the [silhouette,calinski_harabasz,davies_bouldin]
    cluster_number = 15            # Select cluster='X', considering the fact that "Spectral_cluster_assignments_k=X" and "Spectral_fisher_test_{metric}_k=X" exist in Spectral_clustering_imputed directory

    lab_path = f"preprocessing_data/filtered_common_labs_{PCT_LABEL}.pkl"
    cluster_assignments_path = (
        f"metric_results/{PCT_LABEL}_threshold/Spectral_clustering_imputed/Spectral_cluster_assignments_k={cluster_number}.csv"
    )
    fisher_results_path = (
        f"metric_results/{PCT_LABEL}_threshold/Spectral_clustering_imputed/Spectral_fisher_test_{metric}_k={cluster_number}.csv"
    )
    clustering_dir = f"cluster_interpretation/{PCT_LABEL}_threshold/for_clusters_k={cluster_number}"

    merged_df, stats_df = run_full_analysis(
        lab_path,
        cluster_assignments_path,
        fisher_results_path,
        clustering_dir,
        PCT_LABEL,
        top_n_labs=20
    )
    logger.info(f"Merged data preview:\n{merged_df.head(10)}\n")
    logger.info(f"Statistical test results preview:\n{stats_df.head(10)}\n")



