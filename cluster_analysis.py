from imports import *

# ---------- Utility extraction functions -----------
def extract_k(filename: str):
    """Extracts the cluster count (k) from a filename using regex."""
    match = re.search(r'k=(\d+)', filename)
    return int(match.group(1)) if match else None

def extract_metric(filename: str):
    """Returns the metric type from the filename ('silhouette', 'calinski_harabasz', or 'davies_bouldin')."""
    fname_lower = filename.lower()
    metrics_found = [m for m in ['silhouette', 'calinski_harabasz', 'davies_bouldin'] if m in fname_lower]
    return metrics_found[0] if metrics_found else None

def get_data_type(name: str) -> str:
    """Determines the input data type label from a string."""
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

# Supported distance and algorithm types for clustering evaluation
DISTANCES = ["manhattan", "euclidean", "cosine", "dtw_tslearn", "dtw_fast", "mahalanobis", "ward", "jaccard", "hamming", "dice"]

def extract_distance(cluster_name: str, file_name: str) -> str:
    """Infers the distance metric used from either the cluster folder or filename."""
    cname = cluster_name.lower()
    fname = file_name.lower()
    for d in DISTANCES:
        if d in cname or d in fname:
            return d
    return "euclidean"

ALGORITHMS = ["agglomerative", "k_medoids", "kmeans", "spectral", "dbscan"]

def extract_algorithm(cluster_name: str) -> str:
    """Infers the clustering algorithm from the cluster folder name."""
    cname = cluster_name.lower().strip()
    for a in ALGORITHMS:
        if a in cname:
            return a
    return "unknown"

# ----------- Aggregate all clustering results metrics -----------
root = Path("metric_results")
all_dfs = []
# Load metrics CSVs and tag with experiment parameters
for pc_folder in root.glob("*pct_threshold"):
    pc_name = pc_folder.name
    for cluster_folder in pc_folder.iterdir():
        if cluster_folder.is_dir():
            cluster_name = cluster_folder.name
            for f in cluster_folder.glob("*metrics*.csv"):
                df = pd.read_csv(f)
                df["pct_threshold"] = pc_name
                df["cluster_type"] = cluster_name
                df["file_name"] = f.name
                df.rename(columns={"Unnamed: 0":"k"}, inplace=True)
                all_dfs.append(df)
all_df = pd.concat(all_dfs, ignore_index=True)

# Extract labels for type, distance, algorithm, and reshape for tidy analysis
final_df = all_df.copy()
final_df["pct_threshold"] = final_df["pct_threshold"].str.replace("pct_threshold", "%")
final_df["data_type"] = final_df["cluster_type"].apply(lambda x: get_data_type(x))
final_df["algorithm"] = final_df["cluster_type"].apply(lambda x: extract_algorithm(x))
final_df["distance"] = final_df.apply(lambda x: extract_distance(x.cluster_type, x.file_name), axis=1)
# Keep only relevant fields in a long-format DataFrame
final_df = final_df[['k', 'silhouette', 'calinski_harabasz', 'davies_bouldin', 'data_type', 'algorithm', 'distance', 'pct_threshold']]
final_df = final_df.melt(
    id_vars=['k', 'data_type', 'algorithm', 'distance', 'pct_threshold'],
    value_vars=['silhouette', 'calinski_harabasz', 'davies_bouldin'],
    var_name='metric',
    value_name='score'
)
# Save metrics table for downstream use
folder_name = "cluster analysis"
metrics_file_name = "all_metrics_dataframe.csv"
file_path = os.path.join(folder_name, metrics_file_name)
os.makedirs(folder_name, exist_ok=True)
final_df.to_csv(file_path, index=False)
print(f"Saved DataFrame to: {file_path}")


"""
CLUSTERING RESULTS 1: Comparison of Clustering Algorithms on Imputed Data (Euclidean Distance):
'''
Initial benchmarking included k-means, agglomerative, spectral clustering, k-medoids, and DBSCAN.
K-medoids and DBSCAN yielded unstable, near-zero silhouette scores at all thresholds/k and are not discussed further.
The plot below summarizes clustering validity for agglomerative, k-means, and spectral clustering (on imputed/euclidean data), with line style showing feature thresholds.
Spectral and agglomerative methods consistently show stronger clustering (higher silhouette and Calinski-Harabasz, lower Davies-Bouldin) than k-means, and their performance is robust to threshold selection.
'''
"""

# --- Plot Clustering Results 1: main algorithm comparison by threshold ----
algos_to_plot = ["agglomerative", "kmeans", "spectral"]
sub1 = final_df[
    (final_df["data_type"] == "imputed") &
    (final_df["distance"] == "euclidean") &
    (final_df["algorithm"].isin(algos_to_plot))
].dropna()

style_dict = {"1%": "-", "50%": "--", "75%": ":"}
palette = sns.color_palette("Set1", n_colors=len(algos_to_plot))

fig, axes = plt.subplots(1, 3, figsize=(18, 6))
metrics = ['silhouette', 'calinski_harabasz', 'davies_bouldin']
for i, metric in enumerate(metrics):
    ax = axes[i]
    for j, algo in enumerate(algos_to_plot):
        for threshold, linestyle in style_dict.items():
            data = sub1[(sub1['algorithm'] == algo) & (sub1['pct_threshold'] == threshold) & (sub1['metric'] == metric)]
            if not data.empty:
                ax.plot(
                    data['k'], data['score'],
                    label=f"{algo}, {threshold}",
                    linestyle=linestyle,
                    color=palette[j],
                    marker="o",
                    markersize=6
                )
    ax.set_title(f"metric = {metric}")
    ax.set_xlabel("k")
    if i == 0:
        ax.set_ylabel("score")
fig.suptitle("Comparison of Clustering Algorithms on Imputed Data (Euclidean Distance)", y=1.1, fontsize=18)

algorithm_handles = [
    mlines.Line2D([], [], color=palette[i], marker="o", linestyle='-', label=algo)
    for i, algo in enumerate(algos_to_plot)
]
threshold_handles = [
    mlines.Line2D([], [], color="black", linestyle=style_dict[t], label=t)
    for t in style_dict
]
fig.legend(
    handles=algorithm_handles, title="Algorithm",
    loc='lower center', bbox_to_anchor=(0.5, -0.06),
    ncol=3, fontsize='large', title_fontsize='x-large'
)
fig.legend(
    handles=threshold_handles, title="Threshold",
    loc='upper center', bbox_to_anchor=(0.5, 1.03),
    ncol=3, fontsize='large', title_fontsize='x-large'
)
plt.tight_layout(rect=[0, 0.05, 1, 0.98])
output_dir = "Result_figures"
os.makedirs(output_dir, exist_ok=True)
figpath = os.path.join(output_dir, "clustering_1.png")
fig.savefig(figpath, bbox_inches='tight', dpi=300)
plt.show()
print(f"Plot saved to: {figpath}")



"""
CLUSTERING RESULTS 2: Agglomerative Clustering on Imputed Data using Multiple Distance Metrics (Average Vs Ward Linkage):
'''
Cluster validation was benchmarked for agglomerative clustering with all supported distance metrics and linkages.
K-medoids was excluded due to poor scores. Agglomerative clustering with average linkage (not Ward) and DTW/Euclidean distances gave the most robust results across all thresholds.
Ward linkage was tested but performed worse for silhouette/Davies-Bouldin. Summary plots below show that optimal distances and k are consistent across thresholds.
'''
"""

# --- Plot Clustering Results 2: Distance metric benchmarking for Agglomerative clustering ---
sub2 = final_df[
    (final_df["data_type"] == "imputed") &
    (final_df["algorithm"] == "agglomerative")
].dropna()
distances_to_plot = sub2["distance"].unique().tolist()
palette = sns.color_palette("Set2", n_colors=len(distances_to_plot))
g = sns.FacetGrid(
    sub2,
    row="pct_threshold",
    col="metric",
    hue="distance",
    margin_titles=True,
    sharey=False,
    height=3,
    aspect=0.4
)
g.map(sns.lineplot, "k", "score", marker="o")
g.add_legend(
    title="Distance",
    bbox_to_anchor=(0.5, 0.1),
    loc="upper center",
    ncol=len(distances_to_plot),
    frameon=True
)
g._legend.get_frame().set_edgecolor('black')
g._legend.get_frame().set_facecolor('white')
g.fig.suptitle(
    "Agglomerative Clustering on Imputed Data using Multiple Distance Metrics (Average Vs Ward Linkage)", y=0.98, fontsize=14
)
plt.tight_layout(rect=[0, 0.09, 1, 0.98])
output_dir = "Result_figures"
os.makedirs(output_dir, exist_ok=True)
figpath = os.path.join(output_dir, "clustering_2.png")
g.savefig(figpath, bbox_inches='tight', dpi=300)
plt.show()
print(f"Plot saved to: {figpath}")


"""
CLUSTERING RESULTS 3: Imputed versus Unimputed_normalized data & Binary vs Numeric representations Comparison:
'''
Comparing clustering outcomes for imputed versus unimputed_normalized data using all main numeric distances, and binary vs numeric representations (on best-performing distances).
Imputed data consistently improved all metrics regardless of distance, and binary approaches achieved the highest silhouette scores and strong overall clustering—motivating future work combining both.
'''
"""

# --- Plot Clustering Results 3: Imputed vs Unimputed, plus Binary vs Numeric comparisons ---
sub_num = final_df[
    (final_df["algorithm"]=="agglomerative") &
    (final_df["pct_threshold"]=="50%") &
    (final_df["data_type"].isin(["imputed", "unimputed_normalized"])) &
    (final_df["distance"].isin(["euclidean", "cosine", "mahalanobis", "manhattan",]))
].dropna()
linestyle_map = {"imputed": "-", "unimputed_normalized": "--"}
color_map = {
    "euclidean": "#1f77b4",   # blue
    "cosine": "#ff7f0e",      # orange
    "mahalanobis": "#2ca02c", # green
    "manhattan": "#d62728"    # red
}
metrics = ["silhouette", "calinski_harabasz", "davies_bouldin"]
fig, axes = plt.subplots(1, 3, figsize=(18, 6))
for i, metric in enumerate(metrics):
    ax = axes[i]
    for distance in ["euclidean", "cosine", "mahalanobis", "manhattan"]:
        for dtype in ["imputed", "unimputed_normalized"]:
            sub = sub_num[(sub_num["metric"] == metric) &
                          (sub_num["distance"] == distance) &
                          (sub_num["data_type"] == dtype)]
            if not sub.empty:
                ax.plot(
                    sub["k"], sub["score"],
                    label=f"{distance} ({dtype})",
                    color=color_map[distance],
                    linestyle=linestyle_map[dtype],
                    marker="o",
                    markersize=6
                )
    ax.set_title(f"metric = {metric}")
    ax.set_xlabel("k")
    if i == 0:
        ax.set_ylabel("score")
distance_handles = [
    mlines.Line2D([], [], color=color_map[dist], linestyle='-', marker="o", label=dist.capitalize())
    for dist in color_map
]
type_handles = [
    mlines.Line2D([], [], color="black", linestyle=linestyle_map[dt], label=dt.replace("_", " ").capitalize())
    for dt in linestyle_map
]
fig.legend(handles=distance_handles, title="Distance", loc='upper center', bbox_to_anchor=(0.5, 1), ncol=4)
fig.legend(handles=type_handles, title="Data Type", loc='lower center', bbox_to_anchor=(0.5, -0.02), ncol=2)
plt.suptitle("Imputed vs Unimputed_Normalized - All Numeric Distances", y=1.05, fontsize=18)
plt.tight_layout(rect=[0, 0.05, 1, 0.98])
output_dir = "Result_figures"
os.makedirs(output_dir, exist_ok=True)
figpath = os.path.join(output_dir, "clustering_3_datatype.png")
plt.savefig(figpath, bbox_inches='tight', dpi=300)
plt.show()
print(f"Plot saved to: {figpath}")

# ---- Binary vs Numeric Comparisons ----
numeric_types = ["imputed", "unimputed_normalized"]
binary_types = ["binary"]
num_distances = ["euclidean"]
bin_distances = ["jaccard", "hamming", "dice"]
sub_best = final_df[
    (final_df["algorithm"] == "agglomerative") &
    (final_df["pct_threshold"] == "50%") &
    (
        ((final_df["data_type"].isin(numeric_types)) & (final_df["distance"].isin(num_distances))) |
        ((final_df["data_type"].isin(binary_types)) & (final_df["distance"].isin(bin_distances)))
    )
].dropna()
sub_best['Type'] = sub_best['data_type'].str.capitalize() + " (" + sub_best['distance'].str.capitalize() + ")"
g = sns.FacetGrid(
    sub_best,
    col="metric",
    hue="Type",
    margin_titles=True,
    sharey=False,
    height=5,
    aspect=0.4,
    legend_out=True
)
g.map(sns.lineplot, "k", "score", marker="o")
g.set(ylim=(0, None))
g.add_legend(
    title="Setting",
    bbox_to_anchor=(0.5, 0.15),
    loc="upper center",
    ncol=3,
    frameon=True
)
legend = g._legend
legend.get_frame().set_edgecolor('black')
legend.get_frame().set_facecolor('white')
g.fig.subplots_adjust(top=0.8)
g.fig.suptitle("Binary vs Numeric Input: Agglomerative Clustering (Euclidean)", y=0.95, fontsize=13)
plt.tight_layout(rect=[0, 0.12, 1, 0.97])
output_dir = "Result_figures"
os.makedirs(output_dir, exist_ok=True)
figpath = os.path.join(output_dir, "clustering_3_binary_vs_numeric.png")
plt.savefig(figpath, bbox_inches='tight', dpi=300)
plt.show()
print(f"Plot saved to: {figpath}")

# -----------------------------------------------------------------------
#   CLUSTER INTERPRETATION: LAB DIFFERENCES ACROSS SIGNIFICANT CLUSTERS
# -----------------------------------------------------------------------

# Aggregate all Fisher test results for cluster significance.
# Each CSV contains results for a clustering experiment: this loads, tags, and merges all such summaries.
all_fts = []
for pc_folder in root.glob("*pct_threshold"):
    pc_name = pc_folder.name  # e.g. "50pc_threshold"
    for cluster_folder in pc_folder.iterdir():
        if cluster_folder.is_dir():
            cluster_name = cluster_folder.name  # e.g. "kmeans_raw"
            # Look for all files containing Fisher test results
            for f in cluster_folder.glob("*fisher_test*.csv"):
                df = pd.read_csv(f)
                df["pct_threshold"] = pc_name
                df["cluster_type"] = cluster_name
                df["file_name"] = f.name
                df["k"] = extract_k(f.name)
                df["metric_used"] = extract_metric(f.name)
                df.rename(columns={"Unnamed: 0":"Cluster"}, inplace=True)
                all_fts.append(df)
all_ft = pd.concat(all_fts, ignore_index=True)

# Annotate Fisher results with experiment meta-data and join with clustering metrics table for context.
final_ft = all_ft.copy()
final_ft["pct_threshold"] = final_ft["pct_threshold"].str.replace("pct_threshold", "%")
final_ft["data_type"] = final_ft["cluster_type"].apply(lambda x: get_data_type(x))
final_ft["algorithm"] = final_ft["cluster_type"].apply(lambda x: extract_algorithm(x))
final_ft["distance"] = final_ft.apply(lambda x: extract_distance(x.cluster_type, x.file_name), axis=1)
final_ft = final_ft[['Cluster', 'Corrected_p_value', 'Significant_after_correction',
                     'pct_threshold', 'k', 'metric_used', 'data_type', 'algorithm', 'distance']]
# Merge Fisher test results with full clustering metrics
final_ft = final_ft.merge(final_df, on=['k', 'data_type', 'algorithm', 'distance', 'pct_threshold'])
# Save comprehensive Fisher+metrics table for reporting and traceability
folder_name = "cluster analysis"
file_name = "final_dataframe.csv"
file_path = os.path.join(folder_name, file_name)
os.makedirs(folder_name, exist_ok=True)
final_ft.to_csv(file_path, index=False)
print(f"Saved DataFrame to: {file_path}")

# Preview: Top non-binary and binary clusters ranked by silhouette score
final_ft[(final_ft["Significant_after_correction"])
        & (final_ft["data_type"]!="binary")
        & (final_ft["metric"]=="silhouette")
        ].sort_values("score", ascending = False).head(20)
final_ft[(final_ft["Significant_after_correction"])
        & (final_ft["data_type"]=="binary")
        & (final_ft["metric"]=="silhouette")
        ].sort_values("score", ascending = False).head(20)

#   --------------------------------------------------
#     Cluster Interpretation Pipeline Implementation
#   --------------------------------------------------

# Interpretation of cluster content is done via the following workflow:
#    (a) Aggregate per-patient, per-lab statistics (mean, std, min, max) BEFORE imputation,
#    (b) Merge lab statistics with cluster assignments and Fisher cluster significance from prior steps,
#    (c) Visualize lab distributions across clusters and test for lab-level statistical differences,
#    (d) Output boxed-plot figures and Kruskal/Wilcoxon test results for manuscript/reporting.

def load_cluster_assignments(csv_path):
    """
    Loads cluster assignments mapping each hadm_id to a cluster label.
    Args:
        csv_path (str): Path to cluster assignment CSV ('hadm_id', 'cluster_label')
    Returns:
        pd.DataFrame: DataFrame with assignments for merging.
    """
    df = pd.read_csv(csv_path)
    if 'hadm_id' not in df.columns:
        raise ValueError("Cluster assignments CSV must have 'hadm_id' column")
    return df

def load_fisher_results(csv_path):
    """
    Loads Fisher test results for clusters (e.g., significance after Bonferroni/FDR correction)
    """
    df = pd.read_csv(csv_path)
    if 'Significant_after_correction' not in df.columns:
        raise ValueError("Fisher results CSV must have 'Significant_after_correction' column")
    return df

def aggregate_lab_stats(lab_df):
    """
    Summarizes (mean, std, min, max) raw lab measurements for each admission/lab pair.
    Args:
        lab_df (pd.DataFrame): ['hadm_id', 'itemid', 'valuenum'] for preprocessed labs.
    Returns:
        pd.DataFrame: ('hadm_id', 'itemid', 'mean', 'std', 'min', 'max')
    """
    logger.info("Aggregating lab measurements per admission and lab...")
    agg = lab_df.groupby(['hadm_id', 'itemid'])['valuenum'].agg(['mean', 'std', 'min', 'max']).reset_index()
    logger.info(f"Aggregated lab stats: {agg.shape[0]} rows\n{agg.head(5)}\n")
    return agg

def merge_cluster_lab_data(lab_agg_df, cluster_assign_df, fisher_results_df):
    """
    Combines aggregated lab statistics with cluster assignments and significance label for each cluster.
    Args:
        lab_agg_df: Lab stats (mean/std/min/max) per hadm_id/itemid
        cluster_assign_df: Cluster assignment per hadm_id
        fisher_results_df: Significance label per cluster
    Returns:
        pd.DataFrame: Ready for plotting/statistical testing
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
    Creates boxplots showing mean values for each of the N most frequent labs, grouped by cluster and cluster significance.
    Args:
        df : DataFrame from merge_cluster_lab_data
        output_dir : directory for boxplots
        top_n : number of labs to visualize (by prevalence/count)
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
    Runs Kruskal-Wallis (across clusters) and Mann-Whitney U (significant vs non-significant clusters) for each lab.
    Applies Benjamini-Hochberg FDR correction for multiple testing.
    Saves all results as summary CSV.
    Args:
        df: output from merge_cluster_lab_data
        output_dir: where to write summary CSV
    Returns:
        results_df: DataFrame of test p-values and summary statistics per lab
    """
    logger.info("Running statistical tests for each lab...")
    os.makedirs(output_dir, exist_ok=True)
    results = []
    lab_ids = df['itemid'].unique()
    # Compute p-values across labs
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
    # Multiple hypothesis correction for both tests (Benjamini-Hochberg FDR)
    from statsmodels.stats.multitest import multipletests
    correction = "fdr_bh"
    for test in ['kruskal_p', 'mannwhitney_p']:
        mask = results_df[test].notnull()
        corrected = multipletests(results_df.loc[mask, test], method=correction)
        results_df.loc[mask, f'{test}_adj'] = corrected[1]
    results_path = os.path.join(output_dir, 'statistical_test_results.csv')
    results_df.to_csv(results_path, index=False)
    logger.info(f"Saved statistical test results to {results_path}\n")
    return results_df

def run_full_analysis(lab_pkl_path, cluster_csv_path, fisher_csv_path, output_base_dir, PCT_LABEL, top_n_labs):
    """
    Wrapper for full cluster-level interpretation pipeline:
      - Loads input lab/cluster/Fisher files
      - Aggregates lab features/statistics
      - Merges with cluster significance
      - Visualizes lab-centered boxplots for the most frequent labs
      - Runs FDR-corrected statistical comparison across clusters
    Args:
        lab_pkl_path (str): Path to filtered_common_labs_Xpct.pkl
        cluster_csv_path (str): Path to significant cluster assignments (Spectral/other)
        fisher_csv_path (str): Path to Fisher significance table
        output_base_dir (str): Folder to store plots/statistics
        PCT_LABEL (str): Threshold label (e.g., "1pct") for bookkeeping
        top_n_labs (int): Number of labs to plot
    Returns:
        merged (pd.DataFrame): Ready-for-plotting, joined dataset
        stats_df (pd.DataFrame): Lab-level test results (with FDR correction)
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

#  ---------------------------------------------------------------------------------
#    Example: Running full cluster interpretation for spectral k=12 (1% threshold)
#  ---------------------------------------------------------------------------------

# The following block demonstrates how to run the interpretation pipeline for your key results.
if __name__ == "__main__":
    # You may need to change the threshold, metric, etc. as required.
    threshold = 1                  # Select one of the [1,50,75]
    PCT_LABEL = f"{threshold}pct"
    metric = 'silhouette'   # Select one of the [silhouette,calinski_harabasz,davies_bouldin]
    cluster_number = 12            # Select cluster='X', considering the fact that "Spectral_cluster_assignments_k=X" and "Spectral_fisher_test_{metric}_k=X" exist in Spectral_clustering_imputed directory
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

#  ------------------------------------------------------------------------------
#   Fisher's test significance table per cluster (Spectral, 1%, k=12) with size
#  ------------------------------------------------------------------------------

# Load Fisher's test result and combine with cluster assignment for summary table
fisher_path = "metric_results/1pct_threshold/Spectral_clustering_imputed/Spectral_fisher_test_silhouette_k=12.csv"
df_fisher = pd.read_csv(fisher_path)
assign_path = "metric_results/1pct_threshold/Spectral_clustering_imputed/Spectral_cluster_assignments_k=12.csv"
df_assign = pd.read_csv(assign_path)
# Compute cluster size for each label
cluster_col = 'cluster_label'
sizes = df_assign[cluster_col].value_counts().sort_index()
df_fisher['Cluster_size'] = df_fisher['Cluster'].map(sizes)
# Compose significance summary table for reporting
out_cols = ['Cluster', 'Cluster_size', 'Raw_p_value', 'Corrected_p_value', 'Significant_after_correction']
table = df_fisher[out_cols]
print(table)
output_dir = "Result_figures"
os.makedirs(output_dir, exist_ok=True)
table_path = os.path.join(output_dir, "spectral_cluster_significance_table.csv")
table.to_csv(table_path, index=False)
print(f"Cluster significance table saved to: {table_path}")

#  --------------------------------------------------------------------
#   Cluster Analysis Figure (Metrics & Adjusted Mann-Whitney p-values)
#  --------------------------------------------------------------------

# Top: spectral clustering metrics for k (highlighting chosen k, e.g., 12)
# Bottom: -log10(adjusted Mann-Whitney) for top labs (lab name annotated)
metrics_path = "metric_results/1pct_threshold/Spectral_clustering_imputed/Spectral_metrics_score.csv"
stat_csv = "cluster_interpretation/1pct_threshold/for_clusters_k=12/statistics/statistical_test_results.csv"
lab_lookup_path = "cluster_interpretation/d_labitems.csv"
df_metrics = pd.read_csv(metrics_path)
df_stats = pd.read_csv(stat_csv)
lab_lookup = pd.read_csv(lab_lookup_path)[['itemid', 'label']].drop_duplicates()

# Benjamini-Hochberg FDR correction applied
for test in ['kruskal_p', 'mannwhitney_p']:
    mask = df_stats[test].notnull()
    corrected = multipletests(df_stats.loc[mask, test], method='fdr_bh')
    df_stats.loc[mask, f'{test}_adj'] = corrected[1]

# Merge lab names for nice x-axis display
df_stats = df_stats.merge(lab_lookup, left_on='lab_id', right_on='itemid', how='left')
def merge_label(row):
    if pd.notnull(row['label']):
        return f"{row['label']}\n({int(row['lab_id'])})"
    else:
        return f"{int(row['lab_id'])}"
df_stats['label_with_id'] = df_stats.apply(merge_label, axis=1)
# Use MANH-WHITNEY FDR-adjusted p-values for top lab plot
pvals = df_stats['mannwhitney_p_adj'].replace({0: 1e-300})
df_stats["neglog_p"] = -np.log10(pvals)
df_stats_sorted = df_stats.sort_values('neglog_p', ascending=False).reset_index(drop=True)

# Prepare two-row layout; top: metrics; bottom: bar-plot of –log10(p) for top 20 labs
fig = plt.figure(figsize=(22, 17))
gs = gridspec.GridSpec(2, 3, height_ratios=[1, 1.3])
metrics_list = ['silhouette', 'calinski_harabasz', 'davies_bouldin']
colors = ['r', 'b', 'g']
titles = ["Silhouette Score", "Calinski-Harabasz Index", "Davies-Bouldin Index"]
ks = df_metrics['k']

# Top row: Spectral metrics
for i, metric in enumerate(metrics_list):
    ax = fig.add_subplot(gs[0, i])
    ax.plot(ks, df_metrics[metric], marker='o', color=colors[i], lw=2)
    ax.set_title(titles[i], fontsize=18)
    ax.set_xlabel("k", fontsize=17)
    ax.set_ylabel("score", fontsize=17)
    ax.axvline(12, color='grey', linestyle='--', alpha=0.7, lw=2)
    ax.tick_params(axis='both', labelsize=13)
fig.suptitle("Spectral Clustering Metrics on Imputed Data (1% threshold)", fontsize=22, y=0.945)

# Bottom row: barplot of adjusted -log10(Mann-Whitney) for top 20 labs
ax_bar = fig.add_subplot(gs[1, :])
N_LABELS = 20
tick_positions = np.arange(N_LABELS)
tick_labels = df_stats_sorted['label_with_id'].values[:N_LABELS]
sig_line = -np.log10(0.05)
bars = ax_bar.bar(
    x=np.arange(len(df_stats_sorted)),
    height=df_stats_sorted["neglog_p"],
    color="tab:blue"
)
ax_bar.axhline(sig_line, color='red', linestyle='--', lw=2, label='Significance threshold (p=0.05)')
ax_bar.set_xticks(tick_positions)
ax_bar.set_xticklabels(tick_labels, rotation=65, ha="right", fontsize=14)
ax_bar.set_xlim(-1, N_LABELS + 1)
ax_bar.set_xlabel("Lab Test Name (Lab ID)", fontsize=17)
ax_bar.set_ylabel("-log10(Mann-Whitney p_adj)", fontsize=17)
ax_bar.set_title("Cluster Analysis: Lab Tests Differentiating Significant Clusters (Spectral, k=12, 1% threshold)", fontsize=22, y=1.05)
ax_bar.legend(fontsize=15)
plt.tight_layout(rect=[0, 0, 1, 0.95])
output_dir = "Result_figures"
os.makedirs(output_dir, exist_ok=True)
save_path = os.path.join(output_dir, "clustering_metrics_and_mannwhitney_barplot_adj.png")
plt.savefig(save_path, bbox_inches='tight', dpi=300)
plt.show()
print(f"Combined figure saved to: {save_path}")
