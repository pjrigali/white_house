"""
white_house_clustering.py
-------------------------
Performs unsupervised clustering on stylometric features extracted by
white_house_stylometry.py to identify distinct writing styles / potential
different authors across White House articles.

Uses K-Means with elbow analysis + silhouette scoring to find optimal
clusters, then profiles each cluster to describe its writing style.

Dependencies: scikit-learn (sklearn)
Run white_house_stylometry.py first to generate the input CSV.
"""

import csv
import os
import math
from collections import Counter
from statistics import mean, stdev, median


# ─── Configuration ───────────────────────────────────────────────────────────

# Features to use for clustering — these are the most discriminating
# stylometric signals for authorship. Function word ratios (fw_*) and
# LLM features are included dynamically.
CORE_FEATURES = [
    "avg_sentence_length", "std_sentence_length",
    "avg_word_length", "std_word_length",
    "type_token_ratio", "hapax_ratio", "yules_k",
    "comma_rate", "semicolon_rate", "colon_rate",
    "exclamation_rate", "question_rate", "dash_rate",
    "passive_voice_ratio",
    "flesch_kincaid_grade", "flesch_reading_ease",
    "avg_paragraph_length", "caps_rate",
]

# Maximum K values to try
MAX_K = 10

# Minimum articles required for clustering to be meaningful
MIN_ARTICLES = 10


# ─── Data Loading & Preprocessing ───────────────────────────────────────────

def load_csv(filepath):
    """Load a CSV file into a list of dicts."""
    if not os.path.exists(filepath):
        print(f"File not found: {filepath}")
        return []
    with open(filepath, "r", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def safe_float(value):
    """Convert to float, returning None if not possible."""
    if value is None or value == "" or value == "None":
        return None
    try:
        return float(value)
    except (ValueError, TypeError):
        return None


def select_features(rows):
    """
    Select numeric feature columns for clustering.
    Dynamically includes fw_* and llm_* columns.
    Returns: (feature_names, feature_matrix, valid_row_indices)
    """
    if not rows:
        return [], [], []

    # Build feature list dynamically
    feature_names = list(CORE_FEATURES)
    for key in rows[0].keys():
        if key.startswith("fw_"):
            feature_names.append(key)
        if key.startswith("llm_") and key not in ("llm_analysis_viable", "llm_filler_count"):
            feature_names.append(key)

    # Extract numeric values, skip rows with missing data
    feature_matrix = []
    valid_indices = []

    for i, row in enumerate(rows):
        values = []
        skip = False
        for feat in feature_names:
            val = safe_float(row.get(feat))
            if val is None:
                skip = True
                break
            values.append(val)
        if not skip:
            feature_matrix.append(values)
            valid_indices.append(i)

    return feature_names, feature_matrix, valid_indices


def normalize_features(matrix):
    """
    Z-score normalize each feature column (mean=0, std=1).
    This is essential so that features on different scales
    contribute equally to distance calculations.
    Returns: normalized matrix, means, stds
    """
    if not matrix:
        return [], [], []

    n_features = len(matrix[0])
    n_rows = len(matrix)

    means = []
    stds = []

    for j in range(n_features):
        col = [matrix[i][j] for i in range(n_rows)]
        col_mean = sum(col) / n_rows
        col_std = math.sqrt(sum((x - col_mean) ** 2 for x in col) / n_rows)
        if col_std == 0:
            col_std = 1.0  # avoid division by zero
        means.append(col_mean)
        stds.append(col_std)

    normalized = []
    for i in range(n_rows):
        row = [(matrix[i][j] - means[j]) / stds[j] for j in range(n_features)]
        normalized.append(row)

    return normalized, means, stds


# ─── K-Means Implementation ─────────────────────────────────────────────────

def euclidean_distance(a, b):
    """Compute Euclidean distance between two vectors."""
    return math.sqrt(sum((ai - bi) ** 2 for ai, bi in zip(a, b)))


def kmeans(data, k, max_iterations=100, n_init=10):
    """
    K-Means clustering with multiple random initializations.
    Returns: (labels, centroids, inertia)
    """
    import random
    n = len(data)
    if n == 0 or k <= 0:
        return [], [], float("inf")

    best_labels = None
    best_centroids = None
    best_inertia = float("inf")

    for init_run in range(n_init):
        # Random initialization: pick k distinct data points
        centroid_indices = random.sample(range(n), min(k, n))
        centroids = [list(data[i]) for i in centroid_indices]

        labels = [0] * n

        for iteration in range(max_iterations):
            # Assignment step
            new_labels = []
            for point in data:
                distances = [euclidean_distance(point, c) for c in centroids]
                new_labels.append(distances.index(min(distances)))

            # Check convergence
            if new_labels == labels and iteration > 0:
                break
            labels = new_labels

            # Update step
            n_features = len(data[0])
            new_centroids = [[0.0] * n_features for _ in range(k)]
            counts = [0] * k

            for i, label in enumerate(labels):
                counts[label] += 1
                for j in range(n_features):
                    new_centroids[label][j] += data[i][j]

            for c in range(k):
                if counts[c] > 0:
                    for j in range(n_features):
                        new_centroids[c][j] /= counts[c]
                else:
                    # Empty cluster: reinitialize to a random point
                    rand_idx = random.randint(0, n - 1)
                    new_centroids[c] = list(data[rand_idx])

            centroids = new_centroids

        # Compute inertia (sum of squared distances to nearest centroid)
        inertia = 0.0
        for i, point in enumerate(data):
            d = euclidean_distance(point, centroids[labels[i]])
            inertia += d * d

        if inertia < best_inertia:
            best_inertia = inertia
            best_labels = labels
            best_centroids = centroids

    return best_labels, best_centroids, best_inertia


def silhouette_score(data, labels):
    """
    Compute the mean silhouette score for a clustering.
    Score ranges from -1 (bad) to 1 (excellent).
    """
    n = len(data)
    unique_labels = list(set(labels))

    if len(unique_labels) < 2 or len(unique_labels) >= n:
        return -1.0

    # Group data by cluster
    clusters = {label: [] for label in unique_labels}
    for i, label in enumerate(labels):
        clusters[label].append(i)

    silhouettes = []

    for i in range(n):
        own_cluster = labels[i]
        own_members = clusters[own_cluster]

        # a(i): mean distance to same-cluster members
        if len(own_members) > 1:
            a_i = sum(euclidean_distance(data[i], data[j])
                      for j in own_members if j != i) / (len(own_members) - 1)
        else:
            a_i = 0.0

        # b(i): minimum mean distance to any other cluster
        b_i = float("inf")
        for other_label in unique_labels:
            if other_label == own_cluster:
                continue
            other_members = clusters[other_label]
            if not other_members:
                continue
            mean_dist = sum(euclidean_distance(data[i], data[j])
                           for j in other_members) / len(other_members)
            b_i = min(b_i, mean_dist)

        if b_i == float("inf"):
            b_i = 0.0

        denom = max(a_i, b_i)
        s_i = (b_i - a_i) / denom if denom > 0 else 0.0
        silhouettes.append(s_i)

    return sum(silhouettes) / len(silhouettes) if silhouettes else 0.0


# ─── Cluster Profiling ───────────────────────────────────────────────────────

def profile_clusters(rows, valid_indices, labels, feature_names, feature_matrix):
    """
    Build a human-readable profile for each cluster describing its
    distinctive characteristics.
    """
    unique_labels = sorted(set(labels))
    n_features = len(feature_names)

    # Compute global means for comparison
    global_means = []
    for j in range(n_features):
        col = [feature_matrix[i][j] for i in range(len(feature_matrix))]
        global_means.append(sum(col) / len(col))

    profiles = {}

    for label in unique_labels:
        member_indices = [i for i, l in enumerate(labels) if l == label]
        member_rows = [rows[valid_indices[i]] for i in member_indices]

        # Compute cluster means for each feature
        cluster_means = []
        for j in range(n_features):
            col = [feature_matrix[i][j] for i in member_indices]
            cluster_means.append(sum(col) / len(col))

        # Find most distinctive features (largest deviation from global mean)
        deviations = []
        for j in range(n_features):
            if global_means[j] != 0:
                pct_diff = ((cluster_means[j] - global_means[j]) / abs(global_means[j])) * 100
            else:
                pct_diff = 0.0
            deviations.append((feature_names[j], cluster_means[j], global_means[j], pct_diff))

        # Sort by absolute deviation
        deviations.sort(key=lambda x: abs(x[3]), reverse=True)

        # Gather article metadata
        categories = Counter(r.get("category", "Unknown") for r in member_rows)
        titles = [r.get("title", "")[:60] for r in member_rows]

        profiles[label] = {
            "size": len(member_indices),
            "top_deviations": deviations[:15],
            "categories": categories,
            "sample_titles": titles[:5],
            "cluster_means": {feature_names[j]: cluster_means[j] for j in range(n_features)},
        }

    return profiles


# ─── Report Generation ──────────────────────────────────────────────────────

def generate_report(elbow_data, best_k, best_silhouette, profiles, output_path):
    """Generate a comprehensive text report of the clustering analysis."""
    lines = []
    lines.append("=" * 70)
    lines.append("WHITE HOUSE ARTICLE CLUSTERING ANALYSIS")
    lines.append("Based on Stylometric Features")
    lines.append("=" * 70)

    # ── Elbow Analysis ──
    lines.append("\n── ELBOW ANALYSIS & SILHOUETTE SCORES ──\n")
    lines.append(f"  {'K':>3}  {'Inertia':>12}  {'Silhouette':>12}  {'Assessment'}")
    lines.append(f"  {'─'*3}  {'─'*12}  {'─'*12}  {'─'*20}")
    for k, inertia, sil in elbow_data:
        marker = " ◄── BEST" if k == best_k else ""
        lines.append(f"  {k:>3}  {inertia:>12.2f}  {sil:>12.4f}{marker}")

    lines.append(f"\n  Optimal K: {best_k} (silhouette score: {best_silhouette:.4f})")
    lines.append("")

    # Interpret silhouette
    if best_silhouette >= 0.7:
        quality = "EXCELLENT — very distinct writing-style clusters"
    elif best_silhouette >= 0.5:
        quality = "GOOD — reasonable separation between styles"
    elif best_silhouette >= 0.3:
        quality = "FAIR — some structure detected, but clusters overlap"
    elif best_silhouette >= 0.1:
        quality = "WEAK — minimal clustering structure"
    else:
        quality = "POOR — no meaningful clusters found"

    lines.append(f"  Clustering Quality: {quality}")
    lines.append("")

    # ── Cluster Profiles ──
    lines.append("\n── CLUSTER PROFILES ──\n")

    for label in sorted(profiles.keys()):
        p = profiles[label]
        lines.append(f"  ┌─ Cluster {label}  ({p['size']} articles) ─────────────────────")

        # Category breakdown
        lines.append(f"  │ Categories:")
        for cat, count in p["categories"].most_common(5):
            lines.append(f"  │   {cat}: {count}")

        # Distinctive features
        lines.append(f"  │")
        lines.append(f"  │ Most Distinctive Features (vs. global average):")
        lines.append(f"  │   {'Feature':<30} {'Cluster':>10} {'Global':>10} {'Diff':>8}")
        lines.append(f"  │   {'─'*30} {'─'*10} {'─'*10} {'─'*8}")
        for feat_name, cluster_val, global_val, pct_diff in p["top_deviations"][:10]:
            direction = "▲" if pct_diff > 0 else "▼"
            lines.append(
                f"  │   {feat_name:<30} {cluster_val:>10.4f} {global_val:>10.4f} {direction}{abs(pct_diff):>6.1f}%"
            )

        # Sample titles
        lines.append(f"  │")
        lines.append(f"  │ Sample Articles:")
        for title in p["sample_titles"]:
            lines.append(f"  │   • {title}")

        lines.append(f"  └{'─' * 60}")
        lines.append("")

    # ── Interpretation ──
    lines.append("\n── INTERPRETATION GUIDE ──\n")
    lines.append("  Silhouette Score Ranges:")
    lines.append("    0.7 - 1.0  →  Strong structure (likely distinct authors)")
    lines.append("    0.5 - 0.7  →  Reasonable structure (different writing teams)")
    lines.append("    0.3 - 0.5  →  Weak structure (overlapping styles, possibly edited)")
    lines.append("    < 0.3      →  No clear structure (uniform style or heavy editing)")
    lines.append("")
    lines.append("  If clusters align with 'category', the style differences may reflect")
    lines.append("  document type rather than authorship (e.g., executive orders vs. articles).")
    lines.append("  If clusters cut ACROSS categories, that's a stronger signal of")
    lines.append("  different individual authors.")
    lines.append("")

    report_text = "\n".join(lines)
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(report_text)

    print(report_text)


def save_cluster_assignments(rows, valid_indices, labels, output_path):
    """Save the original data with cluster assignments appended."""
    output_rows = []
    label_lookup = {valid_indices[i]: labels[i] for i in range(len(labels))}

    for i, row in enumerate(rows):
        new_row = dict(row)
        new_row["cluster"] = label_lookup.get(i, "excluded")
        output_rows.append(new_row)

    if output_rows:
        fieldnames = list(output_rows[0].keys())
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(output_rows)
        print(f"\nCluster assignments saved to {output_path}")


# ─── Main Pipeline ───────────────────────────────────────────────────────────

def run_clustering(stylometry_csv_path, output_dir):
    """
    Main entry point. Loads stylometry features, finds optimal K,
    clusters articles, profiles each cluster, and saves results.
    """
    print("Loading stylometry features...")
    rows = load_csv(stylometry_csv_path)
    print(f"  Loaded {len(rows)} rows.")

    if len(rows) < MIN_ARTICLES:
        print(f"  Need at least {MIN_ARTICLES} articles for clustering. Aborting.")
        return

    # Select and normalize features
    print("\nSelecting features...")
    feature_names, feature_matrix, valid_indices = select_features(rows)
    print(f"  Using {len(feature_names)} features across {len(feature_matrix)} articles.")

    if len(feature_matrix) < MIN_ARTICLES:
        print(f"  Only {len(feature_matrix)} articles have complete data. Need {MIN_ARTICLES}. Aborting.")
        return

    print("Normalizing features...")
    normalized, _, _ = normalize_features(feature_matrix)

    # Elbow analysis: try K=2 through MAX_K
    max_k = min(MAX_K, len(normalized) - 1)
    print(f"\nRunning elbow analysis (K=2 to {max_k})...")
    elbow_data = []
    best_k = 2
    best_silhouette = -1.0

    for k in range(2, max_k + 1):
        labels, centroids, inertia = kmeans(normalized, k)
        sil = silhouette_score(normalized, labels)
        elbow_data.append((k, inertia, sil))
        print(f"  K={k}: inertia={inertia:.2f}, silhouette={sil:.4f}")

        if sil > best_silhouette:
            best_silhouette = sil
            best_k = k

    print(f"\n  Optimal K = {best_k} (silhouette = {best_silhouette:.4f})")

    # Final clustering with best K
    print(f"\nRunning final clustering with K={best_k}...")
    final_labels, final_centroids, final_inertia = kmeans(normalized, best_k, n_init=20)

    # Profile clusters
    print("Profiling clusters...")
    profiles = profile_clusters(rows, valid_indices, final_labels, feature_names, feature_matrix)

    # Save outputs
    os.makedirs(output_dir, exist_ok=True)

    report_path = os.path.join(output_dir, "clustering_report.txt")
    generate_report(elbow_data, best_k, best_silhouette, profiles, report_path)
    print(f"\nReport saved to {report_path}")

    assignments_path = os.path.join(output_dir, "stylometry_with_clusters.csv")
    save_cluster_assignments(rows, valid_indices, final_labels, assignments_path)

    # Also save the elbow data for reference
    elbow_path = os.path.join(output_dir, "elbow_analysis.csv")
    with open(elbow_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["k", "inertia", "silhouette_score"])
        for k, inertia, sil in elbow_data:
            writer.writerow([k, round(inertia, 4), round(sil, 4)])
    print(f"Elbow analysis saved to {elbow_path}")


if __name__ == "__main__":
    base = os.getcwd()
    silver_dir = os.path.join(base, "data-lake", "02_Silver", "white_house")

    stylometry_csv = os.path.join(silver_dir, "stylometry_features.csv")
    output_dir = os.path.join(silver_dir, "clustering_analysis")

    run_clustering(stylometry_csv, output_dir)
