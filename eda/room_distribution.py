import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Point these to your files (it will concatenate whichever exist)
CSV_PATHS = [
    "data/train.csv",
    # "data/val.csv",
    # "data/test.csv",   # optional; can include
]

ROOM_CLUSTERS = [
    "andre områder","kjøkken","stue","gang","soverom","bad",
    "bod","vaskerom","wc","kjeller","garasje",
]

def cluster_room(room_name: str) -> str:
    if not isinstance(room_name, str):
        return "ukjent"
    name = room_name.lower()
    if name in ROOM_CLUSTERS:
        return name
    for r in ROOM_CLUSTERS:
        if r in name:
            return r
    return "ukjent"

# ---- Load available CSVs ----
frames = []
for path in CSV_PATHS:
    if os.path.exists(path):
        frames.append(pd.read_csv(path))
    # silently skip missing ones

if not frames:
    raise FileNotFoundError(
        "No input CSVs found. Update CSV_PATHS to point at your dataset (e.g., data/train.csv)."
    )

df = pd.concat(frames, ignore_index=True)

# Must have these columns:
assert {"project_id","room"} <= set(df.columns), "CSV must contain 'project_id' and 'room' columns"

# Cluster rooms like your dataset
df["room_cluster"] = df["room"].map(cluster_room)

# For each project, list its unique clusters
proj_clusters = (
    df.groupby("project_id")["room_cluster"]
      .unique()
      .reset_index()
)

# Expand into (project_id, room_cluster, num_other_clusters)
records = []
for _, row in proj_clusters.iterrows():
    pid = row["project_id"]
    clusters = list(row["room_cluster"])
    k_other = len(clusters) - 1
    for rc in clusters:
        records.append((pid, rc, k_other))
dist_df = pd.DataFrame(records, columns=["project_id","room_cluster","num_other_clusters"])
dist_df["num_other_clusters"] = dist_df["num_other_clusters"].clip(0, 11)

# Count projects per (cluster, k)
agg = (
    dist_df.groupby(["room_cluster","num_other_clusters"])["project_id"]
           .nunique()
           .reset_index(name="count_projects")
)

# Ensure a full grid of clusters × {0..11}
all_clusters = ROOM_CLUSTERS + ["ukjent"]
grid = pd.MultiIndex.from_product([all_clusters, list(range(12))],
                                  names=["room_cluster","num_other_clusters"])
agg = agg.set_index(["room_cluster","num_other_clusters"]).reindex(grid, fill_value=0).reset_index()

# Convert to percentages within each cluster
totals = agg.groupby("room_cluster")["count_projects"].transform("sum").replace(0, np.nan)
agg["pct_projects"] = (agg["count_projects"] / totals * 100).fillna(0.0)

# Pivot for plotting
mat = agg.pivot(index="room_cluster", columns="num_other_clusters", values="pct_projects").fillna(0.0)
mat = mat.reindex(all_clusters)  # order rows

# ---- Plot: stacked bars ----
fig, ax = plt.subplots(figsize=(14, 7))
x = np.arange(len(mat.index))
bottoms = np.zeros(len(mat.index))

for k in range(12):  # 0..11
    y = mat[k].to_numpy() if k in mat.columns else np.zeros(len(mat.index))
    ax.bar(x, y, bottom=bottoms, label=str(k))
    bottoms += y

ax.set_xticks(x)
ax.set_xticklabels(mat.index, rotation=30, ha="right")
ax.set_ylabel("Share of projects (%)")
ax.set_xlabel("Room cluster")
ax.set_title("Distribution of number of other room clusters per project (0..11)")
ax.legend(title="# other clusters", ncol=6, fontsize=8)

plt.tight_layout()
plt.show()