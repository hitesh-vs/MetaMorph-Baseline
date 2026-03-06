import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib.lines import Line2D
import numpy as np
import pickle
from sklearn.manifold import TSNE
from sklearn.neighbors import NearestNeighbors

prefix = "./output_tsne/tsne_data"

X      = np.load(f"{prefix}_embeds.npy")
labels = pickle.load(open(f"{prefix}_labels.pkl", "rb"))

print(f"Loaded {len(X)} embeddings of dim {X.shape[1]}")
print(f"Unique robots:   {sorted(set(l['robot']    for l in labels))}")
print(f"Unique semantic: {sorted(set(l['semantic'] for l in labels))}")
print(f"Unique depths:   {sorted(set(l['depth']    for l in labels))}")

semantic = np.array([l["semantic"] for l in labels])
robots   = np.array([l["robot"]   for l in labels])
depths   = np.array([str(l["depth"]) for l in labels])

X = X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-8)

print(f"Running t-SNE on {X.shape[0]} vectors...")
Z = TSNE(n_components=2, perplexity=40, metric="cosine",
         max_iter=2000, random_state=42).fit_transform(X)

MARKERS = ['o', 's', '^', 'D', 'v', 'P', '*', 'X', 'h', '+']

def scatter(ax, Z, labels, robots, title):
    unique_labels = sorted(set(labels))
    unique_robots = sorted(set(robots))
    cmap          = cmap = cm.get_cmap("tab20", len(unique_labels))
    label2color   = {l: cmap(i) for i, l in enumerate(unique_labels)}
    robot2marker  = {r: MARKERS[i % len(MARKERS)] for i, r in enumerate(unique_robots)}

    for robot in unique_robots:
        for lbl in unique_labels:
            mask = (labels == lbl) & (robots == robot)
            if mask.sum() == 0:
                continue
            ax.scatter(Z[mask, 0], Z[mask, 1],
                       c=[label2color[lbl]],
                       marker=robot2marker[robot],
                       alpha=0.6, s=35)

    # color legend — link types
    color_handles = [
        Line2D([0],[0], marker='o', color='w',
               markerfacecolor=label2color[l], markersize=8, label=l)
        for l in unique_labels
    ]
    # shape legend — robots
    shape_handles = [
        Line2D([0],[0], marker=robot2marker[r], color='gray',
               markersize=8, label=r)
        for r in unique_robots
    ]

    leg1 = ax.legend(handles=color_handles, title="Link type",
                     fontsize=7, loc="upper left")
    ax.add_artist(leg1)
    ax.legend(handles=shape_handles, title="Robot",
              fontsize=7, loc="upper right")
    ax.set_title(title)

fig, axes = plt.subplots(1, 3, figsize=(24, 7))
scatter(axes[0], Z, semantic, robots, "Joint type (semantic)")
scatter(axes[1], Z, robots,   robots, "Robot identity  ← want MIXED")
scatter(axes[2], Z, depths,   robots, "Topological depth")
plt.suptitle("t-SNE of transformer input embedding (baseline: no GCN)")
plt.tight_layout()
plt.savefig(f"{prefix}_tsne.png", dpi=150)
print(f"Saved → {prefix}_tsne.png")

for lbl_arr, name in [(semantic, "semantic"), (robots, "robot"), (depths, "depth")]:
    nn = NearestNeighbors(n_neighbors=16, metric="euclidean").fit(Z)
    _, idx = nn.kneighbors(Z)
    idx = idx[:, 1:]
    purity = np.mean([np.mean(lbl_arr[idx[i]] == lbl_arr[i])
                      for i in range(len(lbl_arr))])
    print(f"  KNN purity [{name:>10s}]: {purity:.3f}  "
          f"(chance={1/len(set(lbl_arr)):.3f})")