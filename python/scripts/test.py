from collections import OrderedDict
import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as colors
from matplotlib.patches import RegularPolygon

# --- Model Loading Section (same as before) ---
from neural.model_factory import model_factory

model = model_factory("ChessDomain")

dir_path = "./../../testing/chess2p/"
item = "475.pt"

try:
    checkpoint = torch.load(f"{dir_path}{item}", map_location="cpu")
    if any(k.startswith("_orig_mod.") for k in checkpoint.keys()):
        new_state_dict = OrderedDict()
        for k, v in checkpoint.items():
            new_state_dict[k.replace("_orig_mod.", "")] = v
        model.load_state_dict(new_state_dict)
    else:
        model.load_state_dict(checkpoint)
    print(f"Successfully loaded model from {dir_path}{item}")

except FileNotFoundError:
    print(f"Error: Model file not found at {dir_path}{item}.")
    print("Using a randomly initialized model for demonstration.")
# --- End of model loading section ---


# --- Re-usable Visualization and Helper Functions ---
def get_cartesian_flat_top(q, r, size=1.0):
    x = size * (3/2 * q)
    y = size * (np.sqrt(3)/2 * q + np.sqrt(3) * r)
    return x, y

def visualize_scores_on_hex_grid(score_grid, title):
    fig, ax = plt.subplots(1, figsize=(10, 10))
    ax.set_aspect('equal')

    range_size = score_grid.shape[0]
    offset = (range_size - 1) // 2

    cmap = plt.get_cmap('RdBu_r')
    max_abs_val = np.max(np.abs(score_grid))
    if max_abs_val == 0: max_abs_val = 1.0 # Avoid division by zero
    norm = colors.TwoSlopeNorm(vmin=-max_abs_val, vcenter=0., vmax=max_abs_val)

    for dq_idx in range(range_size):
        for dr_idx in range(range_size):
            dq = dq_idx - offset
            dr = dr_idx - offset

            score = score_grid[dq_idx, dr_idx]
            color = cmap(norm(score))

            x, y = get_cartesian_flat_top(dq, dr)

            hexagon = RegularPolygon((x, y), numVertices=6, radius=1.0,
                                     orientation=np.deg2rad(30),
                                     facecolor=color, edgecolor='k', linewidth=0.2)
            ax.add_patch(hexagon)

    plt.title(title, fontsize=16)
    mappable = plt.cm.ScalarMappable(norm=norm, cmap=cmap)
    fig.colorbar(mappable, ax=ax, shrink=0.7, label="Activation Value")

    ax.autoscale_view()
    plt.axis('off')
    plt.show()

def visualize_embedding_dimensions(model):
    """
    Visualizes each dimension of the relative position embedding table as a heatmap.
    This reveals the learned geometric basis vectors of the spatial attention space.
    """
    model.eval()

    rel_pos_embeddings = model.relative_pos_embedding_table.weight.detach().cpu()
    d_head = model.d_head
    range_size = model.range_size

    print(f"\n--- Visualizing {d_head} dimensions of the Relative Position Embedding Table ---")

    # We will visualize a few interesting dimensions, not all of them
    # to avoid spamming plots. You can change this.
    dims_to_show = min(d_head, 64)
    # Ko napise 27 je zanimiv!!
    indices_to_show = np.linspace(0, d_head - 1, dims_to_show, dtype=int)

    for d in indices_to_show:
        # Extract the d-th column (values for this dimension across all offsets)
        dimension_values = rel_pos_embeddings[:, d]

        # Reshape into a 2D grid
        score_grid = dimension_values.view(range_size, range_size).numpy()

        # Check if this dimension has learned anything interesting (variance > 0)
        if np.std(score_grid) > 1e-4:
            title = f'Relative Embedding -- Dimension {d}/{d_head}'
            visualize_scores_on_hex_grid(score_grid, title)
        else:
            print(f"Skipping dimension {d} (zero variance).")

# --- Example Usage ---
visualize_embedding_dimensions(model)