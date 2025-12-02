from collections import OrderedDict

import matplotlib.colors as colors
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from matplotlib.patches import RegularPolygon
# We need this for the advanced layout
import matplotlib.gridspec as gridspec

import matplotlib.patheffects as path_effects

# Assuming this import is correct in your environment
# from neural.chess_model_relative_coords import all_hex_coords
from neural.model_factory import model_factory


def load_model(item):
    model = model_factory("ChessDomain")

    dir_path = "./../../testing/chess2p/"
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

    return model


def get_cartesian_flat_top(q, r, size=1.0):
    """
    Standard conversion from axial hex coordinates (q, r) to Cartesian (x, y)
    for a FLAT-TOPPED orientation.
    """
    x = size * (3 / 2 * q)
    y = size * (np.sqrt(3) / 2 * q + np.sqrt(3) * r)
    return x, y


def visualize_multiple_references(model, references, epoch_name=""):
    """
    Visualizes cosine similarity for multiple reference offsets in a single figure
    with a 2x2 subplot layout and a shared, shorter colorbar.
    """
    model.eval()

    # --- THE ROBUST 2x2 LAYOUT SOLUTION USING GridSpec ---
    # 1. Create a figure instance, adjusted for a 2x2 layout
    fig = plt.figure(figsize=(10, 9))

    # 2. Create a GridSpec layout with more "virtual" rows to control colorbar height.
    #    We use 10 rows and 3 columns. The plots will occupy halves of the grid.
    gs = gridspec.GridSpec(10, 3,
                           width_ratios=[1, 1, 0.05], # Plots are wide, colorbar is narrow
                           wspace=0.05, hspace=0.1)

    # 3. Create subplot axes in the 2x2 grid using slices of the GridSpec
    ax1 = fig.add_subplot(gs[0:5, 0])   # Top-left
    ax2 = fig.add_subplot(gs[0:5, 1])   # Top-right
    ax3 = fig.add_subplot(gs[5:10, 0])  # Bottom-left
    ax4 = fig.add_subplot(gs[5:10, 1])  # Bottom-right

    # 4. Create the colorbar axis. By using a slice (e.g., from row 1 to 9),
    #    it becomes shorter than the total height of the plots.
    cax = fig.add_subplot(gs[3:7, 2])

    axes = [ax1, ax2, ax3, ax4]

    # --- Pre-computation and Shared Elements ---
    rel_pos_embeddings = model.relative_pos_embedding_table.weight.detach().cpu()
    max_rel_dist = model.max_rel_dist
    range_size = model.range_size
    offset = max_rel_dist

    # cmap = plt.get_cmap('RdBu')
    cmap = plt.get_cmap('RdBu')
    div_norm = colors.TwoSlopeNorm(vmin=-1., vcenter=0., vmax=1.)

    # --- Loop through each subplot axis and its corresponding reference offset ---
    for ax, (ref_dq, ref_dr) in zip(axes, references):
        ax.set_aspect('equal')
        if not (-max_rel_dist <= ref_dq <= max_rel_dist and -max_rel_dist <= ref_dr <= max_rel_dist):
            ax.text(0.5, 0.5, "Reference out of bounds", ha='center', va='center')
            continue

        ref_index = (ref_dq + offset) * range_size + (ref_dr + offset)
        reference_embedding = rel_pos_embeddings[ref_index].unsqueeze(0)
        similarities = F.cosine_similarity(reference_embedding, rel_pos_embeddings, dim=1)
        similarity_grid = similarities.view(range_size, range_size).numpy()

        hex_patches = {}
        for dq_idx in range(range_size):
            for dr_idx in range(range_size):
                q_idx = dq_idx - offset
                r_idx = dr_idx - offset
                s = -q_idx - r_idx
                distance = (abs(q_idx) + abs(r_idx) + abs(s)) // 2

                if distance > 8:
                    continue

                dq = dq_idx - offset
                dr = dr_idx - offset
                similarity = similarity_grid[dq_idx, dr_idx]
                color = cmap(div_norm(similarity))
                x, y = get_cartesian_flat_top(dq, dr)

                hexagon = RegularPolygon((x, y), numVertices=6, radius=1.0,
                                         orientation=np.deg2rad(30),
                                         facecolor=color, edgecolor='k', linewidth=0.1)
                ax.add_patch(hexagon)
                hex_patches[(dq, dr)] = hexagon

        print(f"Showing {len(hex_patches)} out of {range_size * range_size} hexes for reference ({ref_dq}, {ref_dr})")

        ref_c = "#e67147"
        origin_c = "#6a7ade"

        # --- Formatting and Highlighting for the current subplot ---
        ref_hex = hex_patches.get((ref_dq, ref_dr))
        if ref_hex:
            ref_hex.set_edgecolor(ref_c)
            ref_hex.set_linewidth(2.5)
            ref_hex.set_zorder(10)

        center_hex = hex_patches.get((0, 0))
        if center_hex:
            center_hex.set_edgecolor(origin_c)
            center_hex.set_linewidth(2.5)
            center_hex.set_zorder(10)

        ref_x, ref_y = get_cartesian_flat_top(ref_dq, ref_dr)

        outline_effect = [path_effects.withStroke(linewidth=2, foreground='black')]

        ax.text(ref_x, ref_y + 2.0, 'Reference', ha='center', va='center',
                color=ref_c, fontsize=12, fontweight='bold', path_effects=outline_effect)

        center_x, center_y = get_cartesian_flat_top(0, 0)
        ax.text(center_x, center_y - 2, 'Self', ha='center', va='center',
                color=origin_c, fontsize=12, fontweight='bold', path_effects=outline_effect)

        ax.autoscale_view()
        ax.axis('off')

    mappable = plt.cm.ScalarMappable(norm=div_norm, cmap=cmap)
    fig.colorbar(mappable, cax=cax, label="Cosine Similarity")

    # Use the original tight_layout call
    fig.tight_layout(rect=[0, 0.03, 1, 0.95])

    # Use the original save path, adding the epoch name for uniqueness
    save_path = f"/Users/janvasiljevic/Faks/mag/mag-thesis/figures/results/chess/pos_embeddings.pdf"
    fig.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Figure saved to {save_path}")
    plt.show() # Display the plot


if __name__ == "__main__":
    models_to_load = [
        "475.pt",
    ]

    for item in models_to_load:
        model = load_model(item)
        epoch_name = item.replace('.pt', '')

        # Using your original reference points
        references_to_plot = [
            (1, 2),
            (3, 0),
            (1, 1),
            (-2, 4)
        ]

        visualize_multiple_references(model, references_to_plot, epoch_name=epoch_name)