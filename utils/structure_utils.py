import os

import numpy as np

import pandas as pd
from matplotlib import pyplot as plt
from sklearn.manifold import MDS

from utils.constants import MAIN_DIR


def get_contact_map_from_coords(ca_coords, soft_map=False, threshold=8.0, decay_rate=0.5):
    print(len(ca_coords))
    if len(ca_coords) <= 1:
        return None

    ca_coords = np.array(ca_coords, dtype="float32")
    diff = ca_coords[:, np.newaxis, :] - ca_coords[np.newaxis, :, :]
    distances = np.linalg.norm(diff, axis=-1)

    if soft_map:
        contact_map = np.exp(-decay_rate * distances).astype(int)
    else:
        contact_map = np.where(distances < threshold, 1.0, 0.0)

    return contact_map


def optimize_points_from_distogram(distogram, n_init=4, max_iter=300, random_state=None):
    mds = MDS(n_components=3, dissimilarity="precomputed", n_init=n_init, max_iter=max_iter, random_state=random_state)
    points = mds.fit_transform(distogram)
    return points


def plot_protein_points_cartoon(ground_truth_points, predicted_points, title="Protein 3D Points"):
    fig = plt.figure(figsize=(10, 7))
    ax = fig.add_subplot(111, projection='3d')

    # Plot and connect Ground Truth Points in sequence
    ax.plot(ground_truth_points[:, 0], ground_truth_points[:, 1], ground_truth_points[:, 2],
            color='deepskyblue', label='Ground Truth', marker='o', markersize=10, alpha=0.9,
            linestyle='-', linewidth=2, markerfacecolor='yellow', markeredgewidth=2, markeredgecolor='black')

    # Plot and connect Predicted Points in sequence
    ax.plot(predicted_points[:, 0], predicted_points[:, 1], predicted_points[:, 2],
            color='tomato', label='Predicted', marker='o', markersize=10, alpha=0.9,
            linestyle='-', linewidth=2, markerfacecolor='yellow', markeredgewidth=2, markeredgecolor='black')

    # Labels and Title with a fun font
    ax.set_xlabel('X', fontsize=14, fontweight='bold', color='black')
    ax.set_ylabel('Y', fontsize=14, fontweight='bold', color='black')
    ax.set_zlabel('Z', fontsize=14, fontweight='bold', color='black')
    ax.set_title(title, fontsize=18, fontweight='bold', color='purple')
    ax.legend(fontsize=12, loc='upper left')

    # Adjust background and grid for a cleaner look
    ax.xaxis.pane.fill = False
    ax.yaxis.pane.fill = False
    ax.zaxis.pane.fill = False
    ax.grid(False)

    plt.show()


if __name__ == '__main__':
    input_dir = os.path.join(MAIN_DIR, "PDB", "pdb_data")
    for filename in os.listdir(input_dir):
        if filename.endswith('.json'):
            path = os.path.join(input_dir, filename)
            pdb_df = pd.read_json(path)

            # Process DataFrame
            pdb_df['contact_map'] = pdb_df["coords"].apply(get_contact_map_from_coords)
            pdb_df['distogram'] = pdb_df["coords"].apply(
                lambda coords: get_contact_map_from_coords(coords, soft_map=True))

            # Save processed DataFrame to new file
            output_path = os.path.join(input_dir, filename)
            pdb_df.to_json(path)

