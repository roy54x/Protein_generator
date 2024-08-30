import os

import numpy as np

import pandas as pd
from matplotlib import pyplot as plt
from sklearn.manifold import MDS

from utils.constants import MAIN_DIR, MAX_SIZE


def get_distogram(ca_coords):
    print("number of amino acids to process: " + str(len(ca_coords)))
    if len(ca_coords) <= 1 or len(ca_coords) > MAX_SIZE:
        return np.nan

    ca_coords = np.array(ca_coords, dtype="float16")
    diff = ca_coords[:, np.newaxis, :] - ca_coords[np.newaxis, :, :]
    distances = np.linalg.norm(diff, axis=-1)
    return distances


def get_contact_map(ca_coords, threshold=8.0):
    distances = get_distogram(ca_coords)
    return np.where(distances < threshold, 1.0, 0.0).astype(int)


def get_soft_contact_map(ca_coords, decay_rate=0.5):
    distances = get_distogram(ca_coords)
    return np.exp(-decay_rate * distances)


def optimize_points_from_distogram(distogram, n_init=4, max_iter=300, random_state=None):
    mds = MDS(n_components=3, dissimilarity="precomputed", n_init=n_init, max_iter=max_iter, random_state=random_state)
    points = mds.fit_transform(distogram)
    return points


def align_points(predicted_points, ground_truth_points):
    centroid_pred = np.mean(predicted_points, axis=0)
    centroid_gt = np.mean(ground_truth_points, axis=0)
    pred_centered = predicted_points - centroid_pred
    gt_centered = ground_truth_points - centroid_gt

    # Compute the optimal rotation matrix
    H = np.dot(pred_centered.T, gt_centered)
    U, S, Vt = np.linalg.svd(H)
    R = np.dot(Vt.T, U.T)

    # Check for reflection (det(R) should be 1 for a proper rotation)
    if np.linalg.det(R) < 0:
        Vt[-1, :] *= -1
        R = np.dot(Vt.T, U.T)

    # Apply the rotation matrix to the predicted points
    transformed_predicted_points = np.dot(pred_centered, R)

    # Translate the transformed points
    transformed_predicted_points += centroid_gt

    return transformed_predicted_points


def plot_protein_atoms(predicted_points, ground_truth_points, title="Protein 3D Points"):
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
    input_dir = os.path.join(MAIN_DIR, "PDB", "pdb_data_270000")
    for filename in os.listdir(input_dir):
        if filename.endswith('.json'):
            path = os.path.join(input_dir, filename)
            pdb_df = pd.read_json(path, lines=True)

            # Process DataFrame
            pdb_df['soft_contact_map'] = pdb_df["coords"].apply(get_soft_contact_map)

            # Save processed DataFrame to new file
            output_path = os.path.join(input_dir, filename)
            pdb_df.to_json(path)
