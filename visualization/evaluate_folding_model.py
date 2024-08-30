import torch
import json
import matplotlib.pyplot as plt
import numpy as np

from strategies.sequence_to_contact_map import SequenceToContactMap
from utils.structure_utils import optimize_points_from_distogram, align_points, plot_protein_atoms


def load_json_data(json_path, pdb_id, chain_id):
    with open(json_path, 'r') as f:
        df = json.load(f)
    return df.loc[(df['pdb_id'] == pdb_id) & (df['chain_id'] == chain_id)]


def plot_distograms(predicted_distogram, ground_truth_distogram):
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))

    axes[0].imshow(predicted_distogram, cmap='viridis')
    axes[0].set_title("Predicted Distogram")

    axes[1].imshow(ground_truth_distogram, cmap='viridis')
    axes[1].set_title("Ground Truth Distogram")

    plt.show()


if __name__ == "__main__":
    json_path = "D:\python project\data\Proteins\PDB\pdb_data_130000\pdb_df_9.json"
    pdb_id = "6VMH"
    chain_id = "B"
    model_path = ""

    data = load_json_data(json_path, pdb_id, chain_id)
    ground_truth_coords = np.array(data["coords"], dtype="float16")
    (x_tensor, mask_tensor), ground_truth_contact_map = SequenceToContactMap().load_inputs_and_ground_truth(data)
    model = torch.load(model_path, map_location=torch.device('cpu'))
    predicted_contact_map = model((x_tensor, mask_tensor))

    # Plot the predicted distogram and the ground truth distogram
    plot_distograms(predicted_contact_map, ground_truth_contact_map)

    # Plot the aligned predicted coordinates with the ground truth coordinates
    predicted_coords = optimize_points_from_distogram(predicted_contact_map)
    aligned_predicted_coords = align_points(predicted_coords, ground_truth_coords)
    plot_protein_atoms(aligned_predicted_coords, ground_truth_coords)
