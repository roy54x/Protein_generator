import pandas as pd
import torch
import json
import matplotlib.pyplot as plt
import numpy as np

from strategies.sequence_to_contact_map import SequenceToContactMap
from utils.structure_utils import optimize_points_from_distogram, align_points, plot_protein_atoms, \
    get_distogram_from_soft_contact_map


def plot_distograms(predicted_distogram, ground_truth_distogram):
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))

    axes[0].imshow(predicted_distogram, cmap='viridis')
    axes[0].set_title("Predicted Distogram")

    axes[1].imshow(ground_truth_distogram, cmap='viridis')
    axes[1].set_title("Ground Truth Distogram")

    plt.show()


if __name__ == "__main__":
    strategy = SequenceToContactMap()
    model_path = r"D:\python project\data\Proteins\models\SequenceToContactMap\20240830\best_model.pth"
    state_dict = torch.load(model_path)
    strategy.load_state_dict(state_dict)
    strategy.eval()

    data_path = "D:\python project\data\Proteins\PDB\pdb_data_130000\pdb_df_9.json"
    pdb_df = pd.read_json(data_path, lines=True)
    pdb_id = "6VMH"
    chain_id = "B"
    data = pdb_df.loc[(pdb_df['pdb_id'] == pdb_id) & (pdb_df['chain_id'] == chain_id)].iloc[0]
    ground_truth_coords = np.array(data["coords"], dtype="float16")
    seq_len = len(data["sequence"])
    (x_tensor, mask_tensor), ground_truth_contact_map = strategy.load_inputs_and_ground_truth(data)
    ground_truth_contact_map = ground_truth_contact_map[: seq_len, :seq_len]
    predicted_contact_map = strategy((x_tensor.unsqueeze(0), mask_tensor.unsqueeze(0)))
    predicted_contact_map = predicted_contact_map.squeeze().detach().numpy()
    predicted_contact_map = predicted_contact_map[: seq_len, :seq_len]

    # Plot the predicted distogram and the ground truth distogram
    plot_distograms(predicted_contact_map, ground_truth_contact_map)

    # Plot the aligned predicted coordinates with the ground truth coordinates
    distogram = get_distogram_from_soft_contact_map(predicted_contact_map)
    predicted_coords = optimize_points_from_distogram(distogram)
    aligned_predicted_coords = align_points(predicted_coords, ground_truth_coords)
    plot_protein_atoms(aligned_predicted_coords, ground_truth_coords)
