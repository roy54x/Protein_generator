import torch
import json
import matplotlib.pyplot as plt
import numpy as np
from utils.structure_utils import optimize_points_from_distogram, align_points, plot_protein_atoms


def load_json_data(json_path, id):
    with open(json_path, 'r') as f:
        data = json.load(f)
    return data[str(id)]


def plot_distograms(predicted_distogram, ground_truth_distogram):
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))

    axes[0].imshow(predicted_distogram, cmap='viridis')
    axes[0].set_title("Predicted Distogram")

    axes[1].imshow(ground_truth_distogram, cmap='viridis')
    axes[1].set_title("Ground Truth Distogram")

    plt.show()


if __name__ == "__main__":

    json_path = ""
    pdb_id = ""
    model_path = ""
    data = load_json_data(json_path, id)
    sequence = data['sequence']
    ground_truth_contact_map = np.array(data['soft_contact_map'])
    ground_truth_coords = np.array(data['coords'])

    model = torch.load(model_path, map_location=torch.device('cpu'))
    predicted_contact_map = model(sequence)

    # Plot the predicted distogram and the ground truth distogram
    plot_distograms(predicted_contact_map, ground_truth_contact_map)

    # Plot the aligned predicted coordinates with the ground truth coordinates
    predicted_coords = optimize_points_from_distogram(predicted_contact_map)
    aligned_predicted_coords = align_points(predicted_coords, ground_truth_coords)
    plot_protein_atoms(aligned_predicted_coords, ground_truth_coords)
