import numpy as np
import pandas as pd
import torch

from strategies.sequence_to_distogram import SequenceToDistogram
from utils.structure_utils import optimize_points_from_distogram, align_points, plot_protein_atoms, \
    plot_contact_map

strategy = SequenceToDistogram()
model_path = r"C:\Users\RoyIlani\Desktop\proteins\models\SequenceToDistogram\20240831\best_model.pth"
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
(x_tensor, mask_tensor), ground_truth_distogram = strategy.load_inputs_and_ground_truth(
    data, normalize_distogram=False)
ground_truth_distogram = ground_truth_distogram[: seq_len, :seq_len]

# Get model prediction
predicted_distogram = strategy((x_tensor.unsqueeze(0), mask_tensor.unsqueeze(0)))[0]
predicted_distogram = predicted_distogram.squeeze().detach().numpy()
predicted_distogram = predicted_distogram[: seq_len, :seq_len]
predicted_distogram = (predicted_distogram + predicted_distogram.T) / 2
predicted_distogram *= ground_truth_distogram.max().numpy()

# Plot the predicted distogram and the ground truth distogram
plot_contact_map(predicted_distogram, ground_truth_distogram)

# Plot the aligned predicted coordinates with the ground truth coordinates
predicted_coords = optimize_points_from_distogram(predicted_distogram)
aligned_predicted_coords = align_points(predicted_coords, ground_truth_coords)
plot_protein_atoms(aligned_predicted_coords, ground_truth_coords)
