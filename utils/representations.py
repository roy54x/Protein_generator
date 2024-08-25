import numpy as np
import math

import pandas as pd


def get_contact_map_from_coords(ca_coords, threshold=8.0, soft_map=True):
    ca_coords = np.array(ca_coords)
    num_residues = len(ca_coords)
    contact_map = np.zeros((num_residues, num_residues))
    for i in range(num_residues):
        for j in range(i + 1, num_residues):
            distance = np.linalg.norm(ca_coords[i] - ca_coords[j])
            if soft_map:
                value = distance_to_score(distance)
            else:
                value = 1
            if distance < threshold:
                contact_map[i, j] = value
                contact_map[j, i] = value
    return contact_map


def distance_to_score(distance, decay_rate=0.5):
    if distance < 0:
        raise ValueError("Distance cannot be negative")
    score = math.exp(-decay_rate * distance)
    return score


if __name__ == '__main__':
    path = r"D:\python project\data\Proteins\PDB\pdb_df_100.json"
    pdb_df = pd.read_json(path)
    pdb_df['contact_map'] = pdb_df["coords"].apply(get_contact_map_from_coords)
    pdb_df.to_json(path)
