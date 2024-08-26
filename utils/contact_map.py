import numpy as np
import math

import pandas as pd


def get_contact_map_from_coords(ca_coords, soft_map=False, threshold=8.0, decay_rate=0.5):
    if len(ca_coords) <= 1:
        return np.array([1])

    ca_coords = np.array(ca_coords, dtype="float32")
    diff = ca_coords[:, np.newaxis, :] - ca_coords[np.newaxis, :, :]
    distances = np.linalg.norm(diff, axis=-1)

    if soft_map:
        contact_map = np.exp(-decay_rate * distances)
    else:
        contact_map = np.where(distances < threshold, 1.0, 0.0)

    return contact_map


if __name__ == '__main__':
    path = r"D:\python project\data\Proteins\PDB\pdb_df_150000.json"
    pdb_df = pd.read_json(path)
    pdb_df['contact_map'] = pdb_df["coords"].apply(get_contact_map_from_coords)
    pdb_df['distogram'] = pdb_df["coords"].apply(lambda coords: get_contact_map_from_coords(coords, soft_map=True))
    pdb_df.to_json(path)
