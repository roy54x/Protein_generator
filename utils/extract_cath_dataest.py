import os
import pdb

import esm
import requests
import json
import pandas as pd
from biotite.structure import filter_backbone, get_chains
from biotite.structure.io import pdbx
from esm.inverse_folding.util import load_structure, extract_coords_from_structure

from constants import MAIN_DIR


def download_file(url, save_path):
    """
    Download a file from a given URL and save it to the specified path.
    """
    response = requests.get(url)
    if response.status_code == 200:
        with open(save_path, "wb") as f:
            f.write(response.content)
        print(f"File downloaded successfully: {save_path}")
    else:
        print(f"Failed to download file from {url}. HTTP status code: {response.status_code}")


def process_cath_data(cath_file, splits):
    """
    Process the CATH data file and return a DataFrame with sequences, coordinates, and dataset information.
    """
    rows = []
    with (open(cath_file, "r") as f):
        for line in f:
            entry = json.loads(line)
            chain_id = entry["name"]
            sequence = entry["seq"]
            coords = entry["coords"]

            # Convert coordinates into the desired format: list of lists of lists [residue][atom][x, y, z]
            processed_coords = []
            for residue_idx in range(len(coords["CA"])):
                residue_coords = []
                for atom in ['N', 'CA', 'C', "O"]:  # For each atom in the residue
                    atom_coords = coords[atom][residue_idx]
                    residue_coords.append(atom_coords)
                processed_coords.append(residue_coords)

            dataset = None
            for split, ids in splits.items():
                if chain_id in ids:
                    dataset = split
                    break

            if dataset:
                rows.append({
                    "chain_id": chain_id,
                    "sequence": sequence,
                    "coords": processed_coords,
                    "dataset": dataset
                })
    return pd.DataFrame(rows)


if __name__ == "__main__":
    # Define URLs and file paths
    cath_url = "https://people.csail.mit.edu/ingraham/graph-protein-design/data/cath/chain_set.jsonl"
    splits_url = "https://people.csail.mit.edu/ingraham/graph-protein-design/data/cath/chain_set_splits.json"
    cath_file = os.path.join(MAIN_DIR, "cath_data", "chain_set.jsonl")
    splits_file = os.path.join(MAIN_DIR, "cath_data", "chain_set_splits.json")
    output_dir = os.path.join(MAIN_DIR, "cath_data")

    # Step 1: Download the data
    if not os.path.exists(cath_file):
        download_file(cath_url, cath_file)
    if not os.path.exists(splits_file):
        download_file(splits_url, splits_file)

    with open(splits_file, "r") as f:
        splits = json.load(f)

    cath_df = process_cath_data(cath_file, splits)
    output_path = os.path.join(output_dir, "cath_data.json")
    cath_df.to_json(output_path)
    print(f"Data saved to {output_path}")