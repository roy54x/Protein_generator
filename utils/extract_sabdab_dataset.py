import os

import numpy as np
import pandas as pd
from Bio.PDB import PDBParser, PPBuilder, NeighborSearch
from tqdm import tqdm

# --- Config ---
SUMMARY_TSV = r"C:\Users\RoyIlani\Desktop\personal\proteins\sabdab_data\sabdab_summary_all.tsv"
RAW_PDB_DIR = r"C:\Users\RoyIlani\Desktop\personal\proteins\sabdab_data\all_structures\raw"
OUT_JSON = "sabdab_data.json"

# --- Init ---
parser = PDBParser(QUIET=True)
ppb = PPBuilder()

df = pd.read_csv(SUMMARY_TSV, sep="\t")
df.columns = df.columns.str.strip()
df["pdb"] = df["pdb"].str.lower()

# --- Helper: extract Cα coordinates ---
def get_ca_coords_per_chain(structure, chain_ids):
    chain_coords = {}
    for chain in structure[0]:
        if chain.id in chain_ids:
            coords = [res["CA"].get_coord().tolist() for res in chain if "CA" in res]
            chain_coords[chain.id] = coords
    return chain_coords

# --- Helper: extract sequence ---
def get_sequence(chain):
    try:
        peptides = ppb.build_peptides(chain)
        return "".join(str(peptide.get_sequence()) for peptide in peptides)
    except Exception:
        return ""

# --- Helper: find antibody-antigen contacts ---
def find_contacts_from_coords(antibody_coords_dict, antigen_coords_dict, cutoff=5.0):
    contacts = []

    for ab_chain, ab_coords in antibody_coords_dict.items():
        ab_coords = np.array(ab_coords)
        for ag_chain, ag_coords in antigen_coords_dict.items():
            ag_coords = np.array(ag_coords)

            # Compute pairwise distances: (n_ab, n_ag)
            dists = np.linalg.norm(ab_coords[:, None, :] - ag_coords[None, :, :], axis=-1)

            # Find index pairs where dist <= cutoff
            ab_idx, ag_idx = np.where(dists <= cutoff)
            for i, j in zip(ab_idx, ag_idx):
                contacts.append({
                    "antibody_chain": ab_chain,
                    "antibody_residue_index": i,
                    "antigen_chain": ag_chain,
                    "antigen_residue_index": j
                })

    return contacts

# --- Main loop ---
records = []

for _, row in tqdm(df.iterrows(), total=len(df)):
    pdb_id = row["pdb"]
    pdb_path = os.path.join(RAW_PDB_DIR, f"{pdb_id}.pdb")

    if not os.path.isfile(pdb_path):
        continue

    try:
        structure = parser.get_structure(pdb_id, pdb_path)
        model = structure[0]

        # Chain IDs
        h_chain = str(row["Hchain"]).strip() if pd.notna(row["Hchain"]) and row["Hchain"] != "NA" else None
        l_chain = str(row["Lchain"]).strip() if pd.notna(row["Lchain"]) and row["Lchain"] != "NA" else None
        antigen_chains = []
        if pd.notna(row["antigen_chain"]) and row["antigen_chain"] != "NA":
            antigen_chains = [c.strip() for c in row["antigen_chain"].replace("|", ",").split(",")]

        if not h_chain or not l_chain or h_chain not in model or l_chain not in model or len(antigen_chains) == 0:
            continue  # skip incomplete antibodies

        # Sequences
        antibody_seqs = {}
        for chain_id in [h_chain, l_chain]:
            if chain_id and chain_id in model:
                antibody_seqs[chain_id] = get_sequence(model[chain_id])

        antigen_seqs = {
            chain_id: get_sequence(model[chain_id])
            for chain_id in antigen_chains
            if chain_id in model
        }

        # Coordinates
        antibody_coords = get_ca_coords_per_chain(structure, [h_chain, l_chain])
        antigen_coords = get_ca_coords_per_chain(structure, antigen_chains)

        # Contacts
        binding_contacts = find_contacts_from_coords(antibody_coords, antigen_coords)

        record = row.to_dict()
        record.update({
            "antibody_sequences": antibody_seqs,
            "antigen_sequences": antigen_seqs,
            "antibody_coords": antibody_coords,
            "antigen_coords": antigen_coords,
            "binding_contacts": binding_contacts
        })
        records.append(record)

    except Exception as e:
        print(f"[WARN] {pdb_id}: {e}")
        continue

# --- Save ---
final_df = pd.DataFrame(records)
final_df.to_json(OUT_JSON, orient="records", lines=True)
print(f"✅ Saved {len(final_df)} complete entries with contacts to {OUT_JSON}")