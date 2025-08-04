import os
import pandas as pd
from Bio.PDB import PDBParser, PPBuilder
from tqdm import tqdm

# --- Config ---
SUMMARY_TSV = r"C:\Users\RoyIlani\Desktop\personal\proteins\sabdab_data\sabdab_summary_all.tsv"
RAW_PDB_DIR = r"C:\Users\RoyIlani\Desktop\personal\proteins\sabdab_data\all_structures\raw"
OUT_JSON = "sabdab_data.json"

# --- Init ---
parser = PDBParser(QUIET=True)
ppb = PPBuilder()

df = pd.read_csv(SUMMARY_TSV, sep="\t")
df["pdb"] = df["pdb"].str.lower()

def get_ca_coords(structure, chain_ids):
    coords = []
    for chain in structure[0]:
        if chain.id in chain_ids:
            for res in chain:
                if "CA" in res:
                    coords.append(res["CA"].get_coord().tolist())
    return coords

def get_sequence(chain):
    try:
        peptides = ppb.build_peptides(chain)
        return "".join(str(peptide.get_sequence()) for peptide in peptides)
    except Exception:
        return ""

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

        if not h_chain or not l_chain or h_chain not in model or l_chain not in model:
            continue  # skip incomplete antibodies

        # Sequences
        heavy_seq = get_sequence(model[h_chain])
        light_seq = get_sequence(model[l_chain])
        antigen_seqs = {}
        for chain_id in antigen_chains:
            if chain_id in model:
                antigen_seqs[chain_id] = get_sequence(model[chain_id])

        # Coordinates
        antibody_coords = get_ca_coords(structure, [h_chain, l_chain])
        antigen_coords = get_ca_coords(structure, antigen_chains)

        record = row.to_dict()
        record.update({
            "heavy_sequence": heavy_seq,
            "light_sequence": light_seq,
            "antigen_sequences": antigen_seqs,
            "antibody_coords": antibody_coords,
            "antigen_coords": antigen_coords,
        })
        records.append(record)

    except Exception as e:
        print(f"[WARN] {pdb_id}: {e}")
        continue

# --- Save ---
final_df = pd.DataFrame(records)
final_df.to_json(OUT_JSON, orient="records", lines=True)
print(f"✅ Saved {len(final_df)} complete entries to {OUT_JSON}")