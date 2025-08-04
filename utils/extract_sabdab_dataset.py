import os
import pandas as pd
from Bio.PDB import PDBParser
from tqdm import tqdm

# --- Config ---
PDB_DIR = r"C:\Users\RoyIlani\Desktop\personal\proteins\sabdab_data\all_structures\raw"
OUT_JSON = "sabdab_structures.json"

# --- Init ---
parser = PDBParser(QUIET=True)
all_data = []

# --- Helpers ---
def get_ca_coords(chain):
    coords = []
    for res in chain:
        if 'CA' in res:
            coords.append(res['CA'].get_coord())
    return coords

def get_chain_lengths(structure):
    return {chain.id: sum(1 for res in chain if 'CA' in res) for chain in structure[0]}

# --- Iterate over PDB files ---
for filename in tqdm(os.listdir(PDB_DIR)):
    if not filename.endswith(".pdb"):
        continue

    pdb_path = os.path.join(PDB_DIR, filename)
    pdb_id = filename.replace(".pdb", "").lower()

    try:
        structure = parser.get_structure(pdb_id, pdb_path)
        model = structure[0]

        all_data.append({
            "pdb": pdb_id
        })

    except Exception as e:
        print(f"[WARN] Failed {pdb_id}: {e}")
        continue

# --- Save to JSON ---
df = pd.DataFrame(all_data)
df.to_json(OUT_JSON, orient="records", lines=True)
print(f"✅ Saved {len(df)} entries to {OUT_JSON}")