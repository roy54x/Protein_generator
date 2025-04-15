from Bio import SeqIO
import json

from Bio import SeqIO
import json
import os

from constants import NUM_SAMPLES_IN_DATAFRAME

# Paths
fasta_path = r"C:\Users\RoyIlani\Desktop\personal\proteins\absd_data\Homo_sapiens.fasta"
output_dir = r"C:\Users\RoyIlani\Desktop\personal\proteins\absd_data\data"

# Make sure output directory exists
os.makedirs(output_dir, exist_ok=True)

# Read and chunk
chunk = []
chunk_idx = 0
total_sequences = 0

for record in SeqIO.parse(fasta_path, "fasta"):
    chunk.append({
        "id": record.id,
        "name": record.name,
        "description": record.description,
        "sequence": str(record.seq)
    })
    total_sequences += 1

    if len(chunk) == NUM_SAMPLES_IN_DATAFRAME:
        chunk_path = os.path.join(output_dir, f"chunk_{chunk_idx:04d}.json")
        with open(chunk_path, "w") as f:
            json.dump(chunk, f, indent=2)
        print(f"Saved chunk {chunk_idx} with {len(chunk)} sequences to {chunk_path}")
        chunk = []
        chunk_idx += 1

# Save any remaining sequences
if chunk:
    chunk_path = os.path.join(output_dir, f"chunk_{chunk_idx:04d}.json")
    with open(chunk_path, "w") as f:
        json.dump(chunk, f, indent=2)
    print(f"Saved chunk {chunk_idx} with {len(chunk)} sequences to {chunk_path}")

print(f"Total sequences saved: {total_sequences}")