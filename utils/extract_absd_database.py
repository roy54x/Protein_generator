from Bio import SeqIO
import json

# Paths
fasta_path = r"C:\Users\RoyIlani\Desktop\personal\proteins\Homo_sapiens.fasta"
output_json = r"C:\Users\RoyIlani\Desktop\personal\proteins\Homo_sapiens.json"

# Parse and convert each record to a dict
sequences = []
for record in SeqIO.parse(fasta_path, "fasta"):
    sequences.append({
        "id": record.id,
        "name": record.name,
        "description": record.description,
        "sequence": str(record.seq)
    })

# Save to JSON
with open(output_json, "w") as f:
    json.dump(sequences, f, indent=2)

print(f"Saved {len(sequences)} sequences to {output_json}")