from Bio.SeqUtils.ProtParam import ProteinAnalysis
import os
from pathlib import Path
from colabfold.batch import run as colabfold_run, get_queries
import py3Dmol

from antibody_generators import generate_single_sequence


def compute_stability(sequence):
    analysis = ProteinAnalysis(sequence)
    return analysis.instability_index()


def compute_aggregation_propensity(sequence):
    hydrophobic_residues = set("AILMFWYV")
    return sum(1 for i in range(len(sequence) - 5)
               if all(residue in hydrophobic_residues for residue in sequence[i:i + 5]))


def fold_and_plot_with_alphafold(sequence, tag="antibody", output_dir="folded_structures"):
    Path(output_dir).mkdir(exist_ok=True)
    fasta_path = os.path.join(output_dir, f"{tag}.fasta")
    with open(fasta_path, "w") as f:
        f.write(f">seq\n{sequence}")

    queries, is_complex = get_queries(fasta_path)
    colabfold_run(queries, output_dir, use_templates=False,
                  num_models=1, is_complex=is_complex, num_recycles=3)

    pdb_path = os.path.join(output_dir, f"{tag}_model_1.pdb")

    # Visualize using py3Dmol
    with open(pdb_path, 'r') as f:
        pdb_data = f.read()

    view = py3Dmol.view(js='https://3dmol.org/build/3Dmol.js')
    view.addModel(pdb_data, 'pdb')
    view.setStyle({'cartoon': {'colorscheme': {'prop': 'b', 'gradient': 'roygb', 'min': 0.5, 'max': 0.9}}})
    view.zoomTo()
    view.show()

    # Extract average pLDDT score from B-factor column
    plDDTs = []
    for line in pdb_data.splitlines():
        if line.startswith("ATOM"):
            try:
                plDDT = float(line[60:66].strip())
                plDDTs.append(plDDT)
            except ValueError:
                continue
    return sum(plDDTs) / len(plDDTs) if plDDTs else None


def compute_diversity(sequences):
    def hamming(s1, s2):
        return sum(c1 != c2 for c1, c2 in zip(s1, s2))

    if len(sequences) < 2:
        return 0.0
    total_dist, count = 0, 0
    for i in range(len(sequences)):
        for j in range(i + 1, len(sequences)):
            total_dist += hamming(sequences[i], sequences[j])
            count += 1
    return total_dist / count

if __name__ == "__main__":
    print("Generating antibody sequences...")
    n = 5
    seq_len = 120
    all_sequences = []

    for i in range(n):
        seq = generate_single_sequence(seq_len=seq_len)
        all_sequences.append(seq)

        print(f"\nGenerated Sequence {i + 1}:\n{seq}")
        print("Stability (Instability Index - Lower than 40 is considered good):", compute_stability(seq))
        print("Aggregation Propensity (Hydrophobic Clusters - The lower the better):",
              compute_aggregation_propensity(seq))
        avg_plddt = fold_and_plot_with_alphafold(seq, tag=f"antibody_{i}")
        print("Average pLDDT score:", avg_plddt)

    print("\nDiversity across sequences:", compute_diversity(all_sequences))
