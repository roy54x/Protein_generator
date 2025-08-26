import glob
import shutil

from Bio.SeqUtils.ProtParam import ProteinAnalysis
import os
from pathlib import Path
from colabfold.batch import run as colabfold_run, get_queries
import py3Dmol

from antibody_generators import generate_EvoDiff


def compute_stability(sequence):
    analysis = ProteinAnalysis(sequence)
    return analysis.instability_index()


def compute_aggregation_propensity(sequence):
    hydrophobic_residues = set("AILMFWYV")
    return sum(1 for i in range(len(sequence) - 5)
               if all(residue in hydrophobic_residues for residue in sequence[i:i + 5]))


def fold_and_plot_with_alphafold(sequence, tag="", output_dir="folded_structures"):
    Path(output_dir).mkdir(exist_ok=True)

    # Save sequence with custom tag in FASTA
    fasta_path = os.path.join(output_dir, f"{tag}.fasta")
    with open(fasta_path, "w") as f:
        f.write(f">{tag}\n{sequence}")  # 🟢 Use tag as FASTA name

    # Run ColabFold
    queries, is_complex = get_queries(fasta_path)
    colabfold_run(
        queries, output_dir,
        use_templates=False,
        num_models=1,
        is_complex=is_complex,
        num_recycles=3
    )

    pdb_candidates = glob.glob(os.path.join(output_dir, f"*{tag}*model_1*.pdb"))
    if not pdb_candidates:
        pdb_candidates = glob.glob(os.path.join(output_dir, "*model_1*.pdb"))
        if not pdb_candidates:
            raise FileNotFoundError(f"No model_1 PDB file found for tag '{tag}' in {output_dir}")

    original_pdb_path = pdb_candidates[0]
    tagged_pdb_path = os.path.join(output_dir, f"{tag}_model_1.pdb")
    shutil.copy(original_pdb_path, tagged_pdb_path)

    with open(tagged_pdb_path, 'r') as f:
        pdb_data = f.read()

    view = py3Dmol.view(js='https://3dmol.org/build/3Dmol.js')
    view.addModel(pdb_data, 'pdb')
    view.setStyle({'cartoon': {'colorscheme': {'prop': 'b', 'gradient': 'roygb', 'min': 0, 'max': 100}}})
    view.zoomTo()

    html_path = os.path.join(output_dir, f"{tag}_structure.html")
    with open(html_path, "w") as f:
        f.write(view._make_html())
    print(f"3D structure view saved to: {html_path}")

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
        min_len = min(len(s1), len(s2))
        if min_len == 0:
            return 0.0
        dist = sum(c1 != c2 for c1, c2 in zip(s1[:min_len], s2[:min_len]))
        return dist / min_len  # Normalize by compared length

    if len(sequences) < 2:
        return 0.0
    total_dist, count = 0, 0
    for i in range(len(sequences)):
        for j in range(i + 1, len(sequences)):
            total_dist += hamming(sequences[i], sequences[j])
            count += 1
    return 100 * total_dist / count if count > 0 else 0.0

if __name__ == "__main__":
    print("Generating antibody sequences...")
    n = 100
    min_len = 80
    max_len = 150
    all_sequences = []

    for i in range(n):
        seq = generate_EvoDiff(min_len=min_len, max_len=max_len)
        all_sequences.append(seq)

        print(f"\nGenerated Sequence {i + 1}:\n{seq}")
        print("Stability (Instability Index - Lower than 40 is considered good):", compute_stability(seq))
        print("Aggregation Propensity (Hydrophobic Clusters - The lower the better):",
              compute_aggregation_propensity(seq))
        #avg_plddt = fold_and_plot_with_alphafold(seq, tag=f"antibody_{i}")
        #print("Average pLDDT score - Above 70 is considered good:", avg_plddt)

    print(f"\nDiversity across sequences - above 50 is considered diverse:", compute_diversity(all_sequences))
