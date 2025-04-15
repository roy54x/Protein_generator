import blosum as bl
import numpy as np
import scipy.special as sp


def normalize(x, epsilon=1e-2):
    scale = np.ptp(x)
    if scale > epsilon:
        return (x - x.min()) / scale
    else:
        return np.zeros_like(x)


def get_blosum_probability_function(temp=1.0):
    """Returns a function that takes two amino acids and returns their normalized BLOSUM62 probability (0-1)."""
    # Load BLOSUM62 matrix
    matrix = bl.BLOSUM(62)

    # Get ordered list of amino acids
    amino_acids = sorted(matrix.keys())
    aa_index = {aa: i for i, aa in enumerate(amino_acids)}

    # Create score matrix
    scores = np.array([[matrix[a][b] for b in amino_acids] for a in amino_acids])
    scaled_scores = scores / temp

    # Apply softmax row-wise to each row in the scores matrix
    probabilities = np.apply_along_axis(lambda x: sp.softmax(x), axis=1, arr=scaled_scores)

    # Create lookup function with validation
    def get_prob(aa1, aa2):
        if aa1 not in aa_index or aa2 not in aa_index:
            valid_aas = ", ".join(aa_index.keys())
            raise ValueError(f"Invalid amino acid. Use: {valid_aas}")

        i = aa_index[aa1]
        j = aa_index[aa2]
        return float(probabilities[i, j])

    return get_prob