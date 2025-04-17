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
    """Returns:
    - `get_prob_vec(aa)` to return the full probability vector of swapping `aa` with others.
    - `aa_list`: list of amino acids corresponding to indices in the vector.
    """
    # Load BLOSUM62 matrix
    matrix = bl.BLOSUM(62)

    # Sorted amino acids list
    amino_acids = list(matrix.keys())
    aa_index = {aa: i for i, aa in enumerate(amino_acids)}

    # Compute score matrix and softmax
    scores = np.array([[matrix[a][b] for b in amino_acids] for a in amino_acids])
    scaled_scores = scores / temp
    probabilities = sp.softmax(scaled_scores, axis=1)

    # Function to return the full vector of probabilities
    def get_prob_vec(aa):
        if aa not in aa_index:
            raise ValueError(f"Invalid amino acid: {aa}")
        return probabilities[aa_index[aa]]  # Shape: (20,)

    return get_prob_vec, amino_acids