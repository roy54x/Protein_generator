import numpy as np
import pandas as pd
import umap.umap_ as umap
from matplotlib import pyplot as plt


def one_hot_encode_sequence(sequence):
    # Define the mapping of amino acids to indices
    amino_acids = 'ACDEFGHIKLMNPQRSTVWY'
    aa_to_index = {aa: i for i, aa in enumerate(amino_acids)}

    # Initialize an array of zeros for one-hot encoding
    num_amino_acids = len(amino_acids)
    encoding = np.zeros((len(sequence), num_amino_acids))

    # Set the appropriate index to 1 for each amino acid in the sequence
    for i, aa in enumerate(sequence):
        if aa in aa_to_index:
            encoding[i, aa_to_index[aa]] = 1
        else:
            # Handle unknown or ambiguous amino acids
            encoding[i, :] = np.nan

    return encoding


def tokenize_sequences_one_hot(sequences):
    tokens = []
    for sequence in sequences:
        # One-hot encode each sequence
        encoded_sequence = one_hot_encode_sequence(sequence)
        tokens.append(encoded_sequence)
    return tokens


protein_df = pd.read_csv("protein_df.csv")
sequences = protein_df.sequence
tokens = tokenize_sequences_one_hot(sequences)
umap_colors = protein_df.organism
reducer = umap.UMAP(n_neighbors=15)
embedding = reducer.fit_transform(tokens)
fig, ax = plt.subplots()
sc = ax.scatter(embedding[:, 0], embedding[:, 1], c=umap_colors, cmap='Spectral', lw=0, alpha=1, s=5)
