import numpy as np
import pandas as pd
import umap.umap_ as umap
from matplotlib import pyplot as plt


def encode_sequences(sequences):
    amino_acids = 'ACDEFGHIKLMNPQRSTVWY'
    encoding = {aa: i / len(amino_acids) for i, aa in enumerate(amino_acids)}
    encoded_sequences = []
    for sequence in sequences:
        encoded_sequence = [encoding.get(aa, 0.0) for aa in sequence]
        encoded_sequences.append(encoded_sequence)
    return encoded_sequences


def pad_vectors(vectors):
    max_length = max(len(v) for v in vectors)
    padded_vectors = [v + [0.0] * (max_length - len(v)) for v in vectors]
    return padded_vectors


protein_df = pd.read_csv("protein_df.csv")
encoded_sequences = encode_sequences(protein_df.sequence)
padded_vectors = pad_vectors(encoded_sequences)
umap_colors = protein_df.organism
reducer = umap.UMAP(n_neighbors=15)
embedding = reducer.fit_transform(padded_vectors)
fig, ax = plt.subplots()
sc = ax.scatter(embedding[:, 0], embedding[:, 1], c=umap_colors, cmap='Spectral', lw=0, alpha=1, s=5)
