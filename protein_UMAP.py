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


def pad_sequences(sequences):
    max_length = max(len(s) for s in sequences)
    padded_sequences = [np.array(s + [-1.0] * (max_length - len(s))) for s in sequences]
    return padded_sequences


def encode_values(value_list):
    unique_values = sorted(set(value_list))
    value_to_numeric = {value: i for i, value in enumerate(unique_values)}
    encoded_values = [value_to_numeric[value] for value in value_list]
    return encoded_values, value_to_numeric


protein_df = pd.read_csv("protein_df.csv")
top_5_organisms = protein_df['organism'].value_counts().head(5).index.tolist()
protein_df = protein_df[protein_df['organism'].isin(top_5_organisms)][:2000]
encoded_sequences = encode_sequences(protein_df.sequence)
padded_sequences = pad_sequences(encoded_sequences)
encoded_values, value_to_numeric = encode_values(protein_df.organism.to_list())
numeric_to_value = {numeric: value for value, numeric in value_to_numeric.items()}

reducer = umap.UMAP()
embedding = reducer.fit_transform(padded_sequences)
fig, ax = plt.subplots()

for numeric, value in numeric_to_value.items():
    indices = [i for i, v in enumerate(encoded_values) if v == numeric]
    ax.scatter(embedding[indices, 0], embedding[indices, 1], label=value, cmap='Spectral', s=5)
ax.legend(title='Organism', loc='upper right', fontsize='small')
plt.show()


