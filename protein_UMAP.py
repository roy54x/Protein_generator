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
    padded_sequences = [np.array(s + [0.0] * (max_length - len(s))) for s in sequences]
    return padded_sequences


def encode_values(value_list):
    unique_values = sorted(set(value_list))
    value_to_numeric = {value: i for i, value in enumerate(unique_values)}
    encoded_values = [value_to_numeric[value] for value in value_list]
    return encoded_values, value_to_numeric


protein_df = pd.read_csv("protein_df.csv")
top_5_organisms = protein_df['organism'].value_counts().head(5).index.tolist()
protein_df = protein_df[protein_df['organism'].isin(top_5_organisms)]
encoded_sequences = encode_sequences(protein_df.sequence)
padded_sequences = pad_sequences(encoded_sequences)

encoded_values, value_to_numeric = encode_values(protein_df.organism.to_list())
reducer = umap.UMAP(n_neighbors=15)
embedding = reducer.fit_transform(padded_sequences)
fig, ax = plt.subplots()
sc = ax.scatter(embedding[:, 0], embedding[:, 1], c=encoded_values, cmap='Spectral', lw=0, alpha=1, s=5)

numeric_to_value = {numeric: value for value, numeric in value_to_numeric.items()}
legend_labels = [numeric_to_value[numeric] for numeric in sorted(value_to_numeric.values())]
legend = ax.legend(legend_labels, title='Organism', loc='upper right', fontsize='small')
plt.setp(legend.get_title(), fontsize='medium')
plt.show()
