import numpy as np
import pandas as pd
import umap.umap_ as umap
from matplotlib import pyplot as plt

from extract_pdb_database import amino_acids


def one_hot_encode_sequence(sequences):
    aa_to_index = {aa: i for i, aa in enumerate(amino_acids)}
    num_amino_acids = len(amino_acids)
    max_len = max([len(s) for s in sequences])

    encoded_sequences = []
    for s in sequences:
        encoding = np.zeros((max_len, num_amino_acids), dtype="float32")
        for i, aa in enumerate(s):
            if aa in aa_to_index:
                encoding[i, aa_to_index[aa]] = 1.0
        encoded_sequences.append(encoding.flatten())
    return encoded_sequences


def encode_values(value_list):
    unique_values = sorted(set(value_list))
    value_to_numeric = {value: i for i, value in enumerate(unique_values)}
    encoded_values = [value_to_numeric[value] for value in value_list]
    return encoded_values, value_to_numeric


protein_df = pd.read_csv("D:\python project\data\protein_df.csv")
top_5_organisms = protein_df['organism'].value_counts().head(5).index.tolist()
protein_df = protein_df[protein_df['organism'].isin(top_5_organisms)][:1000]
encoded_sequences = one_hot_encode_sequence(protein_df.sequence)
encoded_values, value_to_numeric = encode_values(protein_df.organism.to_list())
numeric_to_value = {numeric: value for value, numeric in value_to_numeric.items()}

reducer = umap.UMAP()
embedding = reducer.fit_transform(encoded_sequences)
fig, ax = plt.subplots()

for numeric, value in numeric_to_value.items():
    indices = [i for i, v in enumerate(encoded_values) if v == numeric]
    ax.scatter(embedding[indices, 0], embedding[indices, 1], label=value, cmap='Spectral', s=5)
ax.legend(title='Organism', loc='upper right', fontsize='small')
plt.show()


