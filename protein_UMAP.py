import numpy as np
import pandas as pd
import umap.umap_ as umap
from lxml import etree
from matplotlib import pyplot as plt


def parse_uniprot_xml(xml_file):
    data = []
    context = etree.iterparse(xml_file, events=('end',), tag='{http://uniprot.org/uniprot}entry')
    for event, elem in context:
        accession = elem.findtext('{http://uniprot.org/uniprot}accession')
        sequence = elem.findtext('{http://uniprot.org/uniprot}sequence')
        protein_names = [name.text for name in elem.findall('.//{http://uniprot.org/uniprot}fullName')]
        organism = elem.findtext('.//{http://uniprot.org/uniprot}organism/{http://uniprot.org/uniprot}name')
        function = elem.findtext('.//{http://uniprot.org/uniprot}comment[@type="function"]/{http://uniprot.org/uniprot}text')
        subcellular_location = elem.findtext('.//{http://uniprot.org/uniprot}comment[@type="subcellular location"]/{http://uniprot.org/uniprot}subcellularLocation/{http://uniprot.org/uniprot}location')
        tissue_specificity = elem.findtext('.//{http://uniprot.org/uniprot}comment[@type="tissue specificity"]/{http://uniprot.org/uniprot}text')
        domain_structure = [domain.get('description') for domain in
                            elem.findall('.//{http://uniprot.org/uniprot}feature[@type="domain"]')]
        ptms = [ptm.get('description') for ptm in
                elem.findall('.//{http://uniprot.org/uniprot}feature[@type="modified residue"]')]
        interactions = [interaction.text for interaction in
                        elem.findall('.//{http://uniprot.org/uniprot}interactant/{http://uniprot.org/uniprot}geneName')]
        sequence_annotations = [(annot.get('description'), annot.get('evidence')) for annot in
                                elem.findall('.//{http://uniprot.org/uniprot}feature')]

        data.append({
            'accession': accession,
            'sequence': sequence,
            'protein_names': protein_names,
            'organism': organism,
            'function': function,
            'subcellular_location': subcellular_location,
            'tissue_specificity': tissue_specificity,
            'domain_structure': domain_structure,
            'ptms': ptms,
            'interactions': interactions,
            'sequence_annotations': sequence_annotations,
        })
        elem.clear()
        while elem.getprevious() is not None:
            del elem.getparent()[0]

    return pd.DataFrame(data)


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


xml_file_path = r'C:\Users\RoyIlani\Downloads\uniprot_sprot.xml'
protein_df = parse_uniprot_xml(xml_file_path)
sequences = protein_df.sequence
tokens = tokenize_sequences_one_hot(sequences)
umap_colors = None
reducer = umap.UMAP(n_neighbors=15)
embedding = reducer.fit_transform(tokens)
fig, ax = plt.subplots()
sc = ax.scatter(embedding[:, 0], embedding[:, 1], c=umap_colors, cmap='Spectral', lw=0, alpha=1, s=5)
