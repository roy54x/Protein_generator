import numpy as np


def read_fasta_file(file_path):
    sequences = []
    with open(file_path, 'r') as file:
        sequence = ''
        for line in file:
            if line.startswith('>'):
                # If a new sequence header is encountered, store the previous sequence
                if sequence:
                    sequences.append(sequence)
                    sequence = ''
            else:
                # Concatenate lines to form the sequence
                sequence += line.strip()
        # Store the last sequence
        if sequence:
            sequences.append(sequence)
    return sequences


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


# Example usage
file_path = 'uniprot_sprot.fasta'
sequences = read_fasta_file(file_path)
tokens = tokenize_sequences_one_hot(sequences)

# Print the first 100 tokens
print(tokens[:100])