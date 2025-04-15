import numpy as np
import torch
import torch.nn as nn
import blosum as bl
import scipy.special as sp
from matplotlib import pyplot as plt
import seaborn as sns

from constants import MAX_TRAINING_SIZE, AMINO_ACID_TO_INDEX, PAD_IDX, INDEX_TO_AMINO_ACID
from strategies.base import Base


def get_blosum_probability_function():
    """Returns a function that takes two amino acids and returns their normalized BLOSUM62 probability (0-1)."""
    # Load BLOSUM62 matrix
    matrix = bl.BLOSUM(62)

    # Get ordered list of amino acids
    amino_acids = sorted(matrix.keys())
    aa_index = {aa: i for i, aa in enumerate(amino_acids)}

    # Create score matrix
    scores = np.array([[matrix[a][b] for b in amino_acids] for a in amino_acids])

    # Apply softmax row-wise to each row in the scores matrix
    probabilities = np.apply_along_axis(lambda x: sp.softmax(x), axis=1, arr=scores)

    # Create lookup function with validation
    def get_prob(aa1, aa2):
        if aa1 not in aa_index or aa2 not in aa_index:
            valid_aas = ", ".join(aa_index.keys())
            raise ValueError(f"Invalid amino acid. Use: {valid_aas}")

        i = aa_index[aa1]
        j = aa_index[aa2]
        return float(probabilities[i, j])

    return get_prob


blosum_probs = get_blosum_probability_function()


class RobertaBlock(nn.Module):
    def __init__(self, vocab_size, hidden_dim=128, num_layers=2, num_heads=4):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, hidden_dim, padding_idx=PAD_IDX)
        encoder_layer = nn.TransformerEncoderLayer(d_model=hidden_dim, nhead=num_heads, batch_first=True)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.output = nn.Linear(hidden_dim, vocab_size)

    def forward(self, x):
        x = self.embedding(x)
        x = self.transformer(x)
        return self.output(x)


class SequenceDiffusion(Base):
    def __init__(self):
        super().__init__()
        self.device = "cuda:0" if torch.cuda.is_available() else "cpu"
        self.vocab_size = len(AMINO_ACID_TO_INDEX)  # Size based on AMINO_ACID_TO_INDEX
        self.model = RobertaBlock(self.vocab_size).to(self.device)
        self.loss_fn = nn.CrossEntropyLoss(ignore_index=PAD_IDX)

    def pad_sequence(self, sequence):
        tokenized = [AMINO_ACID_TO_INDEX.get(aa, PAD_IDX) for aa in sequence]
        padded = tokenized[:MAX_TRAINING_SIZE] + [PAD_IDX] * (MAX_TRAINING_SIZE - len(tokenized))
        return padded

    def add_noise(self, sequence):
        sequence = sequence.copy()
        valid_len = len([aa for aa in sequence if aa != PAD_IDX])

        if valid_len == 0:
            return sequence

        idx = np.random.randint(valid_len)
        aa_idx = sequence[idx]
        aa = INDEX_TO_AMINO_ACID.get(aa_idx, None)

        if aa and aa in blosum_probs:
            aas, probs = blosum_probs[aa]
            new_aa = np.random.choice(aas, p=probs)
            sequence[idx] = AMINO_ACID_TO_INDEX.get(new_aa, aa_idx)

        # Optional: insertion/deletion for padding effect
        # (same as before...)

        return sequence

    def load_inputs_and_ground_truth(self, batch_data, end=None):
        sequences = []
        noised_sequences = []

        for data in batch_data:
            original_seq = self.pad_sequence(data['sequence'])
            noised_seq = self.add_noise(original_seq)

            sequences.append(torch.tensor(original_seq, dtype=torch.long))
            noised_sequences.append(torch.tensor(noised_seq, dtype=torch.long))

        sequences = torch.stack(sequences).to(self.device)
        noised_sequences = torch.stack(noised_sequences).to(self.device)

        return noised_sequences, sequences

    def forward(self, inputs):
        return self.model(inputs)

    def compute_loss(self, outputs, ground_truth):
        # outputs: (batch_size, seq_len, vocab_size)
        # ground_truth: (batch_size, seq_len)
        return self.loss_fn(outputs.view(-1, self.vocab_size), ground_truth.view(-1))

    def evaluate(self, batch_data):
        inputs, targets = self.load_inputs_and_ground_truth(batch_data)
        with torch.no_grad():
            outputs = self.forward(inputs)
            loss = self.compute_loss(outputs, targets)
        return loss.item()
