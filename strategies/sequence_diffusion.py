import random

import torch
import torch.nn as nn

from constants import MAX_TRAINING_SIZE, AMINO_ACID_TO_INDEX, PAD_IDX
from strategies.base import Base
from utils.utils import get_blosum_probability_function


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
        self.get_prob_vec, self.aa_list = get_blosum_probability_function()
        self.aa_to_idx = {aa: i for i, aa in enumerate(self.aa_list)}
        self.idx_to_aa = {i: aa for aa, i in self.aa_to_idx.items()}

        self.device = "cuda:0" if torch.cuda.is_available() else "cpu"
        self.vocab_size = len(AMINO_ACID_TO_INDEX)
        self.model = RobertaBlock(self.vocab_size).to(self.device)
        self.loss_fn = nn.CrossEntropyLoss(ignore_index=PAD_IDX)

    def pad_sequence(self, sequence):
        tokenized = [AMINO_ACID_TO_INDEX.get(aa, PAD_IDX) for aa in sequence]
        padded = tokenized[:MAX_TRAINING_SIZE] + [PAD_IDX] * (MAX_TRAINING_SIZE - len(tokenized))
        return padded

    def add_noise(self, sequence, t=1, reduce_odds=0.25, addition_odds=0.75):
        """sequence: list of amino acid **letters**"""
        s = list(sequence)

        for _ in range(t):
            # For all valid amino acids, get their prob vectors
            prob_vectors = [self.get_prob_vec(aa) for aa in s]
            prob_tensor = torch.tensor(prob_vectors)  # (n_valid, vocab_size)

            # Sample replacements
            replacements = torch.multinomial(prob_tensor, num_samples=1).squeeze(-1)
            s = "".join([self.aa_list[idx] for idx in replacements])

            # Randomly remove one AA from end
            if len(s) > 0 and random.random() < reduce_odds:
                s = s[:-1]

            # Randomly add one AA to end
            if random.random() < addition_odds:
                s = s + random.choice(self.aa_list)

        return s

    def load_inputs_and_ground_truth(self, batch_data, end=None):
        sequences = []
        noised_sequences = []

        for data in batch_data:
            noised_seq = self.add_noise(data['sequence'])

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
