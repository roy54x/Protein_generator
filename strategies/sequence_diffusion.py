import blosum as bl
import numpy as np
import scipy.special as sp
import torch
import torch.nn as nn

from constants import MAX_TRAINING_SIZE, AMINO_ACID_TO_INDEX, PAD_IDX, INDEX_TO_AMINO_ACID
from strategies.base import Base
from utils.utils import get_blosum_probability_function


class RobertaBlock(nn.Module):
    def __init__(self, vocab_size, hidden_dim=128, num_layers=2, num_heads=4):
        super().__init__()
        self.blosum_probs = get_blosum_probability_function()
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
