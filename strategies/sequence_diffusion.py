import random

import numpy as np
import torch
import torch.nn as nn
from transformers import RobertaModel, RobertaTokenizer

from constants import MAX_TRAINING_SIZE, AMINO_ACID_TO_INDEX, PAD_IDX
from strategies.base import Base
from utils.utils import get_blosum_probability_function


class SequenceDiffusion(Base):
    def __init__(self):
        super().__init__()
        self.get_prob_vec, self.aa_list = get_blosum_probability_function()
        self.aa_to_idx = {aa: i for i, aa in enumerate(self.aa_list)}
        self.idx_to_aa = {i: aa for aa, i in self.aa_to_idx.items()}

        self.vocab_size = len(AMINO_ACID_TO_INDEX)
        self.roberta = RobertaModel.from_pretrained("roberta-base")
        self.lm_head = nn.Linear(self.roberta.config.hidden_size, self.vocab_size)
        self.loss_fn = nn.CrossEntropyLoss(ignore_index=PAD_IDX)
        self.device = "cuda:0" if torch.cuda.is_available() else "cpu"

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
        timesteps = []

        for data in batch_data:
            t = random.randint(1, 5)
            noised_seq = self.add_noise(data['sequence'], t=t)

            sequences.append(torch.tensor(self.pad_sequence(data['sequence']), dtype=torch.long))
            noised_sequences.append(torch.tensor(self.pad_sequence(noised_seq), dtype=torch.long))
            timesteps.append(t)

        sequences = torch.stack(sequences).to(self.device)
        noised_sequences = torch.stack(noised_sequences).to(self.device)
        timesteps = torch.tensor(timesteps, dtype=torch.long).to(self.device)

        return (noised_sequences, timesteps), sequences

    def forward(self, inputs):
        """
        Forward pass to reconstruct the original sequence from the noised input.
        """
        (noised_sequences, timesteps) = inputs
        embeddings = self.roberta.embeddings(input_ids=noised_sequences)
        outputs = self.roberta.encoder(embeddings)
        hidden_states = outputs[0]  # (batch_size, seq_len, hidden_size)
        logits = self.lm_head(hidden_states)
        return logits

    def compute_loss(self, outputs, ground_truth):
        return self.loss_fn(outputs.view(-1, self.vocab_size), ground_truth.view(-1))

    def evaluate(self, batch_data):
        noised_sequences, ground_truth = self.load_inputs_and_ground_truth(batch_data)
        outputs = self.forward(noised_sequences)
        loss = self.compute_loss(outputs, ground_truth)
        return loss
