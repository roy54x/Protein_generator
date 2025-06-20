import random

import numpy as np
import torch
import torch.nn as nn
from evodiff.pretrained import OA_DM_38M
from evodiff.generate import generate_oaardm

from constants import MAX_TRAINING_SIZE, AMINO_ACID_TO_INDEX, PAD_IDX
from strategies.base import Base
from utils.utils import get_blosum_probability_function


class SequenceDiffusion(Base):
    def __init__(self):
        super().__init__()
        self.get_prob_vec, self.aa_list = get_blosum_probability_function()
        self.aa_to_idx = {aa: i for i, aa in enumerate(self.aa_list)}
        self.idx_to_aa = {i: aa for aa, i in self.aa_to_idx.items()}
        self.noise_levels = 3

        # EvoDiff OA_DM_38M model and tokenizer
        checkpoint = OA_DM_38M()
        self.model, _, self.tokenizer, _ = checkpoint
        self.model.train()
        self.device = "cuda:0" if torch.cuda.is_available() else "cpu"
        self.model.to(self.device)
        self.loss_fn = nn.CrossEntropyLoss(ignore_index=PAD_IDX)

    def pad_sequence(self, sequence):
        # Use EvoDiff tokenizer for padding/encoding
        tokenized = self.tokenizer.tokenize(sequence)
        padded = tokenized[:MAX_TRAINING_SIZE] + [PAD_IDX] * (MAX_TRAINING_SIZE - len(tokenized))
        return padded

    def add_noise(self, sequence, t=1, reduce_odds=0.25, addition_odds=0.75):
        """sequence: list of amino acid **letters**"""
        s = list(sequence)

        for _ in range(t):
            # For all valid amino acids, get their prob vectors
            prob_vectors = [self.get_prob_vec(aa) for aa in s]
            prob_tensor = torch.tensor(np.stack(prob_vectors))  # (n_valid, vocab_size)

            # Sample replacements
            replacements = torch.multinomial(prob_tensor, num_samples=1).squeeze(-1)
            s = [self.aa_list[idx] for idx in replacements]

            # Randomly remove one AA from end
            if len(s) > 0 and random.random() < reduce_odds:
                s = s[:-1]

            # Randomly add one AA to end
            if random.random() < addition_odds:
                s.append(random.choice(self.aa_list))

        return "".join(s)

    def load_inputs_and_ground_truth(self, batch_data, t=1):
        sequences = []
        noised_sequences = []
        timesteps = []

        for data in batch_data:
            timestep = (
                random.randint(1, self.noise_levels)
                if self.training or t is None else t
            )
            noised_seq = self.add_noise(data['sequence'], t=timestep)

            sequences.append(torch.tensor(self.pad_sequence(data['sequence']), dtype=torch.long))
            noised_sequences.append(torch.tensor(self.pad_sequence(noised_seq), dtype=torch.long))
            timesteps.append(timestep)

        sequences = torch.stack(sequences).to(self.device)
        noised_sequences = torch.stack(noised_sequences).to(self.device)
        timesteps = torch.tensor(timesteps, dtype=torch.long).to(self.device)

        return (noised_sequences, timesteps, sequences), sequences

    def forward(self, inputs):
        noised_sequences, timesteps, ground_truth = inputs
        logits = self.model(noised_sequences, ground_truth)
        return logits

    def compute_loss(self, outputs, ground_truth):
        return self.loss_fn(outputs.reshape(-1, outputs.size(-1)), ground_truth.reshape(-1))

    def evaluate(self, batch_data):
        """
        Evaluates the model's recovery performance on a batch.
        """
        self.eval()
        with torch.no_grad():
            inputs, ground_truth = batch_data
            self.to(self.device)
            outputs = self(inputs)  # logits: (batch_size, seq_len, vocab_size)
            predicted_indices = torch.argmax(outputs, dim=-1)  # (batch_size, seq_len)

            # Mask padding tokens (assuming 0 is the padding token)
            padding_mask = ground_truth != 1
            correct = (predicted_indices == ground_truth) & padding_mask

            per_sequence_recovery = correct.sum(dim=1).float() / padding_mask.sum(dim=1).float()
            recovery_rate = per_sequence_recovery.mean()

            print(f"Recovery rate: {recovery_rate.item():.4f}")
            return recovery_rate.item()
