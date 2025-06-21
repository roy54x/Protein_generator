import random

import numpy as np
import torch
import torch.nn as nn
from evodiff.pretrained import OA_DM_38M
from evodiff.generate import generate_oaardm

from constants import MAX_TRAINING_SIZE, AMINO_ACID_TO_INDEX
from strategies.base import Base
from utils.utils import get_blosum_probability_function


class SequenceDiffusion(Base):
    def __init__(self):
        super().__init__()
        self.get_prob_vec, self.aa_list = get_blosum_probability_function()
        self.aa_to_idx = {aa: i for i, aa in enumerate(self.aa_list)}
        self.idx_to_aa = {i: aa for aa, i in self.aa_to_idx.items()}

        # EvoDiff OA_DM_38M model and tokenizer
        checkpoint = OA_DM_38M()
        self.model, _, self.tokenizer, _ = checkpoint
        self.pad_id = self.tokenizer.pad_id
        self.mask_id = self.tokenizer.mask_id
        self.model.train()
        self.device = "cuda:0" if torch.cuda.is_available() else "cpu"
        self.model.to(self.device)
        self.loss_fn = nn.CrossEntropyLoss(ignore_index=self.pad_id)

    def tokenize_and_padd(self, sequence):
        tokenized = list(self.tokenizer.tokenize([sequence]))
        padded = tokenized[:MAX_TRAINING_SIZE] + [self.pad_id] * (MAX_TRAINING_SIZE - len(tokenized))
        return padded

    def add_noise(self, input_tensor):
        L = input_tensor.size(0)
        t = random.randint(0, L)

        order = list(range(L))
        random.shuffle(order)

        masked_indices = set(order[:t])
        noised = [
            self.mask_id if i in masked_indices else input_tensor[i].item()
            for i in range(L)
        ]

        return torch.tensor(noised, dtype=torch.long), t

    def load_inputs_and_ground_truth(self, batch_data, t=1):
        gt_tensors = []
        noised_tensors = []
        timesteps = []

        for data in batch_data:
            padded_tensor = torch.tensor(self.tokenize_and_padd(data['sequence']), dtype=torch.long)
            noised_tensor, timestep = self.add_noise(padded_tensor)

            gt_tensors.append(padded_tensor)
            noised_tensors.append(noised_tensor)
            timesteps.append(timestep)

        gt_tensors = torch.stack(gt_tensors).to(self.device)
        noised_tensors = torch.stack(noised_tensors).to(self.device)
        timesteps = torch.tensor(timesteps, dtype=torch.long).to(self.device)

        return (noised_tensors, timesteps, gt_tensors), gt_tensors

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
