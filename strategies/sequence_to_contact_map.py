import ast

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import transformers

from constants import AMINO_ACIDS, MAX_SIZE
from strategies.base import BaseStrategy


class SequenceToContactMapStrategy(BaseStrategy):
    AMINO_ACID_TO_INDEX = {aa: i + 1 for i, aa in enumerate(AMINO_ACIDS)}  # 1-indexed for padding

    def __init__(self):
        super(SequenceToContactMapStrategy, self).__init__()

        # Transformer encoder layer
        config = transformers.RobertaConfig(
            vocab_size=len(AMINO_ACIDS) + 1,
            max_position_embeddings=252,
            hidden_size=36,
            num_attention_heads=6,
            num_hidden_layers=6,
            type_vocab_size=1
        )
        self.transformer = transformers.RobertaModel(config=config)

        # Convolutional layers
        self.conv_layers = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=8, kernel_size=5, padding="same"),
            nn.ReLU(),
            nn.Conv2d(in_channels=8, out_channels=8, kernel_size=5, padding="same"),
            nn.ReLU(),
            nn.Conv2d(in_channels=8, out_channels=1, kernel_size=5, padding="same")
        )

    def load_inputs(self, data):
        sequence = data['sequence']
        tokens = [self.AMINO_ACID_TO_INDEX.get(aa, 0) for aa in sequence]  # 0 for unknown amino acids
        if len(tokens) < MAX_SIZE:
            tokens += [0] * (MAX_SIZE - len(tokens))
        return torch.tensor(tokens[:MAX_SIZE], dtype=torch.int)  # Shape: (max_size)

    def get_ground_truth(self, data):
        # Assume contact_map is stored as a numpy array or similar in the dataframe
        contact_map = np.array(data['contact_map'])
        if contact_map.shape[0] < MAX_SIZE or contact_map.shape[1] < MAX_SIZE:
            padded_contact_map = np.zeros((MAX_SIZE, MAX_SIZE))
            padded_contact_map[:contact_map.shape[0], :contact_map.shape[1]] = contact_map
            return torch.tensor(padded_contact_map, dtype=torch.float)
        return torch.tensor(contact_map[:MAX_SIZE, :MAX_SIZE], dtype=torch.float)

    def forward(self, x):
        # x is of shape (batch_size, max_size, 1)
        transformer_output = self.transformer(x).last_hidden_state  # Shape: (batch_size, max_size, max_size)

        # Outer product to get pairwise interactions
        pairwise_interactions = torch.einsum('bik,bjk->bij', transformer_output, transformer_output)
        outer_product = pairwise_interactions.view(-1, MAX_SIZE, MAX_SIZE).unsqueeze(1)  # Shape: (batch_size, 1, max_size, max_size)

        # Apply convolutional layers
        output = self.conv_layers(outer_product)  # Shape: (batch_size, 1, max_size, max_size)
        return output.squeeze(1)  # Shape: (batch_size, max_size, max_size)

    def compute_loss(self, outputs, ground_truth):
        # Assuming ground_truth is already of shape (batch_size, max_size, max_size)
        return F.cross_entropy(outputs, ground_truth)
