import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from constants import AMINO_ACIDS, MAX_SIZE
from strategies.base import BaseStrategy


class SequenceToContactMapStrategy(BaseStrategy):
    AMINO_ACID_TO_INDEX = {aa: i + 1 for i, aa in enumerate(AMINO_ACIDS)}  # 1-indexed for padding

    def __init__(self):
        super(SequenceToContactMapStrategy, self).__init__()

        # Transformer encoder layer
        self.transformer_layer = nn.TransformerEncoderLayer(d_model=MAX_SIZE, nhead=4, dim_feedforward=256)
        self.transformer = nn.TransformerEncoder(self.transformer_layer, num_layers=4)

        # Convolutional layers
        self.conv_layers = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=8, kernel_size=5, padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=8, out_channels=8, kernel_size=5, padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=8, out_channels=1, kernel_size=5, padding=1)
        )

    def load_inputs(self, data):
        sequence = data['sequence']
        tokens = [self.AMINO_ACID_TO_INDEX.get(aa, 0) for aa in sequence]  # 0 for unknown amino acids
        if len(tokens) < self.max_size:
            tokens += [0] * (self.max_size - len(tokens))
        return torch.tensor(tokens[:self.max_size], dtype=torch.float).unsqueeze(0)  # Shape: (1, max_size)

    def get_ground_truth(self, data):
        # Assume contact_map is stored as a numpy array or similar in the dataframe
        contact_map = data['contact_map']
        if contact_map.shape[0] < self.max_size or contact_map.shape[1] < self.max_size:
            padded_contact_map = np.zeros((self.max_size, self.max_size))
            padded_contact_map[:contact_map.shape[0], :contact_map.shape[1]] = contact_map
            return torch.tensor(padded_contact_map, dtype=torch.float).unsqueeze(0)
        return torch.tensor(contact_map[:self.max_size, :self.max_size], dtype=torch.float).unsqueeze(0)

    def forward(self, x):
        # x is of shape (batch_size, max_size)
        transformer_output = self.transformer(x)  # Shape: (batch_size, max_size, max_size)

        # Outer product to get pairwise interactions
        outer_product = torch.einsum('bij,bik->bijk', transformer_output, transformer_output)
        outer_product = outer_product.view(-1, self.max_size, self.max_size).unsqueeze(
            1)  # Shape: (batch_size, 1, max_size, max_size)

        # Apply convolutional layers
        output = self.conv_layers(outer_product)  # Shape: (batch_size, 1, max_size, max_size)
        return output.squeeze(1)  # Shape: (batch_size, max_size, max_size)

    def compute_loss(self, outputs, ground_truth):
        # Assuming ground_truth is already of shape (batch_size, max_size, max_size)
        return F.cross_entropy(outputs, ground_truth)
