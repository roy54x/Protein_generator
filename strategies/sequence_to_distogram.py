import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import transformers

from constants import AMINO_ACIDS, MAX_TRAINING_SIZE, DECAY_RATE
from strategies.base import Base
from utils.padding_functions import padd_sequence, padd_contact_map
from utils.structure_utils import get_soft_contact_map, get_distogram


class SequenceToDistogram(Base):

    def __init__(self):
        super(SequenceToDistogram, self).__init__()

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

    def load_inputs_and_ground_truth(self, data):
        sequence = data['sequence']

        # Get input
        if self.training:
            start, end = self.get_augmentation_indices(len(sequence))
            sequence = sequence[start: end]
        else:
            sequence = sequence[:MAX_TRAINING_SIZE, :MAX_TRAINING_SIZE]
        x_tensor, mask_tensor = padd_sequence(sequence, MAX_TRAINING_SIZE)

        # Get ground truth
        distogram = get_distogram(data["coords"])
        if self.training:
            distogram = distogram[start: end, start: end]
        else:
            distogram = distogram[:MAX_TRAINING_SIZE, :MAX_TRAINING_SIZE]
        ground_truth = padd_contact_map(distogram, MAX_TRAINING_SIZE)

        return (x_tensor, mask_tensor), ground_truth

    def forward(self, input):
        x, mask = input

        # x is of shape (batch_size, max_size, 1)
        transformer_output = self.transformer(x,
                                              attention_mask=mask).last_hidden_state  # Shape: (batch_size, max_size, max_size)

        # Outer product to get pairwise interactions
        pairwise_interactions = torch.einsum('bik,bjk->bij', transformer_output, transformer_output)
        outer_product = pairwise_interactions.view(-1, MAX_TRAINING_SIZE, MAX_TRAINING_SIZE).unsqueeze(1)  # Shape: (batch_size, 1, max_size, max_size)

        # Apply convolutional layers
        output = self.conv_layers(outer_product)  # Shape: (batch_size, 1, max_size, max_size)
        return output.squeeze(1), mask  # Shape: (batch_size, max_size, max_size)

    def compute_loss(self, outputs, ground_truth):
        prediction, mask = outputs
        mask = mask.unsqueeze(1) * mask.unsqueeze(2)  # Shape: (batch_size, max_size, max_size)
        mse_loss = F.mse_loss(prediction * mask, ground_truth * mask, reduction='sum')
        num_valid_elements = mask.sum()
        if num_valid_elements > 0:
            mse_loss = mse_loss / num_valid_elements
        return mse_loss
