import copy

import numpy as np
import torch
import torch.nn.functional as F
import transformers
from torch import nn

from constants import AMINO_ACIDS, MAX_TRAINING_SIZE
from strategies.base import Base
from utils.padding_functions import padd_sequence, padd_contact_map
from utils.structure_utils import get_distogram


class DistogramToSequence(Base):

    def __init__(self):
        super(DistogramToSequence, self).__init__()
        self.vocab_size = len(AMINO_ACIDS) + 1
        self.hidden_size = 256
        self.num_layers = 4
        self.num_heads = 16
        self.attention_layers = nn.ModuleList([
            nn.MultiheadAttention(embed_dim=self.hidden_size, num_heads=self.num_heads, batch_first=True)
            for _ in range(self.num_layers)
        ])
        self.linear1 = nn.Linear(self.hidden_size * 3, self.hidden_size)
        self.linear2 = nn.Linear(self.hidden_size, self.vocab_size)

    def load_inputs_and_ground_truth(self, data, normalize_distogram=True):
        sequence = data['sequence']
        if self.training:
            start, end = self.get_augmentation_indices(len(sequence))
        else:
            start, end = 0, MAX_TRAINING_SIZE
        sequence = sequence[start: end]
        sequence_tensor, mask_tensor = padd_sequence(sequence, MAX_TRAINING_SIZE)

        # Get ground truth
        ground_truth = copy.deepcopy(sequence_tensor[len(sequence) - 1]).to(torch.long)
        ground_truth = F.one_hot(ground_truth, num_classes=self.vocab_size).float()

        # Get inputs
        sequence_tensor[len(sequence) - 1] = 0

        distogram = get_distogram(data["coords"])
        distances = distogram[start: end, start: end][-1]
        if normalize_distogram:
            distances = (distances - distances.min()) / (distances.max() - distances.min())
        distances = np.pad(distances, (0, MAX_TRAINING_SIZE - len(distances)), mode='constant')

        return (sequence_tensor, distances, mask_tensor), ground_truth

    def forward(self, inputs):
        x, distances, mask_tensor = inputs
        weights = (1 - distances) * mask_tensor

        x, weights = x.unsqueeze(-1).expand(-1, -1, self.hidden_size), weights.unsqueeze(-1).expand(-1, -1, self.hidden_size)

        x = x.to(torch.float32)
        mask_tensor = ~mask_tensor.to(bool)

        for layer_idx, attention in enumerate(self.attention_layers):
            x, _ = attention(weights, x, x, key_padding_mask=mask_tensor)

        pooled_output_mean = x.mean(dim=1)  # Shape: (batch_size, hidden_size)
        pooled_output_max, _ = x.max(dim=1)
        pooled_output_min, _ = x.min(dim=1)

        concatenated_output = torch.cat((pooled_output_mean, pooled_output_max, pooled_output_min), dim=-1)
        output = self.linear1(concatenated_output)
        output = F.relu(output)
        output = self.linear2(output)
        probabilities = F.softmax(output, dim=-1)

        return probabilities  # Shape: (batch_size, vocab_size)


    def compute_loss(self, outputs, ground_truth):
        return F.cross_entropy(outputs, ground_truth)
