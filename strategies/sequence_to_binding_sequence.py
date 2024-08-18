import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import transformers

from utils.constants import AMINO_ACIDS, MAX_SIZE, AMINO_ACID_TO_INDEX
from strategies.base import Base
from utils.padding_functions import padd_sequence


class DiffusionModule(nn.Module):
    def __init__(self, hidden_size, timesteps=100, num_layers=3, noise_steps=5):
        super(DiffusionModule, self).__init__()
        self.timesteps = timesteps
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.noise_steps = noise_steps
        self.layers = nn.ModuleList([nn.Linear(hidden_size, hidden_size) for _ in range(num_layers)])

    def forward(self, x):
        for t in range(self.timesteps):
            for layer in self.layers:
                x = layer(x)
                x = F.relu(x)
                if t % (self.timesteps // self.noise_steps) == 0:
                    noise = torch.randn_like(x)
                    x = x + noise
        return x


class SequenceDiffusionModel(Base):

    def __init__(self):
        super(SequenceDiffusionModel, self).__init__()
        self.vocab_size = len(AMINO_ACIDS) + 1
        self.min_split_percent, self.max_split_percent = 0.5, 0.9
        self.max_input_size, self.max_output_size = int(MAX_SIZE * self.max_split_percent), int(MAX_SIZE * self.min_split_percent)
        self.hidden_size = 8

        config = transformers.RobertaConfig(
            vocab_size=self.vocab_size,
            max_position_embeddings=self.max_input_size+2,
            hidden_size=self.hidden_size,
            num_attention_heads=4,
            num_hidden_layers=4,
            type_vocab_size=1
        )
        self.transformer = transformers.RobertaModel(config=config)
        self.diffusion = DiffusionModule(hidden_size=self.hidden_size, timesteps=10, num_layers=3)
        self.output_layer = nn.Linear(self.hidden_size, self.max_output_size
                                      * self.vocab_size)

    def load_inputs_and_ground_truth(self, data):
        sequence = data['sequence']
        start_idx, end_idx = self.get_augmentation_indices(len(sequence))
        split_percent = np.random.uniform(self.min_split_percent, self.max_split_percent)
        split_idx = start_idx + int((end_idx - start_idx) * split_percent)
        input_seq = sequence[start_idx:split_idx]
        ground_truth_seq = sequence[split_idx:end_idx]

        # Get input
        x_tensor, mask_tensor = padd_sequence(input_seq, self.max_input_size)

        # Get ground truth and one-hot encode it
        ground_truth_indices = [AMINO_ACID_TO_INDEX[aa] for aa in ground_truth_seq]
        ground_truth = torch.zeros((self.max_output_size, self.vocab_size))  # One-hot tensor
        for i, idx in enumerate(ground_truth_indices):
            ground_truth[i, idx] = 1

        return (x_tensor, mask_tensor), ground_truth

    def forward(self, input):
        x, mask = input
        batch_size = x.shape[0]

        # x is of shape (batch_size, max_size, 1)
        transformer_pooled_output = self.transformer(x,
                                                     attention_mask=mask).pooler_output  # Shape: (batch_size, hidden_size)
        diffusion_output = self.diffusion(transformer_pooled_output)  # Shape: (batch_size, hidden_size)
        output = self.output_layer(diffusion_output)  # Shape: (batch_size, max_size * vocab_size)
        output = output.view(batch_size, self.max_output_size, self.vocab_size)
        output = F.softmax(output, dim=-1)
        return output

    def compute_loss(self, outputs, ground_truth):
        # Assuming ground_truth is one-hot encoded and of shape (batch_size, max_size, vocab_size)
        return F.cross_entropy(outputs.view(-1, outputs.size(-1)), ground_truth.argmax(dim=-1).view(-1))
