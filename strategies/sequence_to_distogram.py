import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import transformers

from constants import AMINO_ACIDS, MAX_TRAINING_SIZE
from strategies.base import Base
from utils.padding_functions import padd_sequence, padd_contact_map
from utils.structure_utils import get_distogram, plot_contact_map, optimize_points_from_distogram, align_points, \
    plot_protein_atoms
from utils.utils import normalize


class SequenceToDistogram(Base):

    def __init__(self):
        super(SequenceToDistogram, self).__init__()

        self.hidden_size = 32

        config = transformers.RobertaConfig(
            vocab_size=len(AMINO_ACIDS) + 1,
            max_position_embeddings=MAX_TRAINING_SIZE + 2,
            hidden_size=self.hidden_size,
            num_attention_heads=4,
            num_hidden_layers=4,
            type_vocab_size=1
        )
        self.transformer = transformers.RobertaModel(config=config)

        self.mlp = nn.Sequential(
            nn.Linear(4 * self.hidden_size + 1, 32),
            nn.ReLU(),
            nn.Linear(32, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
            nn.ReLU())

    def load_inputs_and_ground_truth(self, batch_data, normalize_distogram=True):
        x_tensors, mask_tensors, ground_truths = [], [], []

        for data in batch_data:
            sequence = data['sequence']
            ca_coords = [coord[1] for coord in data["coords"]]

            if self.training:
                start, end = self.get_augmentation_indices(len(sequence))
            else:
                start, end = 0, MAX_TRAINING_SIZE

            # Get input
            sequence = sequence[start: end]
            x_tensor, mask_tensor = padd_sequence(sequence, MAX_TRAINING_SIZE)

            # Get ground truth
            distogram = get_distogram(ca_coords)
            distogram = distogram[start: end, start: end]
            if normalize_distogram:
                distogram = normalize(distogram)
            ground_truth = padd_contact_map(distogram, MAX_TRAINING_SIZE)

            # Append to batch lists
            x_tensors.append(x_tensor)
            mask_tensors.append(mask_tensor)
            ground_truths.append(ground_truth)

        # Stack tensors for batch
        x_tensors = torch.stack(x_tensors, dim=0)
        mask_tensors = torch.stack(mask_tensors, dim=0)
        ground_truths = torch.stack(ground_truths, dim=0)

        return (x_tensors, mask_tensors), ground_truths

    def get_indices_difference(self, x, batch_size, max_tokens):
        i_indices = (torch.arange(max_tokens).view(1, max_tokens, 1)
                     .expand(batch_size, max_tokens, max_tokens))
        j_indices = (torch.arange(max_tokens).view(1, 1, max_tokens)
                     .expand(batch_size, max_tokens, max_tokens))
        index_diff = (i_indices - j_indices).abs().float().to(
            device=x.device)  # Shape: (batch_size, max_tokens, max_tokens, 1)
        return index_diff

    def forward(self, input):
        x, mask = input

        # x is of shape (batch_size, max_tokens, 1)
        x = self.transformer(x, attention_mask=mask).last_hidden_state  # Shape: (batch_size, max_tokens, hidden_size)

        batch_size, max_tokens, hidden_size = x.size()
        x_i = x.unsqueeze(2)  # Shape: (batch_size, max_tokens, 1, hidden_size)
        x_i_expanded = x_i.expand(batch_size, max_tokens, max_tokens,
                                  hidden_size)  # Shape: (batch_size, max_tokens, max_tokens, hidden_size)
        x_j = x.unsqueeze(1)  # Shape: (batch_size, 1, max_tokens, hidden_size)
        x_j_expanded = x_j.expand(batch_size, max_tokens, max_tokens,
                                  hidden_size)  # Shape: (batch_size, max_tokens, max_tokens, hidden_size)

        difference = x_i - x_j  # Shape: (batch_size, max_tokens, max_tokens, hidden_size)
        multiplication = x_i * x_j  # Shape: (batch_size, max_tokens, max_tokens, hidden_size)

        index_diff = self.get_indices_difference(x, batch_size, max_tokens).unsqueeze(-1)

        concatenated = torch.cat((x_i_expanded, x_j_expanded, difference, multiplication, index_diff),
                                 dim=-1)  # Shape: (batch_size, max_tokens, max_tokens, 4 * hidden_size)
        concatenated = concatenated.view(batch_size * max_tokens * max_tokens, -1)
        out = self.mlp(concatenated)  # Shape: (batch_size * max_tokens * max_tokens, 1)

        # Zero the diagonal and normalize
        out = out.view(batch_size, max_tokens, max_tokens)
        diagonal_mask = (torch.ones(max_tokens, max_tokens, device=out.device)
                         - torch.eye(max_tokens, device=out.device))
        out = out * diagonal_mask.unsqueeze(0)  # Shape: (batch_size, max_tokens, max_tokens)
        max_values, _ = torch.max(out.view(batch_size, -1), dim=-1, keepdim=True)
        out = out / max_values.view(batch_size, 1, 1)

        return out, mask

    def compute_loss(self, outputs, ground_truth):
        prediction, mask = outputs
        batch_size, max_tokens, _ = prediction.shape
        mask = mask.unsqueeze(1) * mask.unsqueeze(2)  # Shape: (batch_size, max_tokens, max_tokens)
        index_diff = self.get_indices_difference(prediction, batch_size, max_tokens) * mask

        # Compute the L1 loss for each sample individually and multiply by the absolute index difference
        l1_loss_per_sample = F.l1_loss(prediction, ground_truth, reduction='none')
        l1_loss_per_sample *= index_diff  # Shape: (batch_size, max_tokens, max_tokens)

        # Sum the loss over the max_size dimensions
        l1_loss_per_sample = l1_loss_per_sample.sum(dim=[1, 2])  # Shape: batch_size
        elements_per_sample = index_diff.sum(dim=[1, 2])
        valid_mask = elements_per_sample > 0
        l1_loss_per_sample[valid_mask] /= elements_per_sample[valid_mask]

        # Average the loss over the batch
        average_loss = l1_loss_per_sample.mean()  # Shape: 1

        return average_loss

    def evaluate(self, data):
        ground_truth_coords = np.array(data["coords"], dtype="float16")[:MAX_TRAINING_SIZE]
        seq_len = len(data["sequence"])
        (x_tensor, mask_tensor), ground_truth_distogram = self.load_inputs_and_ground_truth(
            data, normalize_distogram=False)
        ground_truth_distogram = ground_truth_distogram[: MAX_TRAINING_SIZE, :MAX_TRAINING_SIZE]

        # Get model prediction
        predicted_distogram = self.forward((x_tensor.unsqueeze(0), mask_tensor.unsqueeze(0)))[0]
        predicted_distogram = predicted_distogram.squeeze().detach().numpy()
        predicted_distogram = predicted_distogram[: seq_len, :seq_len]
        predicted_distogram = (predicted_distogram + predicted_distogram.T) / 2
        predicted_distogram *= ground_truth_distogram.max().numpy()

        # Plot the predicted distogram and the ground truth distogram
        plot_contact_map(predicted_distogram, ground_truth_distogram)

        # Plot the aligned predicted coordinates with the ground truth coordinates
        predicted_coords = optimize_points_from_distogram(predicted_distogram)
        aligned_predicted_coords = align_points(predicted_coords, ground_truth_coords)
        plot_protein_atoms(aligned_predicted_coords, ground_truth_coords)
