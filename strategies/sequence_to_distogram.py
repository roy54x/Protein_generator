import torch
import torch.nn as nn
import torch.nn.functional as F
import transformers

from constants import AMINO_ACIDS, MAX_TRAINING_SIZE
from strategies.base import Base
from utils.padding_functions import padd_sequence, padd_contact_map
from utils.structure_utils import get_distogram
from utils.utils import normalize


class SequenceToDistogram(Base):

    def __init__(self):
        super(SequenceToDistogram, self).__init__()

        self.hidden_size = 64

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
            nn.Linear(4 * self.hidden_size + 1, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.ReLU())

    def load_inputs_and_ground_truth(self, data, normalize_distogram=True):
        sequence = data['sequence']

        if self.training:
            start, end = self.get_augmentation_indices(len(sequence))
        else:
            start, end = 0, MAX_TRAINING_SIZE

        # Get input
        sequence = sequence[start: end]
        x_tensor, mask_tensor = padd_sequence(sequence, MAX_TRAINING_SIZE)

        # Get ground truth
        distogram = get_distogram(data["coords"])
        distogram = distogram[start: end, start: end]
        if normalize_distogram:
            distogram = normalize(distogram)
        ground_truth = padd_contact_map(distogram, MAX_TRAINING_SIZE)

        return (x_tensor, mask_tensor), ground_truth

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

        i_indices = (torch.arange(max_tokens).view(1, max_tokens, 1)
                     .expand(batch_size, max_tokens, max_tokens))
        j_indices = (torch.arange(max_tokens).view(1, 1, max_tokens)
                     .expand(batch_size, max_tokens, max_tokens))
        index_diff = (i_indices - j_indices).unsqueeze(-1).float().to(device=self.device) # Shape: (batch_size, max_tokens, max_tokens, 1)

        concatenated = torch.cat((x_i_expanded, x_j_expanded, difference, multiplication, index_diff),
                                 dim=-1)  # Shape: (batch_size, max_tokens, max_tokens, 4 * hidden_size)
        concatenated = concatenated.view(batch_size * max_tokens * max_tokens, -1)
        out = self.mlp(concatenated)  # Shape: (batch_size * max_tokens * max_tokens, 1)

        out = out.view(batch_size, -1)
        max_values, _ = torch.max(out, dim=-1, keepdim=True)
        out = out / max_values
        out = out.view(batch_size, max_tokens, max_tokens)
        return out, mask

    def compute_loss(self, outputs, ground_truth):
        prediction, mask = outputs
        mask = mask.unsqueeze(1) * mask.unsqueeze(2)  # Shape: (batch_size, max_size, max_size)
        l1_loss = F.l1_loss(prediction * mask, ground_truth * mask, reduction='sum')
        num_valid_elements = mask.sum()
        if num_valid_elements > 0:
            l1_loss /= num_valid_elements
        return l1_loss
