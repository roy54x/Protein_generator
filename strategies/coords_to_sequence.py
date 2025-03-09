import esm
import numpy as np
import torch
import torch.nn.functional as F
from esm.inverse_folding.gvp_transformer import GVPTransformerModel
from esm.inverse_folding.util import CoordBatchConverter

from constants import MAX_TRAINING_SIZE, AMINO_ACIDS, MIN_SIZE
from strategies.base import Base
from utils.padding_functions import padd_sequence


class CoordsToSequence(Base):

    def __init__(self):
        super(CoordsToSequence, self).__init__()
        self.pretrained_model, self.alphabet = esm.pretrained.esm_if1_gvp4_t16_142M_UR50()
        self.args = self.pretrained_model.args
        self.pretrained_model = None

        self.gvp_transformer = GVPTransformerModel(self.args, self.alphabet)
        self.batch_converter = CoordBatchConverter(self.alphabet)
        self.device = next(self.gvp_transformer.parameters()).device

    def load_inputs_and_ground_truth(self, batch_data, end=None):
        batch_converter_input = []

        for data in batch_data:
            sequence = data['sequence']
            coords = [[[float('inf') if x is None else x for x in atom]
                       for atom in residue] for residue in data['coords']]
            batch_converter_input.append((coords, None, sequence))

        coords, confidence, strs, tokens, padding_mask = self.batch_converter(
            batch_converter_input, device=self.device)

        prev_output_tokens = tokens[:, :-1]
        ground_truth = tokens[:, 1:]

        return (coords, padding_mask, confidence, prev_output_tokens), ground_truth

    def forward(self, inputs):
        coords, padding_mask, confidence, prev_output_tokens = inputs
        logits, _ = self.gvp_transformer(coords, padding_mask, confidence, prev_output_tokens)
        return logits, coords[:, 1:-1]

    def compute_loss(self, outputs, ground_truth):
        target_padding_mask = (ground_truth == self.alphabet.padding_idx)
        logits, coords = outputs
        loss = F.cross_entropy(logits, ground_truth, reduction='none')
        coord_mask = torch.isfinite(coords).all(dim=(-1, -2))
        mask = (~target_padding_mask) & (coord_mask)
        ll = torch.sum(loss * mask) / torch.sum(mask)
        return ll

    def evaluate(self, batch_data):
        inputs, ground_truth = batch_data
        coords, padding_mask, confidence, prev_output_tokens = inputs

        recovery_rates = []

        for idx, sample_coords in enumerate(coords):
            sample_coords = sample_coords[~padding_mask[idx]][1:-1]
            ground_truth = ground_truth[idx][ground_truth[idx] != 1]
            ground_truth_sequence = "".join(self.alphabet.get_tok(i) for i in ground_truth)
            predicted_sequence = self.gvp_transformer.sample(sample_coords, temperature=1e-6)
            correct_predictions = sum(a == b for a, b in zip(predicted_sequence, ground_truth_sequence))
            total_predictions = len(ground_truth)
            recovery_rate = correct_predictions / total_predictions if total_predictions > 0 else 0
            recovery_rates.append(recovery_rate)

        avg_recovery_rate = np.mean(recovery_rates)
        print(f"Recovery rate: {avg_recovery_rate}")
        return avg_recovery_rate
