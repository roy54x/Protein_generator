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
        self.args.max_tokens = MAX_TRAINING_SIZE
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
        outputs, _ = self.gvp_transformer(coords, padding_mask, confidence, prev_output_tokens)
        return outputs

    def compute_loss(self, outputs, ground_truth):
        return F.cross_entropy(outputs, ground_truth)

    def evaluate(self, data):
        ground_truth_sequence = data["sequence"]
        coords = data["coords"]
        chain_id = data["chain_id"]

        coords = [[[float('inf') if x is None else x for x in atom]
                   for atom in residue] for residue in coords]

        # Get the predicted sequence from the model
        predicted_sequence = self.gvp_transformer.sample(coords, temperature=1e-6)

        # Compare ground truth and predicted sequence directly using vectorized operations
        correct_predictions = sum(a == b for a, b in zip(predicted_sequence, ground_truth_sequence))
        total_predictions = len(ground_truth_sequence)

        # Calculate and print the average recovery rate
        recovery_rate = correct_predictions / total_predictions if total_predictions > 0 else 0
        print(f"Recovery rate for protein: {chain_id} is {recovery_rate}")
        return recovery_rate
