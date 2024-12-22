import argparse

import esm
import torch
import torch.nn.functional as F
from esm.inverse_folding.gvp_encoder import GVPEncoder
from esm.inverse_folding.gvp_transformer import GVPTransformerModel
from esm.inverse_folding.util import CoordBatchConverter
from proteinbert import load_pretrained_model
from proteinbert.conv_and_global_attention_model import get_model_with_hidden_layers_as_outputs
from torch import nan_to_num

from constants import MAX_TRAINING_SIZE, MIN_SIZE
from strategies.base import Base


class CoordsToLatentSpace(Base):

    def __init__(self):
        super(CoordsToLatentSpace, self).__init__()
        self.pretrained_model, self.alphabet = esm.pretrained.esm_if1_gvp4_t16_142M_UR50()

        self.args = argparse.Namespace()
        for k, v in vars(self.pretrained_model.args).items():
            if k.startswith("gvp_"):
                setattr(self.args, k[4:], v)
        self.args.max_tokens = MAX_TRAINING_SIZE

        pretrained_model_generator, self.protein_bert_tokenizer = load_pretrained_model()
        self.protein_bert_model = get_model_with_hidden_layers_as_outputs(
            pretrained_model_generator.create_model(MAX_TRAINING_SIZE))

        self.gvp_encoder = GVPEncoder(self.args)
        self.batch_converter = CoordBatchConverter(self.alphabet)
        self.device = next(self.gvp_encoder.parameters()).device

    def load_inputs_and_ground_truth(self, batch_data, end=None):
        batch_converter_input = []
        batch_sequences = []

        for data in batch_data:
            sequence = data['sequence']
            coords = [[[float('inf') if x is None else x for x in atom]
                       for atom in residue] for residue in data['coords']]
            batch_sequences.append(data['sequence'])
            batch_converter_input.append((coords, None, sequence))

        coords, confidence, _, _, padding_mask = self.batch_converter(batch_converter_input, device=self.device)
        coord_mask = torch.all(torch.all(torch.isfinite(coords), dim=-1), dim=-1)
        coords = nan_to_num(coords)

        encoded_x = self.protein_bert_tokenizer.encode_X(batch_sequences, MAX_TRAINING_SIZE)
        local_representations, global_representations = self.protein_bert_model.predict(encoded_x, batch_size=1)

        return (coords, coord_mask, padding_mask, confidence), (local_representations, global_representations)

    def forward(self, inputs):
        coords, coord_mask, padding_mask, confidence = inputs
        encoder_out = self.gvp_encoder(coords, coord_mask, padding_mask, confidence)
        return encoder_out

    def compute_loss(self, outputs, ground_truth):
        return F.cross_entropy(outputs, ground_truth)

    def evaluate(self, data):
        ground_truth_sequence = data["sequence"]
        partial_seq = ground_truth_sequence[:MIN_SIZE+1]
        coords = data["coords"]
        predicted_sequence = self.gvp_transformer.sample(coords, partial_seq=partial_seq)

        for idx in range(0, len(ground_truth_sequence)):
            print("real values: " + str(ground_truth_sequence[idx]) + ", predicted values: " + str(predicted_sequence[idx]))
