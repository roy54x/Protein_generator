import esm
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
        self.model, self.alphabet = esm.pretrained.esm_if1_gvp4_t16_142M_UR50()

        self.args = self.model.args
        self.args.encoder_embed_dim = 64
        self.args.decoder_embed_dim = 64
        self.args.decoder_input_dim = 64
        self.args.decoder_output_dim = 64
        self.args.encoder_ffn_embed_dim = 256
        self.args.decoder_ffn_embed_dim = 256
        self.args.encoder_layers = 4
        self.args.decoder_layers = 4
        self.args.encoder_attention_heads = 4
        self.args.decoder_attention_heads = 4
        self.args.gvp_node_hidden_dim_scalar = 128
        self.args.gvp_node_hidden_dim_vector = 32
        self.args.gvp_edge_hidden_dim_scalar = 4
        self.args.max_tokens = MAX_TRAINING_SIZE

        self.gvp_transformer = GVPTransformerModel(self.args, self.alphabet)
        self.batch_converter = CoordBatchConverter(self.alphabet)
        self.device = next(self.model.parameters()).device

    def load_inputs_and_ground_truth(self, batch_data, end=None):
        batch_converter_input = []

        for data in batch_data:
            sequence = data['sequence']
            coords = data['coords']
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
        partial_seq = ground_truth_sequence[:MIN_SIZE+1]
        coords = data["coords"]
        predicted_sequence = self.gvp_transformer.sample(coords, partial_seq=partial_seq)

        for idx in range(0, len(ground_truth_sequence)):
            print("real values: " + str(ground_truth_sequence[idx]) + ", predicted values: " + str(predicted_sequence[idx]))
