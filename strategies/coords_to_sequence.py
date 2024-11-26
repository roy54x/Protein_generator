import esm
import torch
import torch.nn.functional as F
from esm.inverse_folding.gvp_transformer import GVPTransformerModel
from esm.inverse_folding.util import CoordBatchConverter

from constants import MAX_TRAINING_SIZE
from strategies.base import Base


class CoordsToSequence(Base):

    def __init__(self):
        super(CoordsToSequence, self).__init__()
        model, self.alphabet = esm.pretrained.esm_if1_gvp4_t16_142M_UR50()
        self.args = model.args
        self.args.encoder_embed_dim = 128
        self.args.decoder_embed_dim = 128
        self.args.decoder_input_dim = 128
        self.args.decoder_output_dim = 128
        self.args.encoder_ffn_embed_dim = 512
        self.args.decoder_ffn_embed_dim = 512
        self.args.encoder_layers = 4
        self.args.decoder_layers = 4
        self.args.encoder_attention_heads = 4
        self.args.decoder_attention_heads = 4
        self.args.gvp_node_hidden_dim_scalar = 256
        self.args.gvp_node_hidden_dim_vector = 64
        self.args.gvp_edge_hidden_dim_scalar = 8

        self.gvp_transformer = GVPTransformerModel(self.args, self.alphabet)
        self.batch_converter = CoordBatchConverter(self.alphabet)
        self.device = next(model.parameters()).device

    def load_inputs_and_ground_truth(self, data, end=None):
        sequence = data['sequence']
        if self.training:
            start, end = self.get_augmentation_indices(len(sequence))
        elif end:
            start, end = max(0, end-MAX_TRAINING_SIZE), end
        else:
            start, end = 0, MAX_TRAINING_SIZE
        sequence = sequence[start: end]
        coords = data["coords"][start: end]

        return (coords, None, sequence), None

    def collate(self, batch):
        batch = [x[0] for x in batch]
        coords, confidence, strs, tokens, padding_mask = self.batch_converter(batch, device=self.device)
        prev_output_tokens = tokens[:, :-1]
        ground_truth = tokens[:, 1:]

        return (coords, padding_mask, confidence, prev_output_tokens), ground_truth

    def forward(self, inputs):
        coords, padding_mask, confidence, prev_output_tokens = inputs
        outputs, _ = self.gvp_transformer(coords, padding_mask, confidence, prev_output_tokens)
        return outputs

    def compute_loss(self, outputs, ground_truth):
        return F.cross_entropy(outputs, ground_truth, reduction='none')

    def evaluate(self, data):
        pass
