import esm
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

        # Get inputs
        batch = [(coords, None, sequence)]
        coords, confidence, strs, tokens, padding_mask = self.batch_converter(batch, device=self.device)
        prev_output_tokens = tokens[:, :-1]

        # Get ground truth
        ground_truth = tokens[:, 1:]

        return (coords, prev_output_tokens), ground_truth

    def forward(self, inputs):
        (coords, prev_output_tokens) = inputs
        outputs, _ = self.gvp_transformer(coords, None, None, prev_output_tokens)
        return outputs

    def compute_loss(self, outputs, ground_truth):
        return F.cross_entropy(outputs, ground_truth, reduction='none')

    def evaluate(self, data):
        pass
