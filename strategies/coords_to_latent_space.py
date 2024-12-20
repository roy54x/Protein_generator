import esm
import torch
import torch.nn.functional as F
from esm.inverse_folding.gvp_encoder import GVPEncoder
from esm.inverse_folding.gvp_transformer import GVPTransformerModel
from esm.inverse_folding.util import CoordBatchConverter
from proteinbert import load_pretrained_model
from proteinbert.conv_and_global_attention_model import get_model_with_hidden_layers_as_outputs

from constants import MAX_TRAINING_SIZE, MIN_SIZE
from strategies.base import Base


class CoordsToLatentSpace(Base):

    def __init__(self):
        super(CoordsToLatentSpace, self).__init__()
        self.model, self.alphabet = esm.pretrained.esm_if1_gvp4_t16_142M_UR50()

        self.args = self.model.args
        self.args.top_k_neighbors = 30
        self.args.max_tokens = MAX_TRAINING_SIZE

        self.gvp_encoder = GVPEncoder(self.args)
        self.device = next(self.model.parameters()).device

    def load_inputs_and_ground_truth(self, batch_data, end=None):
        batch_sequences = [data['sequence'] for data in batch_data]
        batch_coords = [data['coords'] for data in batch_data]
        input_coords = torch.cat(batch_coords, dim=0)

        pretrained_model_generator, input_encoder = load_pretrained_model()
        model = get_model_with_hidden_layers_as_outputs(pretrained_model_generator.create_model(MAX_TRAINING_SIZE))
        encoded_x = input_encoder.encode_X(batch_sequences, MAX_TRAINING_SIZE)
        local_representations, global_representations = model.predict(encoded_x, batch_size=1)

        return (input_coords, None, None, None), (local_representations, global_representations)

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
