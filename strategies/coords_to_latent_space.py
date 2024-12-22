import esm
import numpy as np
import torch
import torch.nn.functional as F
from esm.inverse_folding.gvp_transformer_encoder import GVPTransformerEncoder
from esm.inverse_folding.util import CoordBatchConverter
from proteinbert import load_pretrained_model
from proteinbert.conv_and_global_attention_model import get_model_with_hidden_layers_as_outputs
from torch import nn

from constants import MAX_TRAINING_SIZE, MIN_SIZE, BATCH_SIZE
from strategies.base import Base


class CoordsToLatentSpace(Base):

    def __init__(self):
        super(CoordsToLatentSpace, self).__init__()
        self.pretrained_model, alphabet = esm.pretrained.esm_if1_gvp4_t16_142M_UR50()

        args = self.pretrained_model.args
        args.encoder_embed_dim = 1562
        args.encoder_ffn_embed_dim = 256
        args.encoder_layers = 4
        args.encoder_attention_heads = 11
        args.gvp_node_hidden_dim_scalar = 128
        args.gvp_node_hidden_dim_vector = 32
        args.gvp_edge_hidden_dim_scalar = 4
        args.max_tokens = MAX_TRAINING_SIZE

        encoder_embed_tokens = self.pretrained_model.build_embedding(
            args, alphabet, args.encoder_embed_dim,
        )

        self.gvp_transformer_encoder = GVPTransformerEncoder(args, alphabet, encoder_embed_tokens)
        self.batch_converter = CoordBatchConverter(alphabet)
        self.device = next(self.gvp_transformer_encoder.parameters()).device
        self.padding_token = alphabet.all_toks[alphabet.padding_idx]

        pretrained_model_generator, self.protein_bert_tokenizer = load_pretrained_model()
        self.protein_bert_model = get_model_with_hidden_layers_as_outputs(
            pretrained_model_generator.create_model(MAX_TRAINING_SIZE + 2))

        self.loss_fn = nn.CosineEmbeddingLoss(reduction="sum")

    def load_inputs_and_ground_truth(self, batch_data, end=None):
        batch_converter_input = []
        batch_sequences = []

        for data in batch_data:
            sequence = data['sequence']
            sequence_padded = sequence + self.padding_token * (MAX_TRAINING_SIZE - len(sequence))
            coords = [[[float('inf') if x is None else x for x in atom]
                       for atom in residue] for residue in data['coords']]
            coords_padded = (coords + [[[np.nan] * len(coords[0][0])] * len(coords[0])]
                             * (MAX_TRAINING_SIZE - len(sequence)))
            batch_sequences.append(sequence)
            batch_converter_input.append((coords_padded, None, sequence_padded))

        coords, confidence, strs, tokens, padding_mask = self.batch_converter(batch_converter_input, device=self.device)

        encoded_x = self.protein_bert_tokenizer.encode_X(batch_sequences, MAX_TRAINING_SIZE + 2)
        local_representations, global_representations = self.protein_bert_model.predict(encoded_x, batch_size=BATCH_SIZE)

        return (coords, padding_mask, confidence), torch.tensor(local_representations)

    def forward(self, inputs):
        coords, padding_mask, confidence = inputs
        encoder_out = self.gvp_transformer_encoder(coords, padding_mask, confidence)
        return encoder_out["encoder_out"][0].transpose(0, 1)

    def compute_loss(self, outputs, ground_truth):
        outputs = outputs.reshape(-1, outputs.size(-1))
        ground_truth = ground_truth.reshape(-1, ground_truth.size(-1))
        target = torch.ones(outputs.size(0), device=outputs.device)
        return F.cosine_embedding_loss(outputs, ground_truth, target)

    def evaluate(self, data):
        ground_truth_sequence = data["sequence"]
        chain_id = data["chain_id"]
        inputs, gt_representations = self.load_inputs_and_ground_truth([data])

        # Get the predicted representation from the model
        predicted_representations = self(inputs)
        loss = self.compute_loss(predicted_representations, gt_representations).item()
        print(f"Distance in Embedding space is: {loss}")

        # Get the predicted sequence based on the decoder
        predicted_sequence = self.protein_bert_model

        # Compare ground truth and predicted sequence directly using vectorized operations
        correct_predictions = sum(a == b for a, b in zip(predicted_sequence, ground_truth_sequence))
        total_predictions = len(ground_truth_sequence)

        # Calculate and print the average recovery rate
        recovery_rate = correct_predictions / total_predictions if total_predictions > 0 else 0
        print(f"Recovery rate for protein: {chain_id} is {recovery_rate}")
        return recovery_rate
