import esm
import numpy as np
import torch
import torch.nn.functional as F
from esm.inverse_folding.gvp_transformer_encoder import GVPTransformerEncoder
from esm.inverse_folding.util import CoordBatchConverter
from torch import nn

from constants import MAX_TRAINING_SIZE
from strategies.base import Base


class CoordsToLatentSpace(Base):

    def __init__(self):
        super(CoordsToLatentSpace, self).__init__()
        self.pretrained_inverse_model, inverse_alphabet = esm.pretrained.esm_if1_gvp4_t16_142M_UR50()

        args = self.pretrained_inverse_model.args
        args.encoder_embed_dim = 1280
        args.encoder_ffn_embed_dim = 256
        args.encoder_layers = 4
        args.encoder_attention_heads = 4
        args.gvp_node_hidden_dim_scalar = 128
        args.gvp_node_hidden_dim_vector = 32
        args.gvp_edge_hidden_dim_scalar = 4
        args.max_tokens = MAX_TRAINING_SIZE

        encoder_embed_tokens = self.pretrained_inverse_model.build_embedding(
            args, inverse_alphabet, args.encoder_embed_dim,
        )
        self.gvp_transformer_encoder = GVPTransformerEncoder(args, inverse_alphabet, encoder_embed_tokens)
        self.inverse_batch_converter = CoordBatchConverter(inverse_alphabet)

        self.padding_token = inverse_alphabet.all_toks[inverse_alphabet.padding_idx]

        self.loss_fn = nn.CosineEmbeddingLoss(reduction="sum")
        self.device = "cuda:0"

    def load_inputs_and_ground_truth(self, batch_data, end=None):
        inverse_batch_converter_input = []
        representations = []

        for data in batch_data:
            sequence = data['sequence']
            padding_size = MAX_TRAINING_SIZE - len(sequence)

            sequence_padded = sequence + self.padding_token * padding_size
            coords = [[[float('inf') if x is None else x for x in atom]
                       for atom in residue] for residue in data['coords']]
            coords_padded = (coords + [[[np.nan] * len(coords[0][0])]
                                       * len(coords[0])] * padding_size)
            inverse_batch_converter_input.append((coords_padded, None, sequence_padded))

            representation = torch.tensor(data['representations'])
            representation_padded = torch.nn.functional.pad(
                representation[:-1],
                (0, 0, 0, padding_size),
                mode="constant", value=0)
            representation_padded = torch.concat([representation_padded,
                                                  torch.unsqueeze(representation[-1],
                                                                  0)], 0)
            representations.append(representation_padded)

        coords, confidence, strs, tokens, padding_mask = self.inverse_batch_converter(
            inverse_batch_converter_input, device=self.device)

        return (coords, padding_mask, confidence), torch.stack(representations, 0)

    def forward(self, inputs):
        coords, padding_mask, confidence = inputs
        encoder_out = self.gvp_transformer_encoder(coords, padding_mask, confidence)
        return encoder_out["encoder_out"][0].transpose(0, 1), padding_mask

    def compute_loss(self, outputs, ground_truth):
        prediction, padding_mask = outputs
        prediction = prediction.reshape(-1, prediction.size(-1))
        ground_truth = ground_truth.reshape(-1, ground_truth.size(-1))

        target = torch.ones(prediction.size(0), device=prediction.device)
        loss = F.cosine_embedding_loss(prediction, ground_truth, target)
        return loss

    def evaluate(self, data):
        ground_truth_sequence = data["sequence"]
        chain_id = data["chain_id"]
        inputs, gt_representations = self.load_inputs_and_ground_truth([data])

        # Get the predicted representation from the model
        self.gvp_transformer_encoder.to(self.device)
        predicted_representations = self(inputs)
        loss = self.compute_loss(predicted_representations, gt_representations).item()
        print(f"Distance in Embedding space is: {loss}")

        # Get the predicted sequence based on the decoder
        predicted_logits = self.pretrained_llm.lm_head(predicted_representations[:len(ground_truth_sequence)])
        predicted_indices = torch.argmax(predicted_logits, dim=-1)
        predicted_sequence = ''.join([self.llm_alphabet.get_tok(i) for i in predicted_indices.squeeze().tolist()])[5:-5]

        # Compare ground truth and predicted sequence directly using vectorized operations
        correct_predictions = sum(a == b for a, b in zip(predicted_sequence, ground_truth_sequence))
        total_predictions = len(ground_truth_sequence)

        # Calculate and print the average recovery rate
        recovery_rate = correct_predictions / total_predictions if total_predictions > 0 else 0
        print(f"Recovery rate for protein: {chain_id} is {recovery_rate}")
        return recovery_rate

    def get_parameter_count(self):
        return sum(p.numel() for p in self.gvp_transformer_encoder.parameters() if p.requires_grad)
