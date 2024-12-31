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
        self.pretrained_llm, self.llm_alphabet = esm.pretrained.esm2_t30_150M_UR50D()
        self.pretrained_llm.eval().cuda()
        self.batch_converter = self.llm_alphabet.get_batch_converter()
        self.device = next(self.pretrained_llm.parameters()).device

        self.pretrained_inverse_model, inverse_alphabet = esm.pretrained.esm_if1_gvp4_t16_142M_UR50()

        args = self.pretrained_inverse_model.args
        args.encoder_embed_dim = 640
        args.encoder_ffn_embed_dim = 256
        args.encoder_layers = 4
        args.encoder_attention_heads = 8
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

    def load_inputs_and_ground_truth(self, batch_data, end=None):
        inverse_batch_converter_input = []
        batch_converter_input = []

        for data in batch_data:
            sequence = data['sequence']
            coords = [[[float('inf') if x is None else x for x in atom]
                       for atom in residue] for residue in data['coords']]
            batch_converter_input.append((data['chain_id'], sequence))
            inverse_batch_converter_input.append((coords, None, sequence))

        batch_labels, batch_strs, batch_tokens = self.batch_converter(batch_converter_input)
        batch_tokens = batch_tokens.to(self.device)
        result = self.pretrained_llm(batch_tokens, repr_layers=[30])
        representations = result["representations"][30]

        coords, confidence, strs, tokens, padding_mask = self.inverse_batch_converter(
            inverse_batch_converter_input, device=self.device)

        return (coords, padding_mask, confidence), representations

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
        self.gvp_transformer_encoder.to(self.device)
        predicted_representations = self(inputs)
        loss = self.compute_loss(predicted_representations, gt_representations).item()
        print(f"Distance in Embedding space is: {loss}")

        # Get the predicted sequence based on the decoder
        predicted_logits = self.pretrained_llm.lm_head(predicted_representations)
        predicted_indices = torch.argmax(predicted_logits, dim=-1)
        predicted_sequence = ''.join([self.llm_alphabet.get_tok(i) for i in predicted_indices.squeeze().tolist()])

        # Compare ground truth and predicted sequence directly using vectorized operations
        correct_predictions = sum(a == b for a, b in zip(predicted_sequence, ground_truth_sequence))
        total_predictions = len(ground_truth_sequence)

        # Calculate and print the average recovery rate
        recovery_rate = correct_predictions / total_predictions if total_predictions > 0 else 0
        print(f"Recovery rate for protein: {chain_id} is {recovery_rate}")
        return recovery_rate
