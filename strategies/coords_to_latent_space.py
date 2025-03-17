import esm
import numpy as np
import torch
import torch.nn.functional as F
from esm.inverse_folding.gvp_transformer_encoder import GVPTransformerEncoder
from esm.inverse_folding.util import CoordBatchConverter
from torch import nn, optim
from transformers import RobertaConfig, RobertaModel

from constants import MAX_TRAINING_SIZE
from strategies.base import Base


class CoordsToLatentSpace(Base):

    def __init__(self):
        super(CoordsToLatentSpace, self).__init__()
        pretrained_llm, self.llm_alphabet = esm.pretrained.esm2_t33_650M_UR50D()
        self.lm_head = pretrained_llm.lm_head
        for name, param in self.lm_head.named_parameters():
            param.requires_grad = False

        pretrained_inverse_model, inverse_alphabet = esm.pretrained.esm_if1_gvp4_t16_142M_UR50()
        self.gvp_transformer_encoder = pretrained_inverse_model.encoder
        for name, param in self.gvp_transformer_encoder.named_parameters():
            param.requires_grad = False
        self.inverse_batch_converter = CoordBatchConverter(inverse_alphabet)
        self.padding_token = inverse_alphabet.all_toks[inverse_alphabet.padding_idx]

        self.linear = nn.Linear(512, 1280)
        roberta_config = RobertaConfig(
            hidden_size=1280,
            num_hidden_layers=2,
            num_attention_heads=4,
            intermediate_size=512
        )
        roberta = RobertaModel(roberta_config)
        self.roberta_encoder = roberta.encoder

        self.softmax = nn.Softmax(dim=-1)
        self.cross_entropy = nn.CrossEntropyLoss()
        self.cosine_loss = nn.CosineEmbeddingLoss()
        self.cosine_coefficient = 10

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
        extra_value = torch.full((*tokens.shape[:-1], 1), self.llm_alphabet.padding_idx,
                                 dtype=tokens.dtype, device=tokens.device)
        tokens = torch.cat([tokens, extra_value], dim=-1)

        return (coords, padding_mask, confidence), (torch.stack(representations, 0), tokens)

    def forward(self, inputs):
        coords, padding_mask, confidence = inputs
        encoder_out = self.gvp_transformer_encoder(coords, padding_mask, confidence)
        x = encoder_out["encoder_out"][0].transpose(0, 1)
        x = self.linear(x)
        roberta_output = self.roberta_encoder(hidden_states=x,
                                              attention_mask=~padding_mask[:, None, None, :]).last_hidden_state
        return roberta_output, padding_mask

    def compute_loss(self, outputs, ground_truth):
        prediction, padding_mask = outputs
        ground_truth_representations, ground_truth_tokens = ground_truth
        logits = self.softmax(self.lm_head(prediction))
        num_classes = logits.shape[-1]  # Get the number of classes
        ground_truth_tokens = torch.clamp(ground_truth_tokens, min=0, max=num_classes - 1)

        padding_mask[:, 0] = True
        padding_mask[:, -1] = True
        prediction = prediction.reshape(-1, prediction.size(-1))
        padding_mask = padding_mask.reshape(-1)
        ground_truth_representations = (ground_truth_representations.
                                        reshape(-1, ground_truth_representations.size(-1)))
        ground_truth_tokens = ground_truth_tokens.reshape(-1)
        logits = logits.reshape(-1, logits.size(-1))

        filtered_prediction = prediction[~padding_mask]
        filtered_representations = ground_truth_representations[~padding_mask]
        filtered_tokens = ground_truth_tokens[~padding_mask]
        filtered_logits = logits[~padding_mask]

        # Compute the cosine embedding loss
        target = torch.ones(filtered_prediction.size(0), device=prediction.device)
        cosine_loss = self.cosine_loss(filtered_prediction, filtered_representations, target)

        cross_entropy_loss = self.cross_entropy(filtered_logits, filtered_tokens)

        return self.cosine_coefficient * cosine_loss + cross_entropy_loss

    def evaluate(self, batch_data):
        # Load batched inputs and ground truths
        inputs, ground_truth = batch_data
        ground_truth_representations, ground_truth_tokens = ground_truth

        # Get the predicted representation from the model
        self.gvp_transformer_encoder = self.gvp_transformer_encoder.to(self.device)
        self.lm_head = self.lm_head.to(self.device)
        self.to(device=self.device)
        outputs = self(inputs)
        prediction, padding_mask = outputs
        logits = self.softmax(self.lm_head(prediction))
        predicted_indices = torch.argmax(logits, dim=-1)

        padding_mask[:, 0] = True
        padding_mask[:, -1] = True

        per_sequence_recovery = []
        for i in range(predicted_indices.shape[0]):
            valid_mask = ~padding_mask[i]
            filtered_tokens = ground_truth_tokens[i][valid_mask]
            filtered_indices = predicted_indices[i][valid_mask]

            if len(filtered_tokens) > 0:
                recovery = (filtered_indices == filtered_tokens).float().mean()
                per_sequence_recovery.append(recovery)

        # Compute mean across sequences
        recovery_rate = torch.stack(per_sequence_recovery).mean() \
            if per_sequence_recovery else torch.tensor(0.0)

        print(f"Recovery rate: {recovery_rate.item():.4f}")

        return recovery_rate.item()
