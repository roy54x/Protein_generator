import esm
import torch
from esm.inverse_folding.util import CoordBatchConverter
from torch import nn
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

        self.loss_fn = nn.CosineEmbeddingLoss()
        self.device = "cuda:0"

    def load_inputs_and_ground_truth(self, batch_data, end=None):
        inverse_batch_converter_input = []
        representations = []

        for data in batch_data:
            sequence = data['sequence']
            padding_size = MAX_TRAINING_SIZE - len(sequence)
            sequence_padded = sequence + self.padding_token * padding_size

            coords = [[[float("inf") if x is None else x for x in atom]
                       for atom in residue] for residue in data['coords']]

            coords_padded = (coords + [[[float("inf")] * len(coords[0][0])] * len(coords[0])] * padding_size)
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
        x = encoder_out["encoder_out"][0].transpose(0, 1)
        x = self.linear(x)
        roberta_output = self.roberta_encoder(hidden_states=x,
                                              attention_mask=~padding_mask[:, None, None, :]).last_hidden_state
        return roberta_output, padding_mask

    def compute_loss(self, outputs, ground_truth):
        prediction, padding_mask = outputs
        prediction = prediction.reshape(-1, prediction.size(-1))
        ground_truth = ground_truth.reshape(-1, ground_truth.size(-1))
        padding_mask = padding_mask.reshape(-1)

        filtered_prediction = prediction[~padding_mask]
        filtered_ground_truth = ground_truth[~padding_mask]

        # Create target tensor for cosine embedding loss
        target = torch.ones(filtered_prediction.size(0), device=prediction.device)

        # Compute the cosine embedding loss
        loss = self.loss_fn(filtered_prediction, filtered_ground_truth, target)
        return loss

    def evaluate(self, data):
        ground_truth_sequence = data["sequence"]
        chain_id = data["chain_id"]
        inputs, gt_representations = self.load_inputs_and_ground_truth([data])

        # Get the predicted representation from the model
        gt_representations = gt_representations.to(self.device)
        self.gvp_transformer_encoder = self.gvp_transformer_encoder.to(self.device)
        self.to(device=self.device)
        outputs = self(inputs)
        loss = self.compute_loss(outputs, gt_representations).item()
        print(f"Distance in Embedding space is: {loss}")

        # Get the predicted sequence based on the decoder
        predicted_representations, padding_mask = outputs
        self.lm_head = self.lm_head.to(self.device)
        predicted_logits = self.lm_head(predicted_representations[:, 1:len(ground_truth_sequence) + 1])
        predicted_indices = torch.argmax(predicted_logits, dim=-1)
        predicted_sequence = ''.join([self.llm_alphabet.get_tok(i) for i in predicted_indices.squeeze().tolist()])

        # Compare ground truth and predicted sequence directly using vectorized operations
        correct_predictions = sum(a == b for a, b in zip(predicted_sequence, ground_truth_sequence))
        total_predictions = len(ground_truth_sequence)

        # Calculate and print the average recovery rate
        recovery_rate = correct_predictions / total_predictions if total_predictions > 0 else 0
        print(f"Recovery rate for protein: {chain_id} is {recovery_rate}")
        return loss, recovery_rate
