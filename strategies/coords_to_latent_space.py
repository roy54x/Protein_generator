import esm
import numpy as np
import torch
import torch.nn.functional as F
from esm.inverse_folding.gvp_transformer_encoder import GVPTransformerEncoder
from esm.inverse_folding.util import CoordBatchConverter
from torch import nn

from constants import MAX_TRAINING_SIZE, AMINO_ACIDS
from strategies.base import Base

from ProRefiner.model.model import Model


class CoordsToLatentSpace(Base):

    def __init__(self):
        super(CoordsToLatentSpace, self).__init__()
        pretrained_llm, self.llm_alphabet = esm.pretrained.esm2_t33_650M_UR50D()
        self.lm_head = pretrained_llm.lm_head

        class Args:
            in_dim = 1280
            hidden_dim = 1280
            trans_layers = 6
            th = 0.5
            seq_noise = 0.1
            backbone_noise = 0.1
            dropout = 0.1
            drop_edge = 0
            gvp = True
            h_attend = False

        args = Args()
        k_neighbors = 30
        self.inverse_model = Model(args, k_neighbors)

        self.loss_fn = nn.CosineEmbeddingLoss()
        self.device = "cuda:0"

    def load_inputs_and_ground_truth(self, batch_data, end=None):
        """ Pack and pad batch into torch tensors """
        B = len(batch_data)
        lengths = np.array([min(len(b['sequence']), MAX_TRAINING_SIZE) for b in batch_data], dtype=np.int32)
        L_max = max([min(len(b['sequence']), MAX_TRAINING_SIZE) for b in batch_data])
        X = np.zeros([B, L_max, 4, 3])
        S = np.zeros([B, L_max], dtype=np.int32)
        residue_idx = -100 * np.ones([B, L_max], dtype=np.int32)
        chain_encoding_all = np.zeros([B, L_max], dtype=np.int32)

        # Build the batch
        for i, data in enumerate(batch_data):
            x = data["coords"]
            l = len(data['sequence'])
            x_pad = np.pad(x, [[0, L_max - l], [0, 0], [0, 0]], 'constant', constant_values=(np.nan,))
            X[i, :, :, :] = x_pad
            residue_idx[i, 0: l] = np.arange(0, l)
            chain_encoding_all[i, 0: l] = np.ones(l)
            indices = np.asarray([AMINO_ACIDS.index(a) for a in data['sequence']], dtype=np.int32)
            S[i, :l] = indices

        # Mask
        isnan = np.isnan(X)
        mask = np.isfinite(np.sum(X, (2, 3))).astype(np.float32)
        X[isnan] = 0.

        # Conversion
        S = torch.from_numpy(S).to(dtype=torch.long, device=self.device)
        X = torch.from_numpy(X).to(dtype=torch.float32, device=self.device)
        mask = torch.from_numpy(mask).to(dtype=torch.float32, device=self.device)
        residue_idx = torch.from_numpy(residue_idx).to(dtype=torch.long, device=self.device)
        chain_encoding_all = torch.from_numpy(chain_encoding_all).to(dtype=torch.long, device=self.device)
        lengths = torch.from_numpy(lengths).to(dtype=torch.long, device=self.device)

        # Process representations separately
        representations = []
        for data in batch_data:
            representation = torch.tensor(data['representations'])
            padding_size = max(lengths.cpu().numpy()) - representation.size(0)
            representation_padded = torch.nn.functional.pad(
                representation[:-1],
                (0, 0, 0, padding_size),
                mode="constant", value=0)
            representation_padded = torch.concat([representation_padded,
                                                  torch.unsqueeze(representation[-1],
                                                                  0)], 0)
            representations.append(representation_padded)

        # Return features from get_features and stacked representations
        return (X, S, mask, residue_idx, chain_encoding_all), torch.stack(representations, 0)

    def forward(self, inputs):
        X, S, mask, residue_idx, chain_encoding_all = inputs
        logits, features = self.inverse_model(X, S, mask, residue_idx, chain_encoding_all, feat=True)
        return features, mask

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
        return recovery_rate

    def get_parameter_count(self):
        return sum(p.numel() for p in self.inverse_model.parameters() if p.requires_grad)
