import copy

import numpy as np
import scipy as sp
import torch
import torch.nn.functional as F
import transformers
from torch import nn
from torch_geometric.nn import GATConv, GCNConv
from torch_geometric.utils import from_scipy_sparse_matrix

from constants import AMINO_ACIDS, MAX_TRAINING_SIZE, BATCH_SIZE
from strategies.base import Base
from utils.padding_functions import padd_sequence, padd_contact_map
from utils.structure_utils import get_distogram, get_contact_map


class ContactMapToSequence(Base):

    def __init__(self):
        super(ContactMapToSequence, self).__init__()
        self.vocab_size = len(AMINO_ACIDS) + 1
        self.hidden_size = 180
        self.num_layers = 6
        self.num_heads = 6
        self.graph_layers = torch.nn.ModuleList([
            GATConv(self.vocab_size, self.hidden_size, heads=self.num_heads)]
             + [GATConv(self.hidden_size * self.num_heads, self.hidden_size, heads=self.num_heads) for _ in range(self.num_layers - 1)])
        self.linear1 = nn.Linear(self.hidden_size * self.num_heads, self.hidden_size)
        self.linear2 = nn.Linear(self.hidden_size, self.vocab_size)

    def load_inputs_and_ground_truth(self, batch_data, end=None):
        input_tensors, edge_indices, mask_tensors, ground_truths = [], [], [], []

        for data in batch_data:
            sequence = data['sequence']
            ca_coords = [coord[1] for coord in data["coords"]]

            # Determine start and end indices for augmentation or slicing
            if self.training:
                start, end = self.get_augmentation_indices(len(sequence))
            elif end:
                start, end = max(0, end - MAX_TRAINING_SIZE), end
            else:
                start, end = 0, MAX_TRAINING_SIZE

            # Slice sequence and create sequence tensor and mask tensor
            sequence = sequence[start:end]
            sequence_tensor, mask_tensor = padd_sequence(sequence, MAX_TRAINING_SIZE)

            # Prepare ground truth
            ground_truth = copy.deepcopy(sequence_tensor[len(sequence) - 1]).to(torch.long)
            ground_truth = F.one_hot(ground_truth, num_classes=self.vocab_size).float()

            # Prepare inputs
            input_tensor = F.one_hot(sequence_tensor.to(torch.long), num_classes=self.vocab_size)
            input_tensor[len(sequence) - 1] = 0
            input_tensor = input_tensor.to(torch.float32)

            # Prepare contact map and edge index
            contact_map = get_contact_map(ca_coords)
            contact_map = contact_map[start:end, start:end]
            contact_map = padd_contact_map(contact_map, MAX_TRAINING_SIZE)
            contact_map = sp.sparse.csr_matrix(contact_map)
            edge_index, _ = from_scipy_sparse_matrix(contact_map)

            # Append processed data to batch lists
            input_tensors.append(input_tensor)
            edge_indices.append(edge_index)
            mask_tensors.append(mask_tensor)
            ground_truths.append(ground_truth)

        # Collate batched data
        input_tensors = torch.stack(input_tensors, dim=0)  # Shape: (batch_size, max_size, vocab_size)
        mask_tensors = torch.stack(mask_tensors, dim=0)  # Shape: (batch_size, max_size)
        ground_truths = torch.stack(ground_truths, dim=0)  # Shape: (batch_size, vocab_size)

        return (input_tensors, edge_indices, mask_tensors), ground_truths

    def collate(self, batch):
        inputs_list, ground_truth_list = zip(*batch)
        input_tensors, edge_indices, mask_tensors = zip(*inputs_list)

        input_tensors = torch.cat(input_tensors, dim=0)

        edge_index_list = []
        batch_offsets = []
        total_nodes = 0
        batch_index_tensor = []

        for batch_idx, (edge_index, input_tensor) in enumerate(zip(edge_indices, input_tensors)):
            batch_offset = total_nodes
            batch_offsets.append(batch_offset)

            adjusted_edge_index = edge_index + batch_offset
            edge_index_list.append(adjusted_edge_index)

            num_nodes = input_tensor.size(0)
            batch_index_tensor.append(torch.full((num_nodes,), batch_idx, dtype=torch.long))

            total_nodes += num_nodes

        edge_index = torch.cat(edge_index_list, dim=1)
        mask_tensors = torch.stack(mask_tensors, dim=0)
        ground_truth = torch.stack(ground_truth_list, dim=0)
        batch_index_tensor = torch.cat(batch_index_tensor)

        return (input_tensors, edge_index, mask_tensors), ground_truth

    def forward(self, inputs):
        x, edge_index, mask_tensor = inputs

        for layer_idx, graph_layer in enumerate(self.graph_layers):
            x = graph_layer(x=x, edge_index=edge_index)
            x = F.leaky_relu(x)

        x = x.view((mask_tensor.size(0), MAX_TRAINING_SIZE, self.hidden_size * self.num_heads))

        last_indices = mask_tensor.argmin(dim=1) - 1
        x = x[torch.arange(x.size(0)), last_indices]

        x = self.linear1(x)
        x = F.leaky_relu(x)
        x = self.linear2(x)

        probabilities = F.softmax(x, dim=-1)
        return probabilities

    def compute_loss(self, outputs, ground_truth):
        return F.cross_entropy(outputs, ground_truth)

    def evaluate(self, data):
        ground_truth_sequence = data["sequence"]

        for idx in range(3):
            print("real values: " + str(ground_truth_sequence[idx]) + ", predicted values: " + str(
                ground_truth_sequence[idx]))

        for idx in range(3, len(ground_truth_sequence) - 1):
            (x, edge_index, mask_tensor), _ = self.load_inputs_and_ground_truth(
                data, idx + 1)

            mask_tensor = mask_tensor.unsqueeze(0)
            output = self.forward((x, edge_index, mask_tensor))
            predicted_values = AMINO_ACIDS[torch.argmax(output).item() - 1]

            print("real values: " + str(ground_truth_sequence[idx]) + ", predicted values: " + str(
                predicted_values))
