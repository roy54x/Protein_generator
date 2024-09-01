import os
import time
import warnings
from datetime import datetime

import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset

from constants import MIN_SIZE, MAIN_DIR, AMINO_ACIDS, MAX_SIZE, NUM_SAMPLES_IN_DATAFRAME, BATCH_SIZE
from strategies.sequence_to_distogram import SequenceToDistogram


class CustomDataset(Dataset):
    def __init__(self, dataframe, strategy):
        self.dataframe = dataframe
        self.strategy = strategy

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, idx):
        row = self.dataframe.iloc[idx]
        inputs, ground_truth = self.strategy.load_inputs_and_ground_truth(row)
        return inputs, ground_truth


class Trainer:
    def __init__(self, directory, strategy, batch_size=32, test_size=0.2, device="cuda:0", pretrained_model_path=""):
        self.directory = directory
        self.strategy = strategy.to(device)
        self.pretrained_model_path = pretrained_model_path
        if os.path.exists(self.pretrained_model_path):
            model = torch.load(pretrained_model_path)
            self.strategy.load_state_dict(model)
        else:
            warnings.warn("Pretrained model path does not exist. Skipping")
            self.pretrained_model_path = None

        self.batch_size = batch_size
        self.test_size = test_size
        self.device = device
        self.optimizer = torch.optim.Adam(strategy.parameters(), lr=0.001)
        self.best_test_loss = float('inf')

        # Collect all file paths from the directory
        self.file_paths = [os.path.join(directory, fname) for fname in os.listdir(directory) if fname.endswith('.json')]
        self.train_files, self.test_files = train_test_split(self.file_paths, test_size=test_size, random_state=42,
                                                             shuffle=False)
        self.train_size = len(self.train_files) * NUM_SAMPLES_IN_DATAFRAME
        self.test_size = len(self.test_files) * NUM_SAMPLES_IN_DATAFRAME
        print(f"number of samples in the train set are: {self.train_size}")
        print(f"number of samples in the test set are: {self.test_size}")
        print(f'Number of trainable parameters: '
              f'{sum(p.numel() for p in self.strategy.parameters() if p.requires_grad)}')

    def get_dataloader(self, file_path, mode):
        dataframe = pd.read_json(file_path, lines=True)
        dataframe = dataframe[dataframe['sequence'].apply(lambda seq: len(seq) >= MIN_SIZE)]
        dataframe = dataframe[dataframe['sequence'].apply(lambda seq: len(seq) <= MAX_SIZE)]
        dataframe = dataframe[dataframe['sequence'].apply(lambda seq: all(char in AMINO_ACIDS for char in seq))]
        return DataLoader(CustomDataset(dataframe, self.strategy), batch_size=self.batch_size,
                          shuffle=(mode == "train"))

    def train(self, epochs=100):
        for epoch in range(epochs):
            batch_count = 0
            total_train_loss = 0
            total_train_samples = 0
            start_time = time.time()
            self.strategy.train()

            for train_file in self.train_files:
                train_loader = self.get_dataloader(train_file, mode="train")

                for inputs, ground_truth in train_loader:
                        self.optimizer.zero_grad()
                        outputs = self.strategy((x.to(self.device) for x in inputs))
                        loss = self.strategy.compute_loss(outputs, ground_truth.to(self.device))
                        loss.backward()
                        self.optimizer.step()

                        total_train_loss += loss.item()
                        total_train_samples += len(inputs)
                        batch_count += 1

                        # Print every 100 batches
                        if batch_count % 100 == 0:
                            avg_train_loss = total_train_loss / total_train_samples
                            elapsed_time = time.time() - start_time
                            print(f'Epoch {epoch + 1}, Batch {batch_count} of {self.train_size//self.batch_size}, '
                                  f'Training Loss: {avg_train_loss:.4f}, '
                                  f'Time taken: {elapsed_time:.4f} seconds.')
                            total_train_loss = 0
                            total_train_samples = 0
                            start_time = time.time()

            # Evaluate on test data
            self.strategy.eval()
            total_test_loss = 0
            total_test_samples = 0
            with torch.no_grad():

                for test_file in self.test_files:
                    test_loader = self.get_dataloader(test_file, mode="test")

                    for inputs, ground_truth in test_loader:
                        outputs = self.strategy((x.to(self.device) for x in inputs))
                        loss = self.strategy.compute_loss(outputs, ground_truth.to(self.device))
                        total_test_loss += loss.item()
                        total_test_samples += len(inputs)

            average_test_loss = total_test_loss / total_test_samples
            print(f'Epoch {epoch + 1}, Test Loss: {average_test_loss:.4f}')

            # Save the model if the test loss is the best seen so far
            if average_test_loss < self.best_test_loss:
                self.best_test_loss = average_test_loss
                self.save_model()

    def save_model(self):
        if self.pretrained_model_path:
            directory = os.path.dirname(self.pretrained_model_path)
            model_path = os.path.join(directory, 'pretrained.pth')
        else:
            directory = os.path.join(MAIN_DIR, "models", self.strategy.__class__.__name__,
                                 datetime.now().strftime("%Y%m%d"))
            if not os.path.exists(directory):
                os.makedirs(directory)
            model_path = os.path.join(directory, 'best_model.pth')

        torch.save(self.strategy.state_dict(), model_path)
        print(f'Model saved at {model_path}')


if __name__ == '__main__':
    data_path = os.path.join(MAIN_DIR,"pdb_data_130000")
    strategy = SequenceToDistogram()
    trainer = Trainer(data_path, strategy, batch_size=BATCH_SIZE, test_size=0.15)
    trainer.train()
