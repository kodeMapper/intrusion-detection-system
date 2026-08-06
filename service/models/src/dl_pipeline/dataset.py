import torch
from torch.utils.data import Dataset, WeightedRandomSampler
import numpy as np

class FlowSequenceDataset(Dataset):
    """
    Dataset for serving sequence data to the LSTM model.
    Expects pre-built sequence arrays of shape (N, seq_len, features)
    """
    def __init__(self, X: np.ndarray, y: np.ndarray):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.long)
        
    def __len__(self):
        return len(self.X)
        
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

class AutoencoderDataset(Dataset):
    """
    Dataset for serving 2D flow vectors to the Autoencoder model.
    """
    def __init__(self, X: np.ndarray):
        self.X = torch.tensor(X, dtype=torch.float32)
        
    def __len__(self):
        return len(self.X)
        
    def __getitem__(self, idx):
        # Autoencoders map X -> X, so we can return just X or (X, X)
        return self.X[idx], self.X[idx]

def create_weighted_sampler(y: np.ndarray) -> WeightedRandomSampler:
    """
    Calculates class weights based on inverse frequency to construct
    a WeightedRandomSampler. This solves class starvation by oversampling
    minority classes during batch generation.
    """
    class_counts = np.bincount(y)
    # Avoid division by zero
    class_counts[class_counts == 0] = 1 
    
    # Calculate weights as inverse of counts
    class_weights = 1.0 / class_counts
    
    # Map the class weights to every sample in the dataset
    sample_weights = np.array([class_weights[label] for label in y])
    
    sampler = WeightedRandomSampler(
        weights=torch.from_numpy(sample_weights).float(),
        num_samples=len(sample_weights),
        replacement=True
    )
    return sampler
