"""
PyTorch Dataset for feasibility prediction.

Loads collected data and provides samples for training.
"""

import torch
from torch.utils.data import Dataset, DataLoader
import h5py
import numpy as np
from typing import Optional, Tuple, Dict


class FeasibilityDataset(Dataset):
    """
    PyTorch Dataset for feasibility classification.
    
    Supports both state-vector and image-based inputs.
    """
    
    def __init__(
        self,
        data_path: str,
        use_images: bool = False,
        normalize: bool = True,
        transform: Optional[callable] = None
    ):
        """
        Initialize the dataset.
        
        Args:
            data_path: Path to HDF5 dataset file
            use_images: Whether to include images in samples
            normalize: Whether to normalize state/action vectors
            transform: Optional transform for images
        """
        self.data_path = data_path
        self.use_images = use_images
        self.normalize = normalize
        self.transform = transform
        
        # Load dataset into memory
        with h5py.File(data_path, 'r') as f:
            self.state_vectors = f['state_vectors'][:]
            self.action_vectors = f['action_vectors'][:]
            self.feasible = f['feasible'][:]
            
            if use_images:
                self.images = f['images'][:]
            else:
                self.images = None
                
            self.num_samples = f.attrs['num_samples']
            
        # Compute normalization statistics
        if normalize:
            self.state_mean = self.state_vectors.mean(axis=0)
            self.state_std = self.state_vectors.std(axis=0) + 1e-8
            self.action_mean = self.action_vectors.mean(axis=0)
            self.action_std = self.action_vectors.std(axis=0) + 1e-8
        else:
            self.state_mean = 0
            self.state_std = 1
            self.action_mean = 0
            self.action_std = 1
            
    def __len__(self) -> int:
        return self.num_samples
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        Get a sample.
        
        Returns:
            Dict with:
                - 'state': Normalized state vector
                - 'action': Normalized action vector
                - 'input': Concatenated state+action for simple architectures
                - 'image': RGB image (if use_images=True)
                - 'label': Feasibility label (0 or 1)
        """
        state = self.state_vectors[idx]
        action = self.action_vectors[idx]
        label = self.feasible[idx]
        
        # Normalize
        if self.normalize:
            state = (state - self.state_mean) / self.state_std
            action = (action - self.action_mean) / self.action_std
            
        # Convert to tensors
        state = torch.FloatTensor(state)
        action = torch.FloatTensor(action)
        label = torch.FloatTensor([float(label)])
        
        # Concatenate for MLP input
        combined = torch.cat([state, action])
        
        sample = {
            'state': state,
            'action': action,
            'input': combined,
            'label': label
        }
        
        if self.use_images and self.images is not None:
            image = self.images[idx]
            # Convert to float and normalize to [0, 1]
            image = torch.FloatTensor(image).permute(2, 0, 1) / 255.0
            if self.transform:
                image = self.transform(image)
            sample['image'] = image
            
        return sample
    
    def get_input_dims(self) -> Tuple[int, int, int]:
        """
        Get input dimensions.
        
        Returns:
            (state_dim, action_dim, combined_dim)
        """
        state_dim = self.state_vectors.shape[1]
        action_dim = self.action_vectors.shape[1]
        return state_dim, action_dim, state_dim + action_dim
    
    def get_class_weights(self) -> torch.Tensor:
        """
        Compute class weights for imbalanced data.
        
        Returns:
            Tensor of class weights [neg_weight, pos_weight]
        """
        n_pos = self.feasible.sum()
        n_neg = len(self.feasible) - n_pos
        
        # Inverse frequency weighting
        pos_weight = len(self.feasible) / (2 * n_pos + 1e-8)
        neg_weight = len(self.feasible) / (2 * n_neg + 1e-8)
        
        return torch.FloatTensor([neg_weight, pos_weight])
    
    def split(
        self,
        train_ratio: float = 0.8,
        seed: Optional[int] = None
    ) -> Tuple['FeasibilityDataset', 'FeasibilityDataset']:
        """
        Split dataset into train and validation sets.
        
        Args:
            train_ratio: Fraction of data for training
            seed: Random seed for reproducibility
            
        Returns:
            (train_dataset, val_dataset)
        """
        n_train = int(len(self) * train_ratio)
        
        if seed is not None:
            np.random.seed(seed)
            
        indices = np.random.permutation(len(self))
        train_indices = indices[:n_train]
        val_indices = indices[n_train:]
        
        # Create subset datasets
        train_dataset = FeasibilitySubset(self, train_indices)
        val_dataset = FeasibilitySubset(self, val_indices)
        
        return train_dataset, val_dataset


class FeasibilitySubset(Dataset):
    """Subset of FeasibilityDataset for train/val split."""
    
    def __init__(self, dataset: FeasibilityDataset, indices: np.ndarray):
        self.dataset = dataset
        self.indices = indices
        
    def __len__(self) -> int:
        return len(self.indices)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        return self.dataset[self.indices[idx]]
    
    def get_input_dims(self) -> Tuple[int, int, int]:
        return self.dataset.get_input_dims()
    
    def get_class_weights(self) -> torch.Tensor:
        return self.dataset.get_class_weights()


def create_dataloaders(
    data_path: str,
    batch_size: int = 64,
    train_ratio: float = 0.8,
    use_images: bool = False,
    num_workers: int = 4,
    seed: Optional[int] = None
) -> Tuple[DataLoader, DataLoader]:
    """
    Create train and validation data loaders.
    
    Args:
        data_path: Path to HDF5 dataset
        batch_size: Batch size
        train_ratio: Fraction for training
        use_images: Include images in batches
        num_workers: Number of data loading workers
        seed: Random seed
        
    Returns:
        (train_loader, val_loader)
    """
    dataset = FeasibilityDataset(data_path, use_images=use_images)
    train_dataset, val_dataset = dataset.split(train_ratio, seed)
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    return train_loader, val_loader
