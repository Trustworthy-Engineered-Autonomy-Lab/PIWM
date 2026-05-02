import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
import torchvision.transforms as transforms

class DonkeyDataset(Dataset):
    def __init__(self, data_dir, transform=None, use_noisy=False, noise_level='025'):
        """
        Args:
            data_dir: Directory containing .npz files
            transform: Optional transform to be applied on images
            use_noisy: Whether to use noisy states instead of clean states
            noise_level: '025', '050', or '100' for different noise levels
        """
        self.data_dir = Path(data_dir)
        self.transform = transform
        self.use_noisy = use_noisy
        self.noise_level = noise_level

        # Get all npz files
        self.files = list(self.data_dir.glob("*.npz"))

        # Load all data into memory for faster training
        self.images = []
        self.states = []

        print(f"Loading data from {data_dir}...")
        for file_path in self.files:
            data = np.load(file_path)

            # Get images (normalize to [0,1])
            imgs = data['imgs'].astype(np.float32) / 255.0

            # Get states
            if use_noisy:
                states = data[f'states_noisy_{noise_level}'].astype(np.float32)
            else:
                states = data['states'].astype(np.float32)

            self.images.append(imgs)
            self.states.append(states)

        # Concatenate all data
        if self.images:
            self.images = np.concatenate(self.images, axis=0)
            self.states = np.concatenate(self.states, axis=0)
        else:
            raise ValueError(f"No data found in directory {data_dir}")

        print(f"Loaded {len(self.images)} images with shape {self.images.shape}")
        print(f"Loaded {len(self.states)} states with shape {self.states.shape}")

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image = self.images[idx]
        state = self.states[idx]

        # Convert to tensor and change from HWC to CHW format
        image = torch.from_numpy(image).permute(2, 0, 1)  # (H, W, C) -> (C, H, W)
        state = torch.from_numpy(state)

        if self.transform:
            image = self.transform(image)

        return image, state

def get_dataloader(data_dir, batch_size=32, shuffle=True, num_workers=0, use_noisy=False, noise_level='025'):
    """
    Create a DataLoader for the donkey dataset
    """
    # Define transforms
    transform = transforms.Compose([
        # Images are already normalized to [0,1] in the dataset
        # Add any additional transforms here if needed
    ])

    dataset = DonkeyDataset(
        data_dir=data_dir,
        transform=transform,
        use_noisy=use_noisy,
        noise_level=noise_level
    )

    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True if torch.cuda.is_available() else False
    )

    return dataloader

def create_fold_dataloaders(base_dir, fold_num, batch_size=32, num_workers=0, use_noisy=False, noise_level='025'):
    """
    Create train and validation dataloaders for k-fold cross validation

    Args:
        base_dir: Base directory containing fold_1, fold_2, etc.
        fold_num: Which fold to use as validation (1-5)
        batch_size: Batch size for training
        num_workers: Number of worker processes for data loading
        use_noisy: Whether to use noisy states
        noise_level: Noise level for states

    Returns:
        train_loader, val_loader
    """
    base_path = Path(base_dir)

    # Collect all fold directories
    all_folds = [f"fold_{i}" for i in range(1, 6)]
    val_fold = f"fold_{fold_num}"
    train_folds = [f for f in all_folds if f != val_fold]

    print(f"Using {val_fold} for validation")
    print(f"Using {train_folds} for training")

    # Create temporary combined training directory (in memory)
    train_files = []
    for fold in train_folds:
        fold_path = base_path / fold
        if fold_path.exists():
            train_files.extend(list(fold_path.glob("*.npz")))

    val_path = base_path / val_fold

    # Create custom dataset that can handle multiple files
    class MultiDirDataset(DonkeyDataset):
        def __init__(self, file_list, transform=None, use_noisy=False, noise_level='025'):
            self.files = file_list
            self.transform = transform
            self.use_noisy = use_noisy
            self.noise_level = noise_level

            self.images = []
            self.states = []

            print(f"Loading training data from {len(file_list)} files...")
            for file_path in self.files:
                data = np.load(file_path)

                imgs = data['imgs'].astype(np.float32) / 255.0

                if use_noisy:
                    states = data[f'states_noisy_{noise_level}'].astype(np.float32)
                else:
                    states = data['states'].astype(np.float32)

                self.images.append(imgs)
                self.states.append(states)

            if self.images:
                self.images = np.concatenate(self.images, axis=0)
                self.states = np.concatenate(self.states, axis=0)
            else:
                raise ValueError(f"No training data found in files: {[str(f) for f in file_list]}")

            print(f"Training set: {len(self.images)} images")

    # Create datasets
    train_dataset = MultiDirDataset(
        train_files,
        use_noisy=use_noisy,
        noise_level=noise_level
    )

    val_dataset = DonkeyDataset(
        val_path,
        use_noisy=use_noisy,
        noise_level=noise_level
    )

    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True if torch.cuda.is_available() else False
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True if torch.cuda.is_available() else False
    )

    return train_loader, val_loader