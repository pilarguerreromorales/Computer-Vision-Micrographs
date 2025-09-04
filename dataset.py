import base64
import io
import zlib
import numpy as np
import torch
from torch.utils.data import Dataset
import torchvision.transforms.v2 as transforms
from typing import Optional, Tuple


def decode_array(encoded_base64_str):
    """
    Utility function to extract the image for the string of bytes that is in the csv file
    """
    decoded = base64.b64decode(encoded_base64_str)
    decompressed = zlib.decompress(decoded)
    return np.load(io.BytesIO(decompressed))


def encode_array(array):
    """
    Utility function to convert a numpy array into a string of bytes to be stored in the csv file
    """
    bytes_io = io.BytesIO()
    np.save(bytes_io, array, allow_pickle=False)
    compressed = zlib.compress(bytes_io.getvalue(), level=9)
    return base64.b64encode(compressed).decode('utf-8')


class BaseMicrographDataset(Dataset):
    """Base class for micrograph datasets implementing common functionality"""

    def __init__(self, df, window_size: int):
        """
        Initialize base dataset

        Args:
            df: Pandas DataFrame containing image data
            window_size: Size of the image window/crop
        """
        self.df = df
        self.window_size = window_size

    def __len__(self) -> int:
        return len(self.df)


    def load_and_normalize_image(self, encoded_image: str) -> torch.Tensor:
      image = decode_array(encoded_image).astype(np.float32)

      # Normalize using global mean and std
      mean, std = 0.004292, 0.075022 # Replace with computed values
      image = (image - mean) / std

      if len(image.shape) == 2:
          image = image[np.newaxis, ...]
      return torch.from_numpy(image)


    def load_mask(self, encoded_mask: str) -> torch.Tensor:
      mask = decode_array(encoded_mask).astype(np.float32)

      # Normalize mask to [0, 1]
      mask = mask / mask.max() if mask.max() > 0 else mask

      if len(mask.shape) == 2:
          mask = mask[np.newaxis, ...]
      return torch.from_numpy(mask)

    def pad_to_min_size(self, image: torch.Tensor, min_size: int) -> Tuple[torch.Tensor, Tuple[int, int]]:
        """Pad image to minimum size if needed"""
        _, h, w = image.shape
        pad_h = max(0, min_size - h)
        pad_w = max(0, min_size - w)
        padded = torch.nn.functional.pad(image, (0, pad_w, 0, pad_h), mode="reflect")
        return padded, (pad_h, pad_w)

    def load_raw_image(self, encoded_image: str) -> torch.Tensor:
        """Load raw image without normalization."""
        image = decode_array(encoded_image).astype(np.float32)
        if len(image.shape) == 2:
            image = image[np.newaxis, ...]
        return torch.from_numpy(image)


class TrainMicrographDataset(BaseMicrographDataset):
    """Dataset for training with random augmentations"""

    def __init__(self, df, window_size: int):
        super().__init__(df, window_size)

        # Define training-specific transforms
        self.shared_transform = transforms.Compose([
            transforms.RandomCrop(window_size),
            transforms.RandomVerticalFlip(),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(degrees=(-10, 10))
        ])


    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        row = self.df.iloc[idx]

        # Load and preprocess image
        image = self.load_and_normalize_image(row['image'])
        image, _ = self.pad_to_min_size(image, self.window_size)

        # Load and preprocess mask
        mask = self.load_mask(row['mask'])
        mask, _ = self.pad_to_min_size(mask, self.window_size)

        # Apply shared transforms to both image and mask
        stacked = torch.cat([image, mask], dim=0)
        stacked = self.shared_transform(stacked)
        image, mask = torch.split(stacked, [1, 1], dim=0)

        return image, mask


class ValidationMicrographDataset(BaseMicrographDataset):
    """Dataset for validation using corner crops. This is a good idea because the regions of interest can be
        at the edges of the image"""

    def __init__(self, df, window_size: int):
        super().__init__(df, window_size)
        # Define 5 fixed crops: 4 corners + center
        self.n_crops = 5

    def __len__(self) -> int:
        return len(self.df) * self.n_crops

    def get_crop_coordinates(self, image_shape: Tuple[int, int], crop_idx: int) -> Tuple[int, int]:
        """Get coordinates for specific crop index"""
        h, w = image_shape

        if crop_idx == 4:  # Center crop
            h_start = (h - self.window_size) // 2
            w_start = (w - self.window_size) // 2
        else:
            h_start = 0 if crop_idx < 2 else h - self.window_size
            w_start = 0 if crop_idx % 2 == 0 else w - self.window_size

        return h_start, w_start

    def crop_tensors(self, image: torch.Tensor, mask: torch.Tensor,
                     h_start: int, w_start: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """Extract a crop from both image and mask"""
        h_end = h_start + self.window_size
        w_end = w_start + self.window_size

        return (
            image[:, h_start:h_end, w_start:w_end],
            mask[:, h_start:h_end, w_start:w_end]
        )

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        image_idx = idx // self.n_crops
        crop_idx = idx % self.n_crops
        row = self.df.iloc[image_idx]

        # Load and preprocess image and mask
        image = self.load_and_normalize_image(row['image'])
        image, _ = self.pad_to_min_size(image, self.window_size)

        mask = self.load_mask(row['mask'])
        mask, _ = self.pad_to_min_size(mask, self.window_size)

        # Get specific corner/center crop
        h_start, w_start = self.get_crop_coordinates(image.shape[1:], crop_idx)
        image, mask = self.crop_tensors(image, mask, h_start, w_start)

        return image, mask


class InferenceMicrographDataset(BaseMicrographDataset):
    """Dataset for inference without any augmentations"""

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, str, Tuple[int, int]]:
        row = self.df.iloc[idx]

        # Load and preprocess image
        image = self.load_and_normalize_image(row['image'])
        image, padding = self.pad_to_min_size(image, self.window_size)

        return image, row['Id'], padding