from pathlib import Path
from typing import List
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.transforms import v2
from torch.utils.data import Dataset, DataLoader

from src.neural_net.utils.fen_string_to_target import fen_string_to_target_matrix
from PIL import Image


# Define your Chessboard Dataset
class ChessboardDataset(Dataset):
    def __init__(self, image_path: Path):
        self.image_path = image_path
        self.fen_strings = self._load_fen_strings()
        self.transform = v2.Compose(
            [
                v2.ToImage(),  # Convert to tensor, only needed if you had a PIL image
                v2.ToDtype(torch.uint8, scale=True),
                # ...
                v2.Resize(size=(256, 256), antialias=True),  # Or Resize(antialias=True)
                # ...
                v2.ToDtype(torch.float32, scale=True),  # Normalize expects float input
                v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ]
        )

    def _load_fen_strings(self) -> List[str]:
        fen_strings_path = self.image_path / "_fens.txt"

        with open(str(fen_strings_path), "r") as f:
            return f.readlines()

    def _load_image(self, idx: int):
        image_path = self.image_path / f"{idx}.png"
        image = Image.open(str(image_path))
        return image

    def __len__(self):
        return len(self.fen_strings)

    def __getitem__(self, idx) -> tuple[torch.Tensor, torch.Tensor]:
        image = self._load_image(idx)  # Implement load_image
        fen_string = self.fen_strings[idx]
        image = self.transform(image)

        # Convert FEN string to your target format here
        target = fen_string_to_target_matrix(
            fen_string
        ).flatten()  # Implement fen_string_to_target
        return image, torch.from_numpy(target)
