from pathlib import Path
import torch
from torchvision.transforms import v2
from torch.utils.data import Dataset

from PIL import Image

from src.neural_net.utils.piece_to_target import piece_to_target


# Define your Chessboard Dataset
class CellDataset(Dataset):
    def __init__(self, image_path: Path):
        self.transform = v2.Compose(
            [
                v2.ToImage(),
                v2.ToDtype(torch.float32, scale=True),
                v2.Grayscale(num_output_channels=1),
            ]
        )
        self.image_path = image_path
        self.dataset = self._load_dataset()

    def _load_dataset(self):
        dataset = []
        for file in self.image_path.glob("*.png"):
            piece_name = file.name.split("_")[0]
            image = Image.open(file)

            image = self.transform(image)
            dataset.append({"image": image, "target": piece_to_target(piece_name)})

        return dataset

    def __len__(self):
        # 6 pieces, 2 colors, 2 sides
        # Plus 2 for each empty background
        return 6 * 2 * 2 + 2

    def __getitem__(self, idx) -> tuple[torch.Tensor, torch.Tensor]:
        # Get the image and target
        image = self.dataset[idx]["image"]
        target = self.dataset[idx]["target"]

        return image, target
