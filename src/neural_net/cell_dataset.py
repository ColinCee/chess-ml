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
                v2.RandomResizedCrop(64, scale=(0.8, 1.0), ratio=(1, 1)),
            ]
        )
        self.image_path = image_path
        self.dataset = self._load_dataset()

    def _load_dataset(self):
        dataset = []
        # for each subdir in the image path
        for subdir in self.image_path.glob("*"):
            for file in subdir.glob("*.png"):
                piece_name = file.name.split("_")[0]
                image = Image.open(file)
                # Convert the image to RGB if it has more than 3 channels
                if len(image.split()) > 3:
                    image = image.convert("RGB")
                    
                image = self.transform(image)
                # save to debug
                debug_output_path = Path(__file__).parent / "debug"
                v2.ToPILImage()(image).save(f"{debug_output_path}/{piece_name}.png")
                dataset.append({"image": image, "target": piece_to_target(piece_name)})

        return dataset

    def __len__(self):
        print("Size of dataset is:", len(self.dataset))
        return len(self.dataset)

    def __getitem__(self, idx) -> tuple[torch.Tensor, torch.Tensor]:
        # Get the image and target
        image = self.dataset[idx]["image"]
        target = self.dataset[idx]["target"]

        return image, target
