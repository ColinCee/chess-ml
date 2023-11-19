from pathlib import Path
from src.neural_net.dataset import ChessboardDataset
from src.neural_net.utils.fen_string_to_target import fen_string_to_target_matrix
from src.neural_net.utils.target_to_ascii_board import target_to_ascii_board
from torchvision.transforms.functional import to_pil_image


data = ChessboardDataset()
image, target = data.__getitem__(0)

# Convert the tensor to a PIL Image
image_pil = to_pil_image(image)
image_pil.save(Path(__file__).parent / "test.png")
target_to_ascii_board(target.numpy())
