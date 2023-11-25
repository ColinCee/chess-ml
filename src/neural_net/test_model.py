# load model and test

from pathlib import Path
import torch
from src.neural_net.neural_net import ChessboardFENNet
from src.neural_net.utils.piece_to_target import piece_to_index

model = ChessboardFENNet()
model_path = Path(__file__).parent / "models"
model_name = "2023-11-25_20-37-25_model.pth"
model.load_state_dict(torch.load(f"{model_path}/{model_name}"))

# Load test image and convert to tensor

from torchvision.transforms import v2
from PIL import Image

test_image_path = (
    Path(__file__).parent / "../state_generator/single_cell_set/testing_images/"
)

test_image = Image.open(test_image_path / "b_#86a666.png")
test_image = v2.Compose(
    [
        v2.ToImage(),
        v2.ToDtype(torch.float32, scale=True),
        v2.Grayscale(num_output_channels=1),
    ]
)(test_image)

# Save test image for debugging
debug_output_path = Path(__file__).parent / "debug"
v2.ToPILImage()(test_image).save(f"{debug_output_path}/test_image.png")

# Run the model on the test image

model.eval()
with torch.no_grad():
    pred = model(test_image.unsqueeze(0))
    # print(pred)
    # print(pred.argmax(1))
    # print(pred.argmax(1).item())
    # print(pred.argmax(1).item() == 2)
    # Apply softmax to the output to get probabilities
    probabilities = torch.nn.functional.softmax(pred, dim=1)

    # Convert the tensor to a numpy array
    probabilities = probabilities.numpy()

    # Print the probabilities
    for i, prob in enumerate(probabilities[0]):
        piece = list(piece_to_index.keys())[i]
        print(f"{piece}: {prob*100:.2f}%")
