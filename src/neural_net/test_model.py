import os
from pathlib import Path
import matplotlib.pyplot as plt
import torch
from torchvision.transforms import v2, Lambda
from PIL import Image

from src.neural_net.efficient_net_v2_model import get_efficient_net_v2_model
from src.neural_net.utils.piece_to_target import piece_to_index


def remove_alpha(img):
    return img.convert("RGB")


test_image_path = (
    Path(__file__).parent
    / "../state_generator/single_cell_set/training_images/anarcandy"
)

# Get a list of all files in the "anarcandy" folder
image_files = os.listdir(test_image_path)

# Loop over all files
for image_file in image_files:
    test_image = Image.open(test_image_path / image_file)
    test_image = v2.Compose(
        [
            Lambda(remove_alpha),
            v2.ToImage(),
            v2.ToDtype(torch.float32, scale=True),
            v2.Grayscale(num_output_channels=1),
        ]
    )(test_image)

    model = get_efficient_net_v2_model()
    # Run the model on the test image
    model.eval()
    with torch.no_grad():
        pred = model(test_image.unsqueeze(0))

        # Apply softmax to the output to get probabilities
        probabilities = torch.nn.functional.softmax(pred, dim=1)

        # Convert the tensor to a numpy array
        probabilities = probabilities.numpy()

        correct_piece = image_file.split("_")[0]
        # If the prediction was incorrect, plot and show the image
        if (
            pred.argmax(1).item() != piece_to_index[correct_piece]
        ):  # Replace correct_label with the correct label for the image
            pil_image = v2.ToPILImage()(test_image)
            # plot the predicted piece vs the correct piece
            # Lookup the correct piece key by value
            predicted_piece = list(piece_to_index.keys())[pred.argmax(1).item()]
            plt.title(f"Predicted: {predicted_piece}, Correct: {correct_piece}")
            # Show the probabilities as a bar chart in a side by side plot
            plt.subplot(1, 2, 1)
            plt.bar(
                list(piece_to_index.keys()),
                probabilities[0],
                color="blue",
                width=0.4,
            )
            plt.ylim([0, 1])  # Set the limits of the y-axis
            plt.subplot(1, 2, 2)
            plt.imshow(pil_image)
            plt.show()

        # Print the probabilities
        for i, prob in enumerate(probabilities[0]):
            piece = list(piece_to_index.keys())[i]
            print(f"{piece}: {prob*100:.2f}%")
