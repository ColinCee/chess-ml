import cv2
from pathlib import Path
import numpy as np

# Get the path of the current script
script_location = Path(__file__).absolute().parent

board_theme_name = "blue3"
# Define the relative path to the image
image_path = script_location / "board-assets" / f"{board_theme_name}.jpg"

# Load the image
image = cv2.imread(str(image_path))


# Get the size of the image
height, width, _ = image.shape
# Calculate the size of each cell
cell_height = height // 8
cell_width = width // 8

# Define the letters and numbers for the chessboard
letters = ["a", "b", "c", "d", "e", "f", "g", "h"]
numbers = ["8", "7", "6", "5", "4", "3", "2", "1"]

# Create an empty array to hold the new image
new_image = np.zeros_like(image)

# Use a nested loop to iterate over the cells
for i in range(8):
    for j in range(8):
        # Extract each cell
        cell = image[
            i * cell_height : (i + 1) * cell_height,
            j * cell_width : (j + 1) * cell_width,
        ]

        # Check if the cell is in the bottom row or the rightmost column
        if i == 7:
            cv2.putText(
                cell,
                letters[j],
                (10, cell_height - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0, 0, 0),
                2,
            )
        if j == 7:
            cv2.putText(
                cell,
                numbers[i],
                (cell_height - 30, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0, 0, 0),
                2,
            )

        # Add the cell to the new image
        new_image[
            i * cell_height : (i + 1) * cell_height,
            j * cell_width : (j + 1) * cell_width,
        ] = cell

# Define the output directory
output_dir = script_location / "output"
# Create the output directory if it doesn't exist
output_dir.mkdir(exist_ok=True)

# Save the new image to a file
cv2.imwrite(str(output_dir / f"{board_theme_name}.jpg"), new_image)
