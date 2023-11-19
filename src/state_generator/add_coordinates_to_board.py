import cv2
import numpy as np
from cv2.typing import MatLike


def add_coordinates_to_board(image: MatLike):
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
                    cv2.FONT_HERSHEY_COMPLEX_SMALL,
                    1,
                    (255, 255, 255) if j % 2 == 0 else (0, 0, 0),
                    2,
                )
            if j == 7:
                cv2.putText(
                    cell,
                    numbers[i],
                    (cell_height - 30, 30),
                    cv2.FONT_HERSHEY_COMPLEX_SMALL,
                    1,
                    (255, 255, 255) if i % 2 == 0 else (0, 0, 0),
                    2,
                )

            # Add the cell to the new image
            new_image[
                i * cell_height : (i + 1) * cell_height,
                j * cell_width : (j + 1) * cell_width,
            ] = cell

    return new_image
