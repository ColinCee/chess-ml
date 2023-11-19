import torch
import numpy as np

piece_to_index = {
    "p": 0,
    "n": 1,
    "b": 2,
    "r": 3,
    "q": 4,
    "k": 5,  # Black pieces
    "P": 6,
    "N": 7,
    "B": 8,
    "R": 9,
    "Q": 10,
    "K": 11,  # White pieces
    "1": 12,  # Empty space
}


def fen_string_to_target_matrix(fen_string: str):
    # Split the FEN string and get the piece placement data
    placement = fen_string.split(" ")[0]

    # Initialize a 64x13 matrix of zeros with boolean type
    target = np.zeros((64, 13), dtype=np.bool_)

    # Row and column trackers for the board
    row, col = 0, 0

    for char in placement:
        if char == "/":  # Move to the next row
            row += 1
            col = 0
        elif char.isdigit():  # Empty squares
            target[row * 8 + col : row * 8 + col + int(char), 12] = True
            col += int(char)  # Increment column by the number of empty squares
        else:  # Piece
            target[row * 8 + col, piece_to_index[char]] = True
            col += 1

    # Flatten the matrix to match the network output shape
    return target
