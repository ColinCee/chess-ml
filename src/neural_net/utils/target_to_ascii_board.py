from numpy import ndarray
from src.neural_net.utils.fen_string_to_target import piece_to_index


def target_to_ascii_board(target: ndarray):
    # Initialize an empty board
    target = target.reshape(8, 8, 13)
    board = [["." for _ in range(8)] for _ in range(8)]

    # Iterate over the rows and columns
    for row in range(8):
        for col in range(8):
            # Iterate over the pieces
            for piece in range(13):
                # Check if the piece is present
                if target[row, col, piece]:
                    # Get the piece character
                    piece_char = list(piece_to_index.keys())[
                        list(piece_to_index.values()).index(piece)
                    ]
                    if piece_char == "1":
                        piece_char = "."

                    # Add the piece to the board
                    board[row][col] = piece_char

    # Print the board
    for row in board:
        print(" ".join(row))
