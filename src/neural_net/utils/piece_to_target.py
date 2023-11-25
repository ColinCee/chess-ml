# target is a one hot encoded matrix of the 12 + 1 empty cell
import torch

piece_to_index = {
    "p": 0,
    "n": 1,
    "b": 2,
    "r": 3,
    "q": 4,
    "k": 5,
    "P": 6,
    "N": 7,
    "B": 8,
    "R": 9,
    "Q": 10,
    "K": 11,
    "empty": 12,
}


def piece_to_target(piece: str) -> torch.Tensor:
    """
    Converts a piece to a one hot encoded matrix
    :param piece: A piece string, e.g. "r"
    :return: A one hot encoded matrix
    """
    # Define the piece to index mapping

    # Return the class index for the piece
    return torch.tensor(piece_to_index[piece], dtype=torch.long)
