from pathlib import Path
from typing import List
from fentoimage.board import BoardImage
from PIL import Image

from src.state_generator.utils.record_execution_time import record_execution_time


def save_image(image: Image.Image, filename: str):
    output_path = Path(__file__).parent / "output" / "training_images"
    # Make dir if not exist
    output_path.mkdir(parents=True, exist_ok=True)

    # Save as png
    image.save(output_path / f"{filename}.png")


@record_execution_time
def get_training_fens() -> list[str]:
    # Load csv from ./lichess_db_puzzle.csv
    csv_path = Path(__file__).parent / "lichess_db_puzzle.csv"
    with open(csv_path, "r") as f:
        # skip header and get the second column
        fens = [line.split(",")[1] for line in f.readlines()[1:]]
    return fens


@record_execution_time
def save_fens(fens: list[str]):
    output_path = Path(__file__).parent / "output" / "training_images"
    # Make dir if not exist
    output_path.mkdir(parents=True, exist_ok=True)

    with open(output_path / "_fens.txt", "w") as f:
        f.write("\n".join(fens))


@record_execution_time
def generate_training_images(num_images_to_generate: int):
    fens = get_training_fens()
    # for each fen in row, save board image
    for i, fen in enumerate(fens):
        if i >= num_images_to_generate:
            break

        renderer = BoardImage(fen)
        image = renderer.render()

        save_image(image, f"{i}")

    save_fens(fens[:num_images_to_generate])


def __main__():
    num_images_to_generate = 10000
    generate_training_images(num_images_to_generate)


if __name__ == "__main__":
    __main__()
