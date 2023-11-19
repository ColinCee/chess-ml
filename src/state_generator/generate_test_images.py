from pathlib import Path
from typing import List
from fentoimage.board import BoardImage
from PIL import Image

from src.state_generator.utils.record_execution_time import record_execution_time

testing_image_dir = Path(__file__).parent / "output" / "testing_images"


def save_image(image: Image.Image, filename: str):
    # Make dir if not exist
    testing_image_dir.mkdir(parents=True, exist_ok=True)

    # Save as png
    image.save(testing_image_dir / f"{filename}.png")


@record_execution_time
def get_testing_fens() -> list[str]:
    # Load csv from ./lichess_db_puzzle.csv
    csv_path = Path(__file__).parent / "lichess_db_puzzle.csv"
    # start from the end of the file
    with open(csv_path, "r") as f:
        # skip header and get the second column
        fens = [line.split(",")[1] for line in f.readlines()[::-1][1:]]
    return fens


@record_execution_time
def save_fens(fens: list[str]):
    with open(testing_image_dir / "_fens.txt", "w") as f:
        f.write("\n".join(fens))


@record_execution_time
def generate_test_images(num_images_to_generate: int):
    fens = get_testing_fens()
    # for each fen in row, save board image
    for i, fen in enumerate(fens):
        if i >= num_images_to_generate:
            break

        renderer = BoardImage(fen)
        image = renderer.render()

        save_image(image, f"{i}")

    save_fens(fens[:num_images_to_generate])


def __main__():
    num_images_to_generate = 1000
    generate_test_images(num_images_to_generate)


if __name__ == "__main__":
    __main__()
