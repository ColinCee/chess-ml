from pathlib import Path
from fentoimage.board import BoardImage


def save_board_img_from_fen(fen: str, filename: str):
    renderer = BoardImage(fen)
    image = renderer.render()

    output_path = Path(__file__).parent / "output" / "training_images"
    # Make dir if not exist
    output_path.mkdir(parents=True, exist_ok=True)

    # Save pillow image to file
    image.save(str(output_path / f"{filename}.png"))


def get_training_fens() -> list[str]:
    # Load csv from ./lichess_db_puzzle.csv
    csv_path = Path(__file__).parent / "lichess_db_puzzle.csv"
    with open(csv_path, "r") as f:
        # skip header and get the second column
        fens = [line.split(",")[1] for line in f.readlines()[1:]]
    return fens


def generate_training_images(num_images_to_generate: int):
    fens = get_training_fens()
    # for each fen in row, save board image
    for i, fen in enumerate(fens):
        save_board_img_from_fen(fen, str(i))

        if i >= num_images_to_generate:
            break

    # Save fens to a file as mapping from image to fen, maybe use json
    fen_ouput_path = Path(__file__).parent / "output" / "training_images" / "_fens.txt"
    with open(str(fen_ouput_path), "w") as f:
        for fen in fens[:num_images_to_generate]:
            f.write(fen + "\n")


def __main__():
    num_images_to_generate = 1000
    generate_training_images(num_images_to_generate)


if __name__ == "__main__":
    __main__()
