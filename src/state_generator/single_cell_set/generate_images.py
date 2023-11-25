from io import BytesIO
from pathlib import Path
from PIL import Image
import cairosvg
import drawsvg as draw


light_square_color = "#f0d9b5"
dark_square_color = "#b58863"

pieces = ["b", "k", "n", "p", "q", "r", "B", "K", "N", "P", "Q", "R"]

pieces_path = Path(__file__).parent / "../piece-sets/lichess-standard/"
output_path = Path(__file__).parent / "training_images"


def create_cell_svg(background_color: str, piece: str):
    d = draw.Drawing(64, 64, origin=(0, 0))
    d.append(draw.Rectangle(0, 0, 64, 64, fill=background_color))

    piece_size = 44  # Adjust as needed
    piece_padding = (64 - piece_size) / 2
    d.append(
        draw.Image(
            piece_padding,
            piece_padding,
            piece_size,
            piece_size,
            f"{pieces_path}/{piece}.svg",
            embed=True,
        )
    )

    return d.as_svg()


for piece in pieces:
    for color in [light_square_color, dark_square_color]:
        svg = create_cell_svg(color, piece)
        png = cairosvg.svg2png(bytestring=svg)
        if not png:
            raise Exception("png is empty")

        image = Image.open(BytesIO(png))
        image.save(f"{output_path}/{piece}_{color}.png")
