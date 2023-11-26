from io import BytesIO
from math import pi
import os
from pathlib import Path
import time
from PIL import Image
import cairosvg
import drawsvg as draw
from matplotlib.pylab import f
import pyvips


# light_square_color = "#f0d9b5"
# dark_square_color = "#b58863"

light_square_color = "#ffffdd"
dark_square_color = "#86a666"

piece_to_lichess_filename = {
    "b": "bB",
    "k": "bK",
    "n": "bN",
    "p": "bP",
    "q": "bQ",
    "r": "bR",
    "B": "wB",
    "K": "wK",
    "N": "wN",
    "P": "wP",
    "Q": "wQ",
    "R": "wR",
}
pieces = ["b", "k", "n", "p", "q", "r", "B", "K", "N", "P", "Q", "R"]

pieces_path = Path(__file__).parent / f"../../../../lila/public/piece/"



def create_cell_svg(background_color: str, piece_set_name, piece: str | None = None):
    d = draw.Drawing(64, 64, origin=(0, 0))
    d.append(draw.Rectangle(0, 0, 64, 64, fill=background_color))

    piece_size = 64  # Adjust as needed
    piece_padding = (64 - piece_size) / 2
    output_path = Path(__file__).parent / "training_images" / piece_set_name
    # create output folder if it doesn't exist
    output_path.mkdir(parents=True, exist_ok=True)
    
    if piece is None:
        return d.save_png(f"{output_path}/empty_{background_color}.png")

    image = draw.Image(
        piece_padding,
        piece_padding,
        piece_size,
        piece_size,
        f"{pieces_path}/{piece_set_name}/{piece_to_lichess_filename[piece]}.svg",
        embed=True,
    )
    d.append(draw.Use(image, 0, 0))
    
    # save using cairosvg
    svg = d.as_svg()
    if not svg:
        raise Exception("SVG is empty")

    # use libvips to convert svg to png
    image = pyvips.Image.new_from_buffer(svg.encode("utf8"), "", access="sequential")
    image.write_to_file(f"{output_path}/{piece}_{background_color}.png") # type: ignore
        

# get all subfolders folders in pieces_path


# Get a list of all files and directories in the path
all_files_and_dirs = os.listdir(pieces_path)
# Filter out any files, leaving only directories
dirs = [dir for dir in all_files_and_dirs if os.path.isdir(os.path.join(pieces_path, dir))]
# filter out mono
dirs = [dir for dir in dirs if dir != "mono"]

print(dirs)  # Print the list of directories
for piece_set_name in dirs:
    for piece in pieces:
        for color in [light_square_color, dark_square_color]:
            svg = create_cell_svg(color, piece_set_name, piece)

    create_cell_svg(light_square_color, piece_set_name)
    create_cell_svg(dark_square_color, piece_set_name)
