from PIL import Image
from torchvision import transforms as v2
from pathlib import Path

# Define your transformations
transform = v2.Compose(
    [
        v2.ToTensor(),
        v2.Resize(size=(128, 128)),
        v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        v2.Grayscale(num_output_channels=1),
    ]
)

# Define your image paths
training_image_path = (
    Path(__file__).parent
    / ".."
    / ".."
    / "state_generator"
    / "output"
    / "training_images"
)
transformed_image_path = (
    Path(__file__).parent / "training_images"
)  # Replace with your desired path

# Create the directory to save transformed images if it doesn't exist
transformed_image_path.mkdir(parents=True, exist_ok=True)

# Loop over all images in the directory
for img_file in training_image_path.glob("*.png"):
    # Open the image file
    img = Image.open(img_file)

    # Apply the transformations
    transformed_img = transform(img)

    # Save the transformed image
    transformed_img_file = transformed_image_path / img_file.name
    v2.ToPILImage()(transformed_img).save(transformed_img_file)
