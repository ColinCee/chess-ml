# Instantiate the network
from datetime import date, datetime
from pathlib import Path
from src.neural_net.cell_dataset import CellDataset
from src.neural_net.neural_net import ChessboardFENNet
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from src.neural_net.training import test_loop, train_loop
from torch.utils.data import random_split


device = torch.device("cuda:0")
model = ChessboardFENNet()
model.to(device)

# Assuming that we are on a CUDA machine, this should print a CUDA device:
print(device)

# Instantiate the dataset
generator_path = Path(__file__).parent / ".." / "state_generator" / "single_cell_set"
train_dataset = CellDataset((generator_path / "training_images"))
test_dataset = CellDataset((generator_path / "testing_images"))

# Instantiate the DataLoader
batch_size = 4
# Assuming train_dataset is your original dataset
train_size = int(0.8 * len(train_dataset))  # 80% for training
val_size = len(train_dataset) - train_size  # 20% for validation

train_dataset, val_dataset = random_split(train_dataset, [train_size, val_size])

# Now you can create DataLoaders for training and validation
train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
test_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True, num_workers=4)

# Define loss function and optimizer
loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-3)


epochs = 40
for t in range(epochs):
    print(f"Epoch {t+1}\n-------------------------------")
    train_loop(train_dataloader, model, loss_fn, optimizer, device)
    test_loop(test_dataloader, model, loss_fn, device)

print("Done!")

model_path = Path(__file__).parent / "models"
torch.save(
    model.state_dict(),
    f"{model_path}/{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}_model.pth",
)
print("Saved PyTorch Model State to model.pth")
