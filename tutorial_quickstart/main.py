import torch
from torch import nn
from neural_net import NeuralNetwork
from train import train
from test_nn import test
from training_data import get_training_data, visualize_dataset


def main():
    train_dataloader, test_dataloader, train_data, test_data = get_training_data()
    visualize_dataset(train_data)
    for X, y in test_dataloader:
        print(f"Shape of X [N, C, H, W]: {X.shape}")
        print(f"Shape of y: {y.shape} {y.dtype}")
        break

    # Get cpu, gpu or mps device for training.
    device = "cuda"

    print(f"Using {device} device")
    model = NeuralNetwork().to(device)
    print(model)

    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=1e-3)

    epochs = 5
    for t in range(epochs):
        print(f"Epoch {t+1}\n-------------------------------")
        train(train_dataloader, model, loss_fn, optimizer, device)
        test(test_dataloader, model, loss_fn, device)
    print("Done!")

    torch.save(model.state_dict(), "tutorial_quickstart/model.pth")
    print("Saved PyTorch Model State to model.pth")


if __name__ == "__main__":
    main()
