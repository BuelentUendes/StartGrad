import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm

import torch.nn
from torch.utils.data import Dataset
from utils.general.helper_path import MODELS_PATH
from utils.general.helper import create_directory

FEATURE_DIM = 3
DEVICE = "cuda:1" if torch.cuda.is_available() else "cpu"


class TimeSeriesDataset(Dataset):

    def __init__(self, x, y):
        self.x = x
        self.y = y

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]


def get_test_accuracy(model, test_loader, verbose=False, mortality=False, device=DEVICE):
    test_accuracy = []
    model.eval()
    for batch in test_loader:
        x, y = batch
        y_hat = model.forward(x.to(device))
        if mortality:
            y_hat_softmax = F.softmax(y_hat, dim=1).cpu()
            predictions = torch.argmax(y_hat_softmax, dim=1)
            # predictions = torch.argmax(F.softmax(y_hat, dim=-1).cpu(), dim=1)
        else:
            predictions = (F.sigmoid(y_hat).cpu() >= 0.5)

        accuracy = (predictions == y.cpu()).float().mean().item()
        test_accuracy.append(accuracy)

    total_test_accuracy = torch.round(torch.tensor(test_accuracy).mean(), decimals=4)
    if verbose:
        tqdm.write(f"test_accuracy: {total_test_accuracy}")

    return total_test_accuracy


def train_model(model, train_loader, test_loader, lr=1e-4, lr2=1e-3, epochs=50, verbose=False, mortality=False,
                save_best_results=False, target_path=MODELS_PATH,
                device=DEVICE):
    loss_fn = nn.CrossEntropyLoss() if mortality else nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=lr2)

    total_epoch_losses = []
    total_epoch_accuracies = []
    best_accuracy = 0.0

    for epoch in tqdm(range(epochs)):
        epoch_loss = []
        epoch_accuracy = []
        model.train().to(device)
        for batch in train_loader:
            x, y = batch
            y_hat = model.forward(x.to(device))

            if isinstance(loss_fn, nn.CrossEntropyLoss):
                # Cross entropy expects it to be in the long format
                y = y.long()

            loss = loss_fn(y_hat, y.to(device))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss.append(loss.cpu().item())
            if mortality:
                y_hat_softmax = F.softmax(y_hat, dim=1).cpu()
                predictions = torch.argmax(y_hat_softmax, dim=1)
                # predictions = torch.argmax(F.softmax(y_hat, dim=-1).cpu(), dim=1)
            else:
                predictions = (F.sigmoid(y_hat).cpu() >= 0.5)

            accuracy = (predictions == y.cpu()).float().mean().item()

            epoch_accuracy.append(accuracy)

        epoch_loss = torch.round(torch.tensor(epoch_loss).mean(), decimals=4)
        epoch_accuracy = torch.round(torch.tensor(epoch_accuracy).mean(), decimals=4)

        total_epoch_losses.append(epoch_loss)
        total_epoch_accuracies.append(epoch_accuracy)
        if verbose:
            tqdm.write(f"\nepoch:{epoch}: \ntrain_accuracy: {epoch_accuracy} "
                       f"\ntrain_loss: {epoch_loss}")

        test_accuracy = get_test_accuracy(model, test_loader, verbose, mortality=mortality, device=device)

        if save_best_results:
            if test_accuracy > best_accuracy:
                best_accuracy = test_accuracy
                best_model_state = model.state_dict()  # Save the model's state dictionary

                create_directory(target_path)
                # You can save the best model state to a file, for example:
                torch.save(best_model_state, os.path.join(target_path, 'best_model.pth'))
                print(f"Model saved with accuracy: {best_accuracy:.4f}")

    # load the best model states
    if save_best_results:
        model.load_state_dict(torch.load(os.path.join(target_path, "best_model.pth")))

    return model
