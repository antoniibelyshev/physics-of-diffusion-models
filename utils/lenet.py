import torch
from torch import Tensor, optim
from torch.nn import Module, ModuleList, Linear
from torch.nn.functional import relu, cross_entropy
from torch.utils.data import Dataset, DataLoader
from typing import Callable
from tqdm import tqdm


class LeNet(Module):
    def __init__(self, in_dim: int, out_dim: int, inter_dims: tuple[int, ...] = (300, 100)) -> None:
        super(LeNet, self).__init__()
        cur_dim = in_dim
        self.fc = ModuleList([])
        for dim in inter_dims:
            self.fc.append(Linear(cur_dim, dim))
            cur_dim = dim
        self.fc_last = Linear(cur_dim, out_dim)

    def features(self, x: Tensor) -> Tensor:
        x = x.flatten(1)
        for fc in self.fc:
            x = relu(fc(x))
        return x

    def forward(self, x: Tensor) -> Tensor:
        return self.fc_last(self.features(x)) # type: ignore


def train(
        model: Module,
        train_loader: DataLoader[tuple[Tensor, ...]],
        optimizer: optim.Optimizer,
        criterion: Callable[[Tensor, Tensor], Tensor],
        device: str
) -> None:
    model.train()
    total_loss = 0.
    correct = 0
    for images, labels in tqdm(train_loader, total=len(train_loader)):
        images, labels = images.to(device), labels.to(device)

        outputs = model(images)
        loss = criterion(outputs, labels)

        optimizer.zero_grad()
        loss.backward() # type: ignore
        optimizer.step()

        total_loss += loss.item()
        preds = outputs.argmax(dim=1)
        correct += (preds == labels).sum().item()

    accuracy = correct / len(train_loader.dataset) # type: ignore
    print(f"Train Loss: {total_loss / len(train_loader):.4f}, Train Accuracy: {accuracy:.4f}")


def evaluate(
        model: Module,
        eval_loader: DataLoader[tuple[Tensor, ...]],
        criterion: Callable[[Tensor, Tensor], Tensor],
        device: str,
) -> None:
    model.eval()
    total_loss = 0.
    correct = 0
    with torch.no_grad():
        for images, labels in eval_loader:
            images, labels = images.to(device), labels.to(device)

            outputs = model(images)
            loss = criterion(outputs, labels)

            total_loss += loss.item()
            preds = outputs.argmax(dim=1)
            correct += (preds == labels).sum().item()

    accuracy = correct / len(eval_loader.dataset) # type: ignore
    print(f"Eval Loss: {total_loss / len(eval_loader):.4f}, Eval Accuracy: {accuracy:.4f}")


def train_lenet(
    lenet: LeNet,
    train_dataset: Dataset[tuple[Tensor, Tensor]],
    eval_dataset: Dataset[tuple[Tensor, Tensor]],
    batch_size: int = 64,
    learning_rate: float = 1e-3,
    epochs: int = 10,
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
    criterion: Callable[[Tensor, Tensor], Tensor] = cross_entropy,
) -> LeNet:
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True)
    eval_loader = DataLoader(eval_dataset, batch_size=batch_size, shuffle=False)

    lenet = lenet.to(device)
    optimizer = optim.Adam(lenet.parameters(), lr=learning_rate)

    for epoch in range(1, epochs + 1):
        print(f"Epoch {epoch}/{epochs}")
        train(lenet, train_loader, optimizer, criterion, device)
        evaluate(lenet, eval_loader, criterion, device)

    return lenet.cpu()
