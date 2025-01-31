import torch
from torch.utils.data import TensorDataset
from utils import get_data_tensor, get_labels_tensor, LeNet, train_lenet
from config import with_config, Config


@with_config()
def main(config: Config) -> None:
    train_dataset = TensorDataset(get_data_tensor(config), get_labels_tensor(config))
    eval_dataset = TensorDataset(get_data_tensor(config, train=False), get_labels_tensor(config, train=False))
    lenet = LeNet(train_dataset[0][0].numel(), 10)
    lenet = train_lenet(lenet, train_dataset, eval_dataset)
    torch.save(lenet.state_dict(), f"checkpoints/lenet_{config.data.dataset_name}.pth")


if __name__ == "__main__":
    main()
