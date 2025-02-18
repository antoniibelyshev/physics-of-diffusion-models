import torch
from torch.utils.data import TensorDataset
from utils import get_dataset, LeNet, train_lenet
from config import with_config, Config


@with_config()
def main(config: Config) -> None:
    train_dataset = get_dataset(config)
    eval_dataset = get_dataset(config, train=False)
    lenet = LeNet(train_dataset[0][0].numel(), 10)
    lenet = train_lenet(lenet, train_dataset, eval_dataset)
    torch.save(lenet.state_dict(), f"checkpoints/lenet_{config.data.dataset_name}.pth")


if __name__ == "__main__":
    main()
