from utils import save_tensors_as_images, get_data_tensor
from config import with_config, Config


@with_config()
def main(config: Config) -> None:
    save_tensors_as_images(get_data_tensor(config, train=False), "./tmp_fid_test")


if __name__ == "__main__":
    main()
