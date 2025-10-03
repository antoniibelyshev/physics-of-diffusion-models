# save_imagenet64_like_fixed.py
import os, math, pickle
from pathlib import Path
from typing import Tuple
import numpy as np
from PIL import Image
from datasets import load_dataset

OUT_DIR = "data/imagenet64"   # <- single consistent output dir

def _to_uint8_rgb_64(img) -> np.ndarray:
    if isinstance(img, np.ndarray):
        if img.dtype != np.uint8:
            x = img
            if x.max() <= 1.0:
                x = (x * 255.0).round()
            img = x.clip(0, 255).astype(np.uint8)
        if img.ndim == 2:
            img = np.repeat(img[..., None], 3, axis=2)
        pil = Image.fromarray(img)
    else:
        pil = img.convert("RGB")
    if pil.size != (64, 64):
        pil = pil.resize((64, 64), Image.BICUBIC)
    return np.array(pil, dtype=np.uint8)  # (64,64,3)

def _flatten_cifar_order(arr_hw3: np.ndarray) -> np.ndarray:
    chw = np.transpose(arr_hw3, (2, 0, 1))
    return chw.reshape(-1).astype(np.uint8)  # (12288,)

def _label_to_1_1000(label_zero_based: int) -> int:
    return int(label_zero_based) + 1

def save_val_data(hf_dataset, out_dir: str = OUT_DIR, label_column: str = "label", image_column: str = "image"):
    out_path = Path(out_dir); out_path.mkdir(parents=True, exist_ok=True)
    fout = out_path / "val_data"

    data_buf, labels_buf = [], []
    for ex in hf_dataset:
        arr = _to_uint8_rgb_64(ex[image_column])
        data_buf.append(_flatten_cifar_order(arr))
        labels_buf.append(_label_to_1_1000(int(ex[label_column])))

    data_np = np.stack(data_buf, axis=0)
    with open(fout, "wb") as f:
        pickle.dump({"data": data_np, "labels": labels_buf}, f, protocol=pickle.HIGHEST_PROTOCOL)
    print(f"Wrote {fout} (N={data_np.shape[0]})")

def save_train_in_10_batches(hf_dataset, out_dir: str = OUT_DIR, label_column: str = "label", image_column: str = "image", num_batches: int = 10) -> Tuple[int, int]:
    out_path = Path(out_dir); out_path.mkdir(parents=True, exist_ok=True)
    N = len(hf_dataset); per = math.ceil(N / num_batches)

    total = 0; last_sz = 0; start = 0
    for bi in range(num_batches):
        stop = min(N, start + per)
        if start >= stop: break
        chunk = hf_dataset.select(range(start, stop))

        data_buf, labels_buf = [], []
        for ex in chunk:
            arr = _to_uint8_rgb_64(ex[image_column])
            data_buf.append(_flatten_cifar_order(arr))
            labels_buf.append(_label_to_1_1000(int(ex[label_column])))

        data_np = np.stack(data_buf, axis=0)
        last_sz = data_np.shape[0]; total += last_sz
        fout = out_path / f"train_data_batch_{bi+1}"
        with open(fout, "wb") as f:
            pickle.dump({"data": data_np, "labels": labels_buf}, f, protocol=pickle.HIGHEST_PROTOCOL)
        print(f"Wrote {fout} (M={last_sz})")
        start = stop
    return total, last_sz

if __name__ == "__main__":
    # Example HF source with 64x64 images already:
    ds_train = load_dataset("benjamin-paine/imagenet-1k-64x64", split="train")
    ds_val   = load_dataset("benjamin-paine/imagenet-1k-64x64", split="validation")

    save_val_data(ds_val, out_dir=OUT_DIR)
    save_train_in_10_batches(ds_train, out_dir=OUT_DIR)

    # Quick listing to confirm files:
    print("Wrote files:", sorted(os.listdir(OUT_DIR)))
