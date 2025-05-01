#!/bin/bash

python scripts/compute_fid.py --dataset_name=lsun-bedrooms
python scripts/compute_fid.py --dataset_name=celeba-hq-256-30k
