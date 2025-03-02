#!/bin/bash

python scripts/train_diffusion.py --diffusion.noise_schedule_type entropy_u --data.dataset_name cifar10
python scripts/train_diffusion.py --diffusion.noise_schedule_type cosine --data.dataset_name cifar10
python scripts/train_diffusion.py --diffusion.noise_schedule_type linear_beta --data.dataset_name cifar10

python scripts/compute_fid.py --data.dataset_name cifar10
