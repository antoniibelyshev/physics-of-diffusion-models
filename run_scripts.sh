#!/bin/bash

python scripts/train_diffusion.py --diffusion.noise_schedule_type entropy
python scripts/train_diffusion.py --diffusion.noise_schedule_type entropy_u
python scripts/train_diffusion.py --diffusion.noise_schedule_type cosine
python scripts/train_diffusion.py --diffusion.noise_schedule_type linear_beta

