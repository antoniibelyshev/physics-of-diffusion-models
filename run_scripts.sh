#!/bin/bash

python scripts/compute_stats_forward.py
python scripts/compute_stats_forward_unbiased.py
python scripts/train_diffusion.py
python scripts/compute_fid.py
