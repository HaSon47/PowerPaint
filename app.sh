#!/bin/bash
CUDA_VISIBLE_DEVICES='1' python app.py \
--share \
--version ppt-v1 \
--checkpoint_dir /mnt/disk2/hachi/checkpoints/ppt-v1 \