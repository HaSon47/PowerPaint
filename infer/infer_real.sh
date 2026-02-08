#!/bin/bash
CUDA_VISIBLE_DEVICES='2' python infer_real.py \
--version ppt-v2 \
--checkpoint_dir /mnt/disk2/hachi/checkpoints/ppt-v2 \
--output_path /mnt/disk2/hachi/Output/test_to_define/Real/Real_difficult/ctxt-guided_ppt2 \
--data_folder_path /mnt/disk2/hachi/data/Real/Real_difficult \
--task ctxt-guided