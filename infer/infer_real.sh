#!/bin/bash
CUDA_VISIBLE_DEVICES='2' python infer_real.py \
--version ppt-v1 \
--checkpoint_dir /mnt/disk2/hachi/checkpoints/ppt-v1 \
--output_path /mnt/disk2/hachi/Output/test_to_define/Real/Real_difficult/ctxt-guided_ppt1 \
--data_folder_path /mnt/disk2/hachi/data/Real/Real_difficult \
--task ctxt-guided
