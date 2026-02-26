#!/bin/bash
CUDA_VISIBLE_DEVICES='1' python infer_FSC.py \
--version ppt-v1 \
--checkpoint_dir /mnt/disk2/hachi/checkpoints/ppt-v1 \
--output_path /mnt/disk2/hachi/Output/test_to_define/FSC_final/FSC_easy_overlap/ctxt-guided_ppt1 \
--data_folder_path /mnt/disk2/hachi/data/FSC_final/FSC_easy_overlap \
--task ctxt-guided
