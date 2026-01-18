#!/bin/bash
CUDA_VISIBLE_DEVICES='1' python infer_test_to_define.py \
--version ppt-v1 \
--checkpoint_dir /workdir/radish/hachi/checkpoints/ppt-v1 \
--output_path /workdir/radish/hachi/Output/test_to_define/FSC_final/FSC_easy_no_overlap/text-guided \
--data_folder_path /workdir/radish/hachi/OBJ_INS/FSC_final/FSC_easy_no_overlap \
--task text-guided