#!/bin/bash
CUDA_VISIBLE_DEVICES='2' python infer_real.py \
--version ppt-v1 \
--checkpoint_dir /workdir/radish/hachi/checkpoints/ppt-v1 \
--output_path /workdir/radish/hachi/Output/test_to_define/Real/Real_difficult/text-guided \
--data_folder_path /workdir/radish/hachi/OBJ_INS/Real/Real_difficult \
--task text-guided