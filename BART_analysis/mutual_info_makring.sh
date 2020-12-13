#!/bin/bash

# 1. Create your environement locally
module load anaconda/3
source activate py37

python mutual_info_makring.py \
    --source_path  /home/mila/c/caomeng/Downloads/summarization/XSum/fairseq_files/train.source \
    --target_path  /home/mila/c/caomeng/Downloads/summarization/XSum/fairseq_files/train.target \
    --ent_path /home/mila/c/caomeng/Missing_information/Build_dataset/xsum_train_ents.json \
    --bart_path /home/mila/c/caomeng/fairseq/checkpoints/xsum_cedar_cmlm \
    --checkpoint_file checkpoint_best.pt \
    --data_name_or_path  /home/mila/c/caomeng/Downloads/summarization/XSum/fairseq_files/xsum-bin \
    --start_pos 170000 --end_pos 203575 \
    --output_file posterior_170000_203575;