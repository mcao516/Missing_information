#!/bin/bash

# 1. Create your environement locally
source ~/env37/bin/activate

python mutual_info_makring.py \
    --source_path  /home/mcao610/scratch/summarization/XSum/fairseq_files/train.source \
    --target_path  /home/mcao610/scratch/summarization/XSum/fairseq_files/train.target \
    --ent_path /home/mcao610/Missing_information/Build_dataset/xsum_train_ents.json \
    --bart_path /home/mcao610/scratch/BART_models/xsum_cedar_cmlm \
    --checkpoint_file checkpoint_best.pt \
    --data_name_or_path  /home/mcao610/scratch/summarization/XSum/fairseq_files/xsum-bin \
    --start_pos 70000 --end_pos 90000 \
    --output_file posterior_70000_90000;
