#!/bin/bash

python mutual_info_makring.py \
    --source_path /home/ml/users/cadencao/XSum/fairseq_files/train.source \
    --target_path /home/ml/users/cadencao/XSum/fairseq_files/train.target \
    --ent_path /home/ml/users/cadencao/Missing-Info/Build_dataset/xsum_train_ents.json \
    --bart_path /home/ml/users/cadencao/fairseq/checkpoints/xsum_cmlm_bos \
    --checkpoint_file checkpoint_best.pt \
    --data_name_or_path /home/ml/users/cadencao/XSum/fairseq_files/xsum-bin;