#!/bin/zsh
set -Eeuox pipefail

EPOCH=5
SEED=43
EMB_FILE=../data/vectors.txt
TRAIN_FILE=../data/gaiic_track3_round1_train_20210228.tsv
RATIO=0.95,0.05
VOCAB_SIZE=9445
EMB_SIZE=256
LEARNING_RATE=0.0000001
BATCH_SIZE=4
MAX_LENGTH=64
MAX_GRAD_NORM=10
CHECKPOINT=../checkpoint
SAVE_DIR=../checkpoint


python ../src/train.py --epoch=$EPOCH --seed=$SEED --emb_file=$EMB_FILE --train_file=$TRAIN_FILE --max_grad_norm=$MAX_GRAD_NORM --save_dir=$SAVE_DIR\
    --ratio=$RATIO --vocab_size=$VOCAB_SIZE --emb_size=$EMB_SIZE --learning_rate=$LEARNING_RATE --batch_size=$BATCH_SIZE --max_length=$MAX_LENGTH --checkpoint=$CHECKPOINT

