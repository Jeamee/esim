#!/bin/zsh
set -Eeuox pipefail

EPOCH=5
SEED=43
EMB_FILE=../data/vectors.txt
TRAIN_FILE=../data/gaiic_track3_round1_train_20210228.tsv
RATIO=0.97,0.03
VOCAB_SIZE=10000
EMB_SIZE=300
LEARNING_RATE=0.0001
BATCH_SIZE=2
MAX_LENGTH=12
CHECKPOINT=../checkpoint

python ../src/train.py --epoch=$EPOCH --seed=$SEED --emb_file=$EMB_FILE --train_file=$TRAIN_FILE \
    --ratio=$RATIO --vocab_size=$VOCAB_SIZE --emb_size=$EMB_SIZE --learning_rate=$LEARNING_RATE --batch_size=$BATCH_SIZE --max_length=$MAX_LENGTH --checkpoint=$CHECKPOINT

