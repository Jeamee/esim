#!/bin/bash
set -Eeoxu

SEED=43
EMB_FILE=../data/vectors.txt
MODEL=../data/models/esim.pt
TEST_FILE=../data/gaiic_track3_round1_testA_20210228.tsv
BATCH_SIZE=16
MAX_LENGTH=64
OUTPUT_FILE=../data/t3.result.txt
EMB_SIZE=256

python ../src/test.py --seed=$SEED --emb_size=$EMB_SIZE --emb_file=$EMB_FILE --model=$MODEL --test_file=$TEST_FILE --batch_size=$BATCH_SIZE\
    --max_length=$MAX_LENGTH --output_file=$OUTPUT_FILE

