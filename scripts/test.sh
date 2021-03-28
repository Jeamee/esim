#!/bin/sh
set -Eeoxu

SEED=43
EMB_FILE=../data/vectors.txt
MODEL=../data/t3.pt
TEST_FILE=../data/gaiic_track3_round1_test_20210228.tsv
BATCH_SIZE=2
MAX_LENGTH=15
OUTPUT_FILE=../data/t3.result.txt

python ../src/test.py --seed=$SEED --emb_file=$EMB_FILE --model=$MODEL --test_file=$TEST_FILE --batch_size=$BATCH_SIZE\
    --max_length=$MAX_LENGTH --output_file=$OUTPUT_FILE

