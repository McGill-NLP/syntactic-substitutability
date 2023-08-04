#!/bin/bash

# Name of the eval split
SPLIT="ptb3-wsj-test_10"

# Path to CONLL file
CONLLU_FILE="data/ptb3-wsj-test_10.conllx"

# Number of substitutions per position in sentence, k
NUMBER_SENTS="1"

OUTDIR=./out/${SPLIT}

for number_sents in $NUMBER_SENTS; do
    python generate_substitutions.py $CONLLU_FILE $OUTDIR $number_sents
done
