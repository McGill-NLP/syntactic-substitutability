#!/bin/bash

# Name of the eval split
SPLIT="obj_rel_sampled"

# Path to CONLL file
CONLLU_FILE='data/obj_rel_sampled_new.conllx'

# Number of substitutions per position in sentence, k
NUMBER_SENTS="1"

OUTDIR=./out/${SPLIT}

for number_sents in $NUMBER_SENTS; do
    python generate_substitutions.py $CONLLU_FILE $OUTDIR $number_sents
done
