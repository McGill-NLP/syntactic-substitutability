#!/bin/bash

# Name of the eval split
SPLIT=""

# Path to CONLL file
CONLLU_FILE=""

# Number of substitutions per position in sentence, k
NUMBER_SENTS="1 3 5 10"

OUTDIR=./out/${SPLIT}

for number_sents in $NUMBER_SENTS; do
    SENTENCE_FILE=${OUTDIR}/sent_substitutions_${number_sents}.pkl
    python parse_eval.py $SENTENCE_FILE $OUTDIR $SPLIT $number_sents $CONLLU_FILE
done

