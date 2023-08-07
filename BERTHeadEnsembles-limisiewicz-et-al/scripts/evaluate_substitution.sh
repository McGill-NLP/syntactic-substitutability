#!/bin/bash

SELECTION_SPLIT='en_pud-ud-test'
EVAL_SPLIT='en_pud-ud-test'
OUTF="resources/$SELECTION_SPLIT"
OUTF_EVAL="resources/$EVAL_SPLIT"
NUMBER_SENTS=1
ORIGINAL_CONLL=""


python3 head-ensembles/normalize_substitutions.py "${OUTF}_${NUMBER_SENTS}_attentions.npz" "${OUTF}_${NUMBER_SENTS}_source.txt" "${OUTF}_${NUMBER_SENTS}_listed.conllu" "${OUTF}_${NUMBER_SENTS}.json"
python3 head-ensembles/normalize_substitutions.py "${OUTF_EVAL}_${NUMBER_SENTS}_attentions.npz" "${OUTF_EVAL}_${NUMBER_SENTS}_source.txt" "${OUTF_EVAL}_${NUMBER_SENTS}_listed.conllu" "${OUTF_EVAL}_${NUMBER_SENTS}.json"

python3 head-ensembles/head_ensemble_substitutions.py "${OUTF}_${NUMBER_SENTS}_attentions.combined.npz" "${OUTF}_${NUMBER_SENTS}_source.txt" $ORIGINAL_CONLL -j "${OUTF}_${NUMBER_SENTS}_head-ensembles.json" --report-result "results/${SELECTION_SPLIT}.tree.uuas.dep_acc"
python3 head-ensembles/extract_trees_substitutions.py "${OUTF_EVAL}_${NUMBER_SENTS}_attentions.combined.npz" "${OUTF_EVAL}_source.txt" $ORIGINAL_CONLL "${OUTF}_${NUMBER_SENTS}_head-ensembles.json" --report-result "results/${EVAL_SPLIT}.tree.uuas"


