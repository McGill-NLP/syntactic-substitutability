#!/bin/bash

SPLIT='en_pud-ud-test'
CONLL_FILE=''
OUTF="resources/$SPLIT"
NUMBER_SENTS=1
BERT_PATH=""

python3 head-ensembles/conllu2json.py $CONLL_FILE "${OUTF}.json"
python3 attention-analysis-clark-etal/extract_attention_substitutions.py --preprocessed-data-file "${OUTF}.json" --bert-dir $BERT_PATH --max-sequence-length 512 --num-subs $NUMBER_SENTS --out-path "${OUTF}.json"

#RUN THE LINE BELOW IF FIRST TIME ON THIS SPLIT
#python3 attention-analysis-clark-etal/extract_attention.py --preprocessed-data-file "${OUTF}.json" --bert-dir $BERT_PATH --max-sequence-length 512

python3 attention-analysis-clark-etal/convert_sub_to_json.py "${OUTF}_${NUMBER_SENTS}.json"
python3 head-ensembles/json2conllu.py "${OUTF}_${NUMBER_SENTS}_listed.json" "${OUTF}_${NUMBER_SENTS}_listed.conllu"