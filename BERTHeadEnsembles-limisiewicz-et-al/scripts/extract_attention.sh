#!/bin/bash
PREFIXPROCESS=$1
PREFIXEVAL=$2
BERTDIR=$3
RESOURCESDIR='../resources'


PROCESSFILE=$RESOURCESDIR/$PREFIXPROCESS
EVALFILE=$RESOURCESDIR/$PREFIXEVAL


# prepare attention matrices
python3 ../head-ensembles/conllu2json.py "${PROCESSFILE}.conllu" "${PROCESSFILE}.json"
python3 ../attention-analysis-clark-etal/extract_attention.py --preprocessed-data-file "${PROCESSFILE}.json" --bert-dir $BERTDIR --max-sequence-length 512

python3 ../head-ensembles/conllu2json.py "${EVALFILE}.conllu" "${EVALFILE}.json"
python3 ../attention-analysis-clark-etal/extract_attention.py --preprocessed-data-file "${EVALFILE}.json" --bert-dir $BERTDIR --max-sequence-length 512

rm "${PROCESSFILE}.json"
rm "${EVALFILE}.json"