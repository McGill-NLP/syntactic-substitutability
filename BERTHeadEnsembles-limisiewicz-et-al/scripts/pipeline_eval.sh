#!/bin/bash
PREFIXPROCESS=$1
PREFIXEVAL=$2
RESOURCESDIR='../resources'
RESULTDIR='../results'


PROCESSFILE=$RESOURCESDIR/$PREFIXPROCESS
EVALFILE=$RESOURCESDIR/$PREFIXEVAL
RESULTFILE=$RESULTDIR/$PREFIXEVAL

# convert conllus
udapy read.Conllu files="${PROCESSFILE}.conllu" ud.AttentionConvert write.Conllu > "${PROCESSFILE}-conv.conllu"
#
udapy read.Conllu files="${EVALFILE}.conllu" ud.AttentionConvert write.Conllu > "${EVALFILE}-conv.conllu"

# select heads
python3  ../head-ensembles/head_ensemble.py "${PROCESSFILE}_attentions.npz" "${PROCESSFILE}_source.txt" "${PROCESSFILE}-conv.conllu" -j "${PROCESSFILE}_head-ensembles.json" --report-result "${RESULTDIR}/${PREFIXPROCESS}.dep_acc"

#evaluate
python3  ../head-ensembles/head_ensemble.py "${EVALFILE}_attentions.npz" "${EVALFILE}_source.txt" "${EVALFILE}-conv.conllu" -j "${PROCESSFILE}_head-ensembles.json" --report-result "${RESULTFILE}.dep_acc"
python3 ../head-ensembles/extract_trees.py "${EVALFILE}_attentions.npz" "${EVALFILE}_source.txt" "${EVALFILE}.conllu" "${PROCESSFILE}_head-ensembles.json" --report-result "${RESULTFILE}.trees"

# baseline
python3  ../head-ensembles/positional_baseline.py "${PROCESSFILE}-conv.conllu" -j "${PROCESSFILE}_offsets.json" --report-result "${RESULTDIR}/${PREFIXPROCESS}.posistional_basline"
python3  ../head-ensembles/positional_baseline.py  "${EVALFILE}-conv.conllu" -j "${PROCESSFILE}_offsets.json" --report-result "${RESULTFILE}.posistional_basline" -e

# optional cleanup
rm "${PROCESSFILE}-conv.conllu"
rm "${PROCESSFILE}_offsets.json"

rm "${EVALFILE}-conv.conllu"
