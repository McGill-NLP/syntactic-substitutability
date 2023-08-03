#!/bin/bash
#SBATCH --partition=main-cpu
#SBATCH --cpus-per-task=6
#SBATCH --mem=64G                        # Ask for 32 GB of CPU RAM
#SBATCH --time=1:00:00                   # The job will run for 12 hours

ORIGINAL_CONLL="resources/en_pud-ud-test.conllu"
CONVERTED_CONLL="resources/en_pud-ud-test-converted.conllu" 
ATTENTIONS_TARGET="resources/en_pud-ud-test-converted_attentions.npz"
ATTENTIONS_1="en_pud-ud-test-converted_4_attentions.npz"
ATTENTIONS_1_COMBINED="resources/en_pud-ud-test-converted_4_attentions.combined.npz"



#python3 attention-analysis-clark-etal/convert_sub_to_json.py

#python3 head-ensembles/json2conllu.py "resources/en_pud-ud-test-converted_4_listed.json" "resources/en_pud-ud-test-converted_4_listed.conllu"
python3 head-ensembles/normalize_substitutions.py $ATTENTIONS_1 "resources/en_pud-ud-test-converted_4_source.txt" "resources/en_pud-ud-test-converted_4_listed.conllu" "resources/en_pud-ud-test-converted_4.json"

python3 head-ensembles/head_ensemble_substitutions.py "resources/en_pud-ud-test-converted_6_attentions.combined.npz" "resources/en_pud-ud-test-converted_source.txt" $ORIGINAL_CONLL -j "resources/en_pud-ud-test-original_6_head-ensembles.json" --report-result "results/en_pud-ud-test-original_6.dep_acc"

NUMBER_SENTS="4"
#JSON_USED="5"

for number_sents in $NUMBER_SENTS; do
    echo ${number_sents}
    ATTENTIONS_COMBINED=resources/en_pud-ud-test-converted_${number_sents}_attentions.combined.npz
    
    H_E_JSON=resources/en_pud-ud-test-original_${number_sents}_head-ensembles.json
    python3 head-ensembles/extract_trees_substitutions.py $ATTENTIONS_COMBINED "resources/en_pud-ud-test-converted_source.txt" $ORIGINAL_CONLL $H_E_JSON --report-result results/en_pud-ud-test-original_${number_sents}_h-e_${number_sents}.tree
done


python3 head-ensembles/extract_trees_substitutions.py "resources/en_pud-ud-test-converted_3_attentions.combined.npz" "resources/en_pud-ud-test-converted_source.txt" $ORIGINAL_CONLL "resources/en_pud-ud-test-original_3_head-ensembles.json" --report-result results/en_pud-ud-test-original_3_h-e_3.tree.uuas


