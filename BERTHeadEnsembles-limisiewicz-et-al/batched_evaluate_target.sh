#!/bin/bash
#SBATCH --partition=main
#SBATCH --cpus-per-task=3
#SBATCH --gres=gpu:rtx8000:1                     # Ask for 1 GPU
#SBATCH --mem=32G                        # Ask for 32 GB of CPU RAM
#SBATCH --time=1:00:00                   # The job will run for 12 hours

ORIGINAL_CONLL="resources/en_pud-ud-test.conllu"
CONVERTED_CONLL="resources/en_pud-ud-test-converted.conllu" 
ATTENTIONS_TARGET="resources/en_pud-ud-test-converted_attentions.npz"


python3 head-ensembles/head_ensemble.py "resources/en_pud-ud-test-converted_attentions.npz" "resources/en_pud-ud-test-converted_source.txt" $ORIGINAL_CONLL -j "resources/en_pud-ud-test-original_0_head-ensembles.json" --report-result "results/en_pud-ud-test-original_0.dep_acc"
python3 head-ensembles/extract_trees.py "resources/en_pud-ud-test-converted_attentions.npz" "resources/en_pud-ud-test-converted_source.txt" $ORIGINAL_CONLL "resources/en_pud-ud-test-original_0_head-ensembles.json" --report-result "results/en_pud-ud-test-original_0_h-e_0.tree.uuas"
