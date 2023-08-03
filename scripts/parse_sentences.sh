#!/bin/bash
#SBATCH --partition=unkillable
#SBATCH --cpus-per-task=3
#SBATCH --gres=gpu:rtx8000:1
#SBATCH --mem=24G
#SBATCH --time=1:00:00

module load anaconda/3
conda activate default-env

# Data split to evaluate on
SPLIT="ptb3-wsj-test_10"
# Path to CONLL file
CONLLU_FILE=""
OUTDIR=./out/${SPLIT}/final
NUMBER_SENTS="1"


for number_sents in $NUMBER_SENTS; do
    SENTENCE_FILE=${OUTDIR}/substitutions_${number_sents}.pkl
    python parse_eval.py $SENTENCE_FILE $OUTDIR $SPLIT $number_sents $CONLLU_FILE
done

