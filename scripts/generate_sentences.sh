#!/bin/bash

#SBATCH --partition=unkillable
#SBATCH --cpus-per-task=3
#SBATCH --gres=gpu:rtx8000:1                     # Ask for 1 GPU
#SBATCH --mem=24G                        # Ask for 32 GB of CPU RAM
#SBATCH --time=2:00:00                   # The job will run for 12 hours

module load anaconda/3
conda activate default-env

SPLIT="ptb3-wsj-test_10"
# Path to CONLL file
CONLLU_FILE=''
MODEL=bert
OUTDIR=./out/${SPLIT}/final

NUMBER_SENTS="1"

for number_sents in $NUMBER_SENTS; do
    echo ${number_sents}
    python generate_substitutions.py $CONLLU_FILE $OUTDIR $MODEL $number_sents
done
