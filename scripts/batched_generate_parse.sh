#!/bin/bash
#SBATCH --partition=unkillable
#SBATCH --cpus-per-task=3
#SBATCH --gres=gpu:rtx8000:1                     # Ask for 1 GPU
#SBATCH --mem=32G                        # Ask for 32 GB of CPU RAM
#SBATCH --time=12:00:00                   # The job will run for 12 hours

module load anaconda/3
conda activate default-env

# Data split to evaluate on
#SPLIT="en_pud-ud-test" 
SPLIT="ptb3-wsj-test_10"

# The matching CONLL file with gold standard parses
#CONLLU_FILE='data/en_pud-ud-test.conllu'
CONLLU_FILE="data/ptb3-wsj-test_10.conllx"

# The model 
MODEL="bert"

# The output directory for generated sentences and results
OUTDIR="/out/${SPLIT}/results"
mkdir -p $OUTDIR

# This part generates the sentence with increasing numbers of perturbations per position
NUMBER_SENTS="1 3 5 10 15"
for number_sents in $NUMBER_SENTS; do
    echo ${number_sents}
    python generate_perturbations.py $CONLLU_FILE $OUTDIR $MODEL $number_sents
done

# The different settings for combining the perturbations from each word
SETTINGS="row"
for number_sents in $NUMBER_SENTS; do
    for setting in $SETTINGS; do
        SENTENCE_FILE=${OUTDIR}/bert.pos_${number_sents}.pkl
        python perturbed_parsing.py $SENTENCE_FILE $OUTDIR $SPLIT $number_sents $setting
    done 
done
