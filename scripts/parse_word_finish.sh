#!/bin/bash
#SBATCH --partition=unkillable
#SBATCH --cpus-per-task=3
#SBATCH --gres=gpu:rtx8000:1                     # Ask for 1 GPU
#SBATCH --mem=24G                        # Ask for 32 GB of CPU RAM
#SBATCH --time=12:00:00                   # The job will run for 12 hours

module load anaconda/3
conda activate default-env

#SPLIT="en_pud-ud-test" # Data split to evaluate on
SPLIT="ptb3-wsj-test_10"
#SPLIT="long_dist_subj"
#CONLLU_FILE='./out/long_dist_obj/obj_rel_sampled.pkl'
MODEL="bert"
OUTDIR=./out/${SPLIT}/final3
#SENTENCE_FILE=./out/${SPLIT}/${MODEL}.generated_sentences_3.pkl # Path to the evalaute data file
NUMBER_SENTS="1 3 5 10 15"
SETTINGS="row all column"


for setting in $SETTINGS; do
    for number_sents in $NUMBER_SENTS; do
        SENTENCE_FILE=${OUTDIR}/bert.generated_sentences_pos_${number_sents}.pkl
        python parse_word_level.py $SENTENCE_FILE $OUTDIR $SPLIT $number_sents $setting
    done 
done
