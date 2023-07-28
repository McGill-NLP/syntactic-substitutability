#!/bin/bash
#SBATCH --partition=main
#SBATCH --cpus-per-task=3
#SBATCH --gres=gpu:rtx8000:1                     # Ask for 1 GPU
#SBATCH --mem=48G                        # Ask for 32 GB of CPU RAM
#SBATCH --time=12:00:00                   # The job will run for 12 hours

module load anaconda/3
conda activate default-env

#SPLIT="en_pud-ud-test" # Data split to evaluate on
#CONLLU_FILE='pud/en_pud-ud-test.conllu'
#SPLIT="ptb3-wsj-test_10"
#CONLLU_FILE='wsj/ptb3-wsj-test_10.conllx' # Path to the evaluate data file
SPLIT="long_dist_subj"
CONLLU_FILE='out/long_dist_subj/subj_rel_sampled.conllx'
MODEL=bert
OUTDIR=./out/${SPLIT}/final

NUMBER_SENTS="1 3 5 10"

for number_sents in $NUMBER_SENTS; do
    echo ${number_sents}
    python w2v_generate.py $CONLLU_FILE $OUTDIR $MODEL $number_sents
done

SPLIT="ptb3-wsj-test_10"

for number_sents in $NUMBER_SENTS; do
    SENTENCE_FILE=${OUTDIR}/bert.generated_sentences_pos_${number_sents}.pkl
    python parse_word_level.py $SENTENCE_FILE $OUTDIR $SPLIT $number_sents $setting
done
