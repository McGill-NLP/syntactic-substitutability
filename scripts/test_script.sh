#!/bin/bash

#SBATCH --partition=unkillable
#SBATCH --cpus-per-task=3
#SBATCH --gres=gpu:rtx8000:1                     # Ask for 1 GPU
#SBATCH --mem=24G                        # Ask for 32 GB of CPU RAM
#SBATCH --time=1:00:00                   # The job will run for 12 hours

module load anaconda/3
conda activate default-env

python generate_substitutions.py "/home/mila/j/jasper.jian/projects/jaspers-test/data/ptb3-wsj-test_10.conllx" "/home/mila/j/jasper.jian/projects/jaspers-test/test_out" "bert" "1"
