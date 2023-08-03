#!/bin/bash
#SBATCH --partition=short-unkillable
#SBATCH --cpus-per-task=24
#SBATCH --gres=gpu:rtx8000:1                     # Ask for 1 GPU
#SBATCH --mem=128G                        # Ask for 32 GB of CPU RAM
#SBATCH --time=2:00:00                   # The job will run for 12 hours

python3 attention-analysis-clark-etal/extract_attention_substitutions.py --preprocessed-data-file "resources/en_pud-ud-test-converted.json" --bert-dir bert-base/uncased_L-12_H-768_A-12 --max-sequence-length 512