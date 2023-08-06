# Syntactic Substitutability as Unsupervised Dependency Syntax

This is the code for the paper ['Syntactic Substitutability as Unsupervised Dependency Syntax'](https://arxiv.org/abs/2211.16031). Instructions to replicate experiments are below.

# Data format

CONLL-formatted dependency treebanks are required. The PUD treebank can be downloaded from the [Universal Dependencies project website](https://universaldependencies.org/).

Results are outputted in CSV format. Induced trees can be saved and viewed in a Latex tikz format.

Substitutions must be generated before parsing sentences. See below for how to generate a parse for a single sentence.

# Generating substitutions

The script `generate_sentences.sh` can be used to generate substituted sentences. The following variables can be set:
* `SPLIT`: name of the dataset
* `CONLLU_FILE`: path to the CONLL-formatted treebank to parse
* `NUMBER_SENTS`: the number of substitutions to generate at each position in the sentence

# Inducing trees

The script `parse_sentences.sh` can be used to parse and evaluate on each dataset. The following variables can be set:
* `SPLIT`: name of the dataset
* `CONLLU_FILE`: path to the CONLL-formatted treebank to parse
* `NUMBER_SENTS`: the number of substitutions to generate at each position in the sentence

It will save to the output directory a CSV formatted file containing the UUAS scores of the induced trees.

# Induce your own parses!

Given any CONLL-file, the parses can be obtained by running it through the pipeline described above. Otherwise, the structure induced for single sentences can also be obtained by running `parse_single_sentence.py` directly in the command line as below:

```python parse_single_sentence.py [NUMBER OF SUBSTITUTIONS]```

Following the instructions in the command line, this will output a list of edges and a simple text description of the induced tree.