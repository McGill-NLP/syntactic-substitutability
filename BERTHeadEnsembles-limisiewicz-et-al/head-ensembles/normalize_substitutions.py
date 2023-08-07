#!/usr/bin/env python3

import argparse
import numpy as np
import json
from tqdm import tqdm
from collections import defaultdict

from dependency import Dependency
from attention_wrapper import AttentionWrapper
from metrics import DepAcc

def combine_attentions(attention_list, sub_dict):
    combined_list = []
    ind = 0

    for d in sub_dict:
        original_matrix = attention_list[ind]
        new_matrix = np.array(original_matrix)
        ind += 1
        for pos, sent_list in d['substitutions']:
            if len(sent_list) == 0:
                continue
            else:
                store_pos_matrices = np.stack([original_matrix] + attention_list[ind:ind+len(sent_list)], axis=0)
                averaged = np.mean(store_pos_matrices, axis=0)
                new_matrix[:, :, pos, :] = averaged[:, :, pos, :]
                ind += len(sent_list)
        combined_list.append(new_matrix)
    
    print(len(combined_list))
    return combined_list

if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument("attentions", type=str, help="NPZ file with attentions")
    ap.add_argument("tokens", type=str, help="Labels (tokens) separated by spaces")
    ap.add_argument("conll", type=str, help="Conll file for head selection.")
    ap.add_argument("sub_dict", type=str, help="JSON file with indexed substitutions")

    ap.add_argument("-s", "--sentences", nargs='*', type=int, default=None,
                    help="Only use the specified sentences; 0-based")

    args = ap.parse_args()
    dependency_tree = Dependency(args.conll, args.tokens)
    bert_attns = AttentionWrapper(args.attentions, dependency_tree.wordpieces2tokens, args.sentences)

    # this gets the list of attentions including all the substitutions
    word_processed_attentions = bert_attns.matrices
    with open(args.sub_dict, 'r') as f: 
        indexed_substitutions = json.load(f)
    
    new_attention_list = combine_attentions(word_processed_attentions, indexed_substitutions)
    np.savez(args.attentions.replace("npz", "") + "combined", *new_attention_list)