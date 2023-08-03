#!/usr/bin/env python3
import argparse
import numpy as np
import matplotlib.pyplot as plt
import os


from dependency import Dependency
from attention_wrapper import AttentionWrapper


def plot_head(att_matrices, sentence_tokens, layer_idx, head_idx, out_file, color_scale="Blues"):
    fig, ax = plt.subplots(1, 1, figsize=(6, 6))
    
    matrix = att_matrices[layer_idx, head_idx, :, :]

    ax.imshow(matrix, cmap=color_scale)
    
    head_title = f"LAYER: {layer_idx} HEAD: {head_idx}"
    ax.set_title(head_title)
    ax.set_yticks(np.arange(len(sentence_tokens)))
    ax.set_xticks(np.arange(len(sentence_tokens)))
    ax.set_xticklabels(sentence_tokens, rotation=90)
    ax.set_yticklabels(sentence_tokens)
    ax.set_ylim(top=-0.5, bottom=len(sentence_tokens) - 0.5)
    
    plt.savefig(out_file, dpi=300, format='pdf')


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument("attentions", type=str, help="NPZ file with attentions")
    ap.add_argument("tokens", type=str, help="Labels (tokens) separated by spaces")
    ap.add_argument("conll", type=str, help="Conll file for head selection.")

    ap.add_argument("-layer_idcs", nargs="*", type=int, default=[5, 3], help = "layer indices to plot")
    ap.add_argument("-head_idcs", nargs="*", type=int, default=[4, 9], help="head indices to plot")
    ap.add_argument("-s", "--sentences", nargs='*', type=int, default=list(range(10)), help="Only use the specified sentences; 0-based")
    
    ap.add_argument("-vis-dir", type=str, default="../results", help="Directory where to save head visualizations")
   
    args = ap.parse_args()
    
    dependency_tree = Dependency(args.conll, args.tokens)
    bert_attns = AttentionWrapper(args.attentions, dependency_tree.wordpieces2tokens, args.sentences)
    
    for sent_idx, attn_mats in bert_attns:
        for l, h in zip(args.layer_idcs, args.head_idcs):
            out_file = os.path.join(args.vis_dir, f"L-{l}_H-{h}_sent-{sent_idx}.pdf")
            plot_head(attn_mats, dependency_tree.tokens[sent_idx], l, h, out_file)
