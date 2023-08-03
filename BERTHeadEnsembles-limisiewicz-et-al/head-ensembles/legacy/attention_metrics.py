# tom lim
# my approach:
#   - eos token is omitted (attention avaraged for the rest of tokens)
#   - subtoken attentions are averaged
#   - long sentences skipped (TODO: include them)

import argparse
import numpy as np
import matplotlib
from matplotlib import pyplot as plt

from tqdm import tqdm

import pandas as pd

from tools import dependency, sentence_attentions


def heatmap(AUC, title, xlabel, ylabel, xticklabels, yticklabels, cmap='bone', color='lightblue', vmax=0.5):
    '''
    Copied form:
    https://stackoverflow.com/questions/20574257/constructing-a-co-occurrence-matrix-in-python-pandas
    '''
    # Plot it out
    fig, (ax, ax2) = plt.subplots(1,2,figsize=(16,6), gridspec_kw={'width_ratios': [3, 1]})

    ax.set_xticklabels(xticklabels, minor=False)
    ax.set_yticklabels(yticklabels, minor=False)

    # put the major ticks at the middle of each cell
    ax.set_xticks(np.arange(AUC.shape[1]))
    ax.set_yticks(np.arange(AUC.shape[0]))

    im = ax.imshow(AUC,cmap=cmap, vmin=0, vmax=vmax)
    valfmt = matplotlib.ticker.StrMethodFormatter("{x:.2f}")
    for i in range(AUC.shape[0]):
        for j in range(AUC.shape[1]):
            text = ax.text(j, i, valfmt(AUC[i, j]), ha="center", va="center", color="black")
    fig.colorbar(im, ax=ax, orientation='horizontal')
    ax2.barh(np.arange(AUC.shape[0]),np.mean(AUC,axis=1), color=color)
    #ax2.set_xticks(np.arange(AUC.shape[1]))
    # set title and x/y labels
    fig.suptitle(title)
    
    ax.set_xlabel(xlabel)
    ax2.set_ylabel(ylabel)
    ax.set_ylabel(ylabel)
    ax.invert_yaxis()
    
    ax.set_title('per head')
    ax2.set_title('per layer') #just average


def plot_matrix(matrix):

    fig, ax1 = plt.subplots(figsize=(9,9), ncols=1)
    im = ax1.imshow(matrix,cmap='Blues')
    fig.colorbar(im)
    plt.show()


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument("attentions", required=True, help="NPZ file with attentions")
    ap.add_argument("tokens", required=True, help="Labels (tokens) separated by spaces")

    ap.add_argument("-u", "--uas", help="Output uas measuere into this file")
    ap.add_argument("-c", "--conllu", help="Eval against the given conllu file")
    ap.add_argument("-T", "--train-conllu", help="Conllu file for training POS",
                    default='/net/projects/LSD/attention_tomasz/lsd/attention-analysis/naacl2019/graph-extraction/entrain.conllu')

    ap.add_argument("-f", "--format", default="png",
                    help="Output visualisation as this format (pdf, png, maybe other options)")
    ap.add_argument("-F", "--fontsize", default=8, type=int,
                    help="Fontsize for heatmap; 8 is good for png. 10 is good for PDF it seems")

    ap.add_argument("-s", "--sentences", nargs='*', type=int, default=None,
                    help="Only use the specified sentences; 0-based")
    ap.add_argument("-m", "--maxlen", type=int, default=1000,
                    help="Skip sentences longer than this many words. A word split into several wordpieces is counted as one word. EOS is not counted.")

    ap.add_argument("-e", "--eos", action="store_true",
                    help="Attentions contain EOS")
    
    ap.add_argument("-n", "--no-softmax", action="store_true",
                    help="Whether not to use softmax for attention matrices, use with bert metrices")

    args = ap.parse_args()

    attentions_loaded = np.load(args.attentions)
    sentences_count = len(attentions_loaded.files)
    layers_count = attentions_loaded['arr_0'].shape[0]
    heads_count = attentions_loaded['arr_0'].shape[1]

    with open(args.tokens) as tokens_file:
        tokens_loaded = [l.split() for l in tokens_file]

    # in dependency_rels for each sentece there is a lists of tuples (token, token's head)
    # in dependency_rels_rev tuples are reversed.
    dependency_rels = dependency.read_conllu(args.conllu, directional=True)

    uas = {aggr: np.zeros((sentences_count, layers_count, heads_count))
           for aggr in dependency.labels}
    
    rel_number = {aggr: np.zeros((sentences_count, 1, 1)) for aggr in dependency.labels}

    grouped_tokens, _ = dependency.group_wordpieces(tokens_loaded, args.conllu)

    attention_gen = sentence_attentions.generate_matrices(attentions_loaded, grouped_tokens, args.eos, args.no_softmax,
                                                          args.maxlen, args.sentences)
    
    dependency_rels_labeled = dependency.read_conllu_labeled(args.conllu, convert=True)
    
    pos = {posl: np.zeros((sentences_count, layers_count, heads_count))
           for posl in dependency.pos_labels}
    pos['root'] = np.zeros((sentences_count, layers_count, heads_count))
    pos_frame = dependency.conllu2freq_frame(args.train_conllu)
    
    sentences_considered = []
    for vis, idx in tqdm(attention_gen):
        sentences_considered.append(idx)
        pos_masks = dict()
        diag_mask = sentence_attentions.diagonal_mask(dependency_rels_labeled[idx])
        sent_relations = dependency_rels_labeled[idx]
        for k in uas.keys():
            rel_number[k][idx, 0, 0] = len(dependency_rels[idx][k])

            pos_masks[k] = diag_mask
        for layer in range(layers_count):
            for head in range(heads_count):

                for k in uas.keys():
                    # NOTE 9: pos soft mask is used (based on frequency of nodes)
                    deps = vis[layer][head] * pos_masks[k]
                    deps = (deps == deps.max(axis=1)[:, None]).astype(int)
                    if len(dependency_rels[idx][k]):
                        uas[k][idx, layer, head] \
                            = np.sum(deps[tuple(zip(*dependency_rels[idx][k]))])
                        
                    for token_id, _, rell, posl in sent_relations:
                        pos[posl][idx, layer, head] += deps[token_id]
                        if rell == 'root':
                            pos['root'][idx, layer, head] += deps[token_id]

    dependency_rels = [dependency_rels[idx] for idx in sentences_considered]
    
    
    # for k in uas.keys():
    #     uas[k] = uas[k][sentences_considered, :, :]
    #     rel_number[k] = rel_number[k][sentences_considered, :, :]
    #
    #     uas[k] = np.sum(uas[k], axis=0) / np.sum(rel_number[k], axis=0)
    #     uas_filename = f'{args.uas}-{k}.{args.format}'
    #
    #     heatmap(uas[k], f"Accuracy {k}", "heads", "layers", np.arange(heads_count), np.arange(layers_count),
    #             cmap='pink', color='sandybrown', vmax=1.0)
    #
    #     plt.savefig(uas_filename, dpi=200, format=args.format)
    #     plt.close()

