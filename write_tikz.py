"""
Adapted from: https://github.com/mcqll/cpmi-dependencies
Hoover et al., "Linguistic Dependencies and Statistical Dependence", EMNLP 2021.
"""
import glob
import os.path
import pandas as pd
import numpy as np
import pickle
import networkx as nx
from argparse import ArgumentParser
from tqdm import tqdm
from ast import literal_eval

#from conll_data import CONLLReader, EXCLUDED_PUNCTUATION, EXCLUDED_PUNCTUATION_UPOS, CONLL_COLS


def is_edge_to_ignore(edge, observation):
    # is_d_punct = bool(observation.FORM[edge[1]-1] in EXCLUDED_PUNCTUATION)
    is_d_punct = bool(observation['upos'] in ["PUNCT"])
    is_h_root = bool(edge[0] == 0) or bool(observation['deprel'] == 'root')
    return is_d_punct or is_h_root


def make_string_safe(input_string, replace_dict):
    '''Make a string safe by replacing all naughty characters
    according to replace_dict
    '''
    new_string = input_string
    for naughty_char in replace_dict.keys():
        new_string = new_string.replace(naughty_char, replace_dict[naughty_char])
    return new_string


def make_tikz_string(
        predicted_edges, observation,
        label1='', label2='', label3=''):
    ''' Writes out a tikz dependency TeX file
    for comparing predicted_edges and gold_edges'''
    gold_edges_list = []
    gold_edge_to_label = {}
    for w in observation:
        e = (w['id'], w['head'], w['deprel'])
        gold_edges_list.append(e)
        if not is_edge_to_ignore(e, w):
            gold_edge_to_label[(e[0], e[1])] = e[2] 
    gold_edges_set = {tuple(sorted(e)) for e in gold_edge_to_label.keys()}

    # note converting to 1-indexing
    predicted_edges_set = {
            tuple(sorted((x[0]+1, x[1]+1))) for x in predicted_edges}
    correct_edges = list(gold_edges_set.intersection(predicted_edges_set))
    incorrect_edges = list(predicted_edges_set.difference(gold_edges_set))
    num_correct = len(correct_edges)
    num_total = len(gold_edges_set)
    uuas = num_correct/float(num_total) if num_total != 0 else np.NaN

    # replace non-TeXsafe characters... add as needed
    tex_replace = {'$': '\\$', '&': '$\\with$', '%': '\\%',
                   '~': '\\textasciitilde', '#': '\\#', '|': '{|}'}

    # make string
    string = "\\begin{dependency}\n\t\\begin{deptext}\n\t\t"
    string += "\\& ".join([make_string_safe(w['text'], tex_replace) 
        for w in observation]) + " \\\\" + '\n'
    string += "\t\\end{deptext}" + '\n'
    for i_index, j_index in gold_edge_to_label:
        string += f'\t\\depedge{{{i_index}}}{{{j_index}}}{{{gold_edge_to_label[(i_index, j_index)]}}}\n'
    for i_index, j_index in correct_edges:
        string += f'\t\\depedge[hide label, edge below, edge style={{-, blue, opacity=0.5}}]{{{i_index}}}{{{j_index}}}{{}}\n'
    for i_index, j_index in incorrect_edges:
        string += f'\t\\depedge[hide label, edge below, edge style={{-, red, opacity=0.5}}]{{{i_index}}}{{{j_index}}}{{}}\n'
    string += "\t\\node (R) at (\\matrixref.east) {{}};\n"
    string += f"\t\\node (R1) [right of = R] {{\\tiny\\textsf{{{label3}}}}};\n"
    string += f"\t\\node (R2) at (R1.north) {{\\tiny\\textsf{{{label2}}}}};\n"
    string += f"\t\\node (R3) at (R2.north) {{\\tiny\\textsf{{{label1}}}}};\n"
    string += f"\t\\node (R4) at (R1.south) {{\\tiny "
    string += f"$ {num_correct}/{num_total} = {uuas*100:.0f}\\% $}};\n"
    string += f"\\end{{dependency}}\n"
    return string


def write_tikz_files(
        outputdir, edges_df, output_suffix='', info_text='', index_info_text='', gold_standard=None):
    ''' writes TikZ string to outputdir,
    a separate file for each sentence index'''
    for ind, sentence_index in enumerate(tqdm(edges_df.keys())):
        #print(len(sentence_index.split()))
        predicted_edges = list(nx.dfs_tree(edges_df[sentence_index]).edges())
        #print(len(predicted_edges))
        gold_standard_sent = gold_standard[sentence_index]
        pos_list = [w['upos'] for w in gold_standard_sent]
        predicted_edges = realign_parses(predicted_edges, pos_list)
        tikz_string = make_tikz_string(predicted_edges,
                                       gold_standard_sent,
                                       label1=output_suffix + ' ' + str(ind),
                                       label2=output_suffix,
                                       label3=info_text)
        tikzf = str(ind) + '_' + output_suffix + ".tikz"
        tikzdir = os.path.join(outputdir, tikzf)
        #print(f'writing tikz to {tikzdir}')
        with open(tikzdir, 'w') as fout:
            fout.write(f"% dependencies for {OUTPUTDIR}\n")
            fout.write(tikz_string)

def head_list(edge_list):
    heads = [-1 for i in range(len(edge_list) + 1)]
    for e in edge_list:
        proposed_head = e[0]
        proposed_dep = e[1]
        assert heads[proposed_dep] == -1
        heads[proposed_dep] = proposed_head
    return heads

def realign_parses(pred_edges, pos):
    new_realigned = []
    map_list = {}
    count = 0
    for j, w in enumerate(pos):
        if w != 'PUNCT':
            map_list[count] = j
            count += 1
    assert count == len(pred_edges) + 1
    for x in pred_edges:
        a, b = x
        new_realigned.append((map_list[a], map_list[b]))
    return new_realigned


if __name__ == '__main__':
    #perturbed_file = '/home/mila/j/jasper.jian/semantic-perturbation/SemanticPerturbation/out/en_pud-ud-test/bert.pos_row_10.perturbed.pkl'
    #target_only_file = '/home/mila/j/jasper.jian/semantic-perturbation/SemanticPerturbation/out/en_pud-ud-test/bert.pos_row_10.target.pkl'
    gold_file = '/home/mila/j/jasper.jian/semantic-perturbation/SemanticPerturbation/out/en_pud-ud-test/pud_parses.pkl'
    perturbed_file = '/home/mila/j/jasper.jian/semantic-perturbation/SemanticPerturbation/out/ptb3-wsj-test_10/bert.test10.perturbed.pkl'
    target_only_file = '/home/mila/j/jasper.jian/semantic-perturbation/SemanticPerturbation/out/ptb3-wsj-test_10/bert.test10.target.pkl'
    #gold_file = '/home/mila/j/jasper.jian/semantic-perturbation/SemanticPerturbation/out/ptb3-wsj-test_10/wsj_parses2.pkl'
    with open(perturbed_file, 'rb') as f:
        perturbed_parses = pickle.load(f)
    with open(target_only_file, 'rb') as f:
        target_only_parses = pickle.load(f)
    with open(gold_file, 'rb') as f:
        gold_parses = pickle.load(f)
    
    OUTPUTDIR = '/home/mila/j/jasper.jian/semantic-perturbation/SemanticPerturbation/out/en_pud-ud-test/' + 'tikztest'
    os.makedirs(OUTPUTDIR, exist_ok=True)

    edge_type = "perturbed"
    write_tikz_files(OUTPUTDIR, perturbed_parses, output_suffix='ssud', gold_standard=gold_parses)
    edge_type = "target"
    write_tikz_files(OUTPUTDIR, target_only_parses, output_suffix='target_only', gold_standard=gold_parses)

    
    TEX_FILEPATH = os.path.join(OUTPUTDIR, 'dependencies.tex')
    with open(TEX_FILEPATH, mode='w') as tex_file:
        print(f'writing TeX to {TEX_FILEPATH}')
        tex_file.write(
            "\\documentclass[tikz]{standalone}\n"
            "\\usepackage{tikz,tikz-dependency}\n"
            "\\usepackage{cmll,xeCJK}\n" # for typesetting '&' and CJK resp
            "\\setmainfont{Arial Unicode MSn}\n"
            "\\setsansfont{Arial Narrow}\n"
            "\\setCJKmainfont{Arial Unicode MS}\n"
            "\\pgfkeys{%\n/depgraph/edge unit distance=.75ex,%\n"
            "%/depgraph/edge horizontal padding=2,%\n"
            "/depgraph/reserved/edge style/.style = {\n->,% arrow properties\n"
            "semithick, solid, line cap=round, % line properties\n"
            "rounded corners=2,% make corners round\n},%\n"
            "/depgraph/reserved/label style/.style = {font=\sffamily,\n"
            "% anchor = mid, draw, solid, black, rotate = 0,"
            "rounded corners = 2pt,%\nscale = .5,%\ntext height = 1.5ex,"
            "text depth = 0.25ex,% needed to center text vertically\n"
            "inner sep=.2ex,%\nouter sep = 0pt,%\ntext = black,%\n"
            "fill = white,% opacity = 0, text opacity = 0 "
            "% uncomment to hide all labels\n},%\n}\n"
            "\\begin{document}\n\n% % Put tikz dependencies here, like\n"
        )
        tex_file.write(f"% dependencies for {OUTPUTDIR}\n")
        TIKZFILES = glob.glob(os.path.join(OUTPUTDIR, '*.tikz'))
        TIKZFILES = [os.path.basename(x) for x in TIKZFILES]
        for tikzfile in sorted(TIKZFILES):
            tex_file.write(f"\\input{{{tikzfile}}}\n")
        tex_file.write("\n\\end{document}")
