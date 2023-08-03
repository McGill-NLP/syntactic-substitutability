import numpy as np
import argparse
from collections import defaultdict, namedtuple
import matplotlib.pyplot as plt
from tqdm import tqdm
import json

from tools import dependency, sentence_attentions
from tools.attention_wrapper import AttentionWrapper
from tools.metrics import DepAcc
TOP_HEADS_NUM = 25


RelData = namedtuple('RelData','layers heads d2p')
RelData2 = namedtuple('RelData', 'layers heads layersT headsT weight weightT')


def average_heads(attn_wrapper, item, ls, hs):
    return np.average(np.array([attn_wrapper.get_head(item, l, h) for l, h in zip(ls, hs)]), axis = 0)


def plot_uas(uas, title, xlabel, ylabel, color='lightblue'):

    fig, ax = plt.subplots(1, 1, figsize=(16, 6))
    ax.plot(range(len(uas)),uas, color=color)

    fig.suptitle(title)

    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument("-a", "--attentions", required=True, help="NPZ file with attentions")
    ap.add_argument("-t", "--tokens", required=True, help="Labels (tokens) separated by spaces")
    
    ap.add_argument("-ta", "--test-attentions", required=True, help="NPZ file with attentions for test")
    ap.add_argument("-tt", "--test-tokens", required=True, help="Labels (tokens) separated by spaces for test")

    ap.add_argument("-c", "--conllu", help="Eval against the given conllu file")
    ap.add_argument("-tc", "--test-conllu", help="Eval against the given conllu file")

    ap.add_argument("-s", "--sentences", type=int, default=None,
                    help="Only use up to s sentences; 0-based")
    
    ap.add_argument("-r", "--randomize", action="store_true",
                    help="whether to shuffle sentences")
    ap.add_argument("-m", "--maxlen", type=int, default=1000,
                    help="Skip sentences longer than this many words. A word split into several wordpieces is counted as one word. EOS is not counted.")

    ap.add_argument("--numheads", type=int, default=3,
                    help="Maximal number of heads averaged.")

    ap.add_argument("-e", "--eos", action="store_true",
                    help="Attentions contain EOS")

    ap.add_argument("-n", "--no-softmax", action="store_true",
                    help="Whether not to use softmax for attention matrices, use with bert metrices")
    
    ap.add_argument("-tn", "--test-no-softmax", action="store_true",
                    help="Whether not to use softmax for attention matrices, use with bert metrices")
    
    ap.add_argument("-j", "--json", type=str, help='Output json file with heads selected.')

    args = ap.parse_args()

    with open(args.tokens) as tokens_file:
        tokens_loaded = [l.split() for l in tokens_file]
    grouped_tokens, _ = dependency.group_wordpieces(tokens_loaded, args.conllu)
    sel_attn = AttentionWrapper(args.attentions, args.tokens, grouped_tokens)
    
    with open(args.test_tokens) as test_token_file:
        test_tokens_loaded = [l.split() for l in test_token_file]
    test_grouped_tokens, _ = dependency.group_wordpieces(test_tokens_loaded, args.test_conllu)
    tst_attn = AttentionWrapper(args.test_attentions, args.test_tokens, test_grouped_tokens)
    
    dependency_rels = dependency.read_conllu(args.conllu, directional=True)
    test_dependency_rels = dependency.read_conllu(args.test_conllu, directional=True)


    dep_acc = DepAcc(dependency_rels)
    tst_dep_acc = DepAcc(test_dependency_rels)

    if args.sentences:
        # if args.randomize:
        #     sentences = list(np.random.choice(sel_attn.sentence_count, args.sentences, replace=False))
        # else:
        sentences = list(np.arange(args.sentences))
    else:
        sentences = None
        
    metric_result = sel_attn.calc_metric(dep_acc,params=dependency_rels[0].keys(),selected_sentences=sentences)

    all_metric = defaultdict(list)
    best_head_mixture = dict()
    max_metric = dict()


    def update_if_canidate(curr_heads_ids, best_heads, max_metric, metric_res, metric, k):
        curr_lids, curr_hids = np.unravel_index(curr_heads_ids, metric_res[k].shape)
        avg_gen = (average_heads(sel_attn, index, curr_lids, curr_hids) for index in range(sel_attn.sentence_count)
                   if sel_attn.check_subtokens(index))
        curr_metric = metric.calculate(avg_gen, k)
        if curr_metric > max_metric[k]:
            max_metric[k] = curr_metric
            best_heads[k] = np.unravel_index(curr_heads_ids, metric_res[k].shape)
            return list(curr_heads_ids)
        else:
            return None


    for k in tqdm(sorted(metric_result.keys())):

        top_heads_ids = np.argsort(metric_result[k], axis=None)[-TOP_HEADS_NUM:][::-1]
        picked_heads_ids = []
        max_metric[k] = - np.inf
        for num in range(0,TOP_HEADS_NUM):
            new_head_id = top_heads_ids[num]

            candidate_heads_ids = None

            if len(picked_heads_ids) < args.numheads:
                curr_heads_ids = list(picked_heads_ids)
                curr_heads_ids.append(new_head_id)
                candidate_heads_ids = update_if_canidate(curr_heads_ids, best_head_mixture, max_metric, metric_result, dep_acc,k)

            else:
                for sub_idx in range(len(picked_heads_ids)):
                    curr_heads_ids = list(picked_heads_ids)
                    curr_heads_ids[sub_idx] = new_head_id
                    candidate_heads_ids = update_if_canidate(curr_heads_ids, best_head_mixture, max_metric, metric_result, dep_acc,k)

            if candidate_heads_ids is not None:
                picked_heads_ids = candidate_heads_ids

            all_metric[k].append(max_metric[k])
            
        rec_best_gen = (average_heads(tst_attn, index, best_head_mixture[k][0], best_head_mixture[k][1]) for
                        index in range(tst_attn.sentence_count) if tst_attn.check_subtokens(index))
        
        tst_res = tst_dep_acc.calculate(rec_best_gen, k)
        
        print(f"Best dependency accuracy for {k} : {tst_res}")

    relation_rules = dict()
    relation_rules2 = dict()
    for k in tqdm(sorted(max_metric.keys())):
        if k.endswith('p2d'):
            alt_k = k[:-4] + '-d2p'
            
            relation_rules2[k] = RelData2(best_head_mixture[k][0].tolist(), best_head_mixture[k][1].tolist(),
                                          best_head_mixture[alt_k][0].tolist(), best_head_mixture[alt_k][1].tolist(),
                                          max_metric[k], max_metric[alt_k])._asdict()
            if max_metric[k] > max_metric[alt_k]:
                relation_rules[k] = RelData(best_head_mixture[k][0].tolist(), best_head_mixture[k][1].tolist(), False)._asdict()
            else:
                relation_rules[k] = RelData(best_head_mixture[alt_k][0].tolist(), best_head_mixture[alt_k][1].tolist(), True)._asdict()

    with open(args.json + '_heads.json', 'w') as outj_1:
        json.dump(relation_rules, fp=outj_1)
    with open(args.json + '_heads2.json', 'w') as outj_2:
        json.dump(relation_rules2, fp=outj_2)
