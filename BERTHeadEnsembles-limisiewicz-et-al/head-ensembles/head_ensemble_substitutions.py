#!/usr/bin/env python3

import argparse
import numpy as np
import json
from tqdm import tqdm
from collections import defaultdict

from dependency import Dependency
from attention_wrapper_substitutions import AttentionWrapper
from metrics import DepAcc


HEADS_TO_CHECK = 25


class HeadEnsemble():

    def __init__(self, relation_label, num_heads):
        self.ensemble = list()
        self.max_metric = 0.
        self.metric_history = list()
        self.max_ensemble_size = num_heads
        self.relation_label = relation_label

    @classmethod
    def from_dict(cls, **entries):
        he_object = cls(entries.get("relation_label"), entries.get("max_ensemble_size"))
        he_object.__dict__.update(entries)
        return he_object

    def consider_candidate(self, candidate, metric, attn_wrapper):
        """ Checks whether given candidate head should be included in the tuple
        Parameters:
            candidate (tuple): (layer_index, head_index) of the candidate head
            metric (Metric): metric to optimize
            attn_wrapper (AttentionWrapper):
        """
        candidate_lid, candidate_hid = candidate
        if not self.ensemble:
            self.max_metric = float(attn_wrapper.calc_metric_ensemble(metric, [candidate_lid], [candidate_hid]))
            self.ensemble.append(tuple(map(int,candidate)))
        elif len(self.ensemble) < self.max_ensemble_size :
            ensemble_lids, ensemble_hids = map(list, zip(*self.ensemble))
            candidate_metric = float(attn_wrapper.calc_metric_ensemble(metric, ensemble_lids + [candidate_lid],
                                                                 ensemble_hids + [candidate_hid]))
            if candidate_metric > self.max_metric:
                self.max_metric = candidate_metric
                self.ensemble.append(tuple(map(int,candidate)))
        else:
            max_candidate_metric = 0.
            opt_substitute_idx = None
            for substitute_idx in range(self.max_ensemble_size ):
                ensemble_lids, ensemble_hids = map(list, zip(*self.ensemble))
                ensemble_lids[substitute_idx] = candidate_lid
                ensemble_hids[substitute_idx] = candidate_hid
                candidate_metric = float(attn_wrapper.calc_metric_ensemble(metric, ensemble_lids, ensemble_hids))
                if candidate_metric > self.max_metric and candidate_metric > max_candidate_metric:
                    max_candidate_metric = candidate_metric
                    opt_substitute_idx = substitute_idx

            if opt_substitute_idx is not None:
                self.ensemble[opt_substitute_idx] = tuple(map(int,candidate))
                self.max_metric = max_candidate_metric
        self.metric_history.append(self.max_metric)

    def calc_metric(self, metric, attenion_wrapper):
        ensemble_lids, ensemble_hids = map(list, zip(*self.ensemble))
        return float(attenion_wrapper.calc_metric_ensemble(metric, ensemble_lids, ensemble_hids))


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument("attentions", type=str, help="NPZ file with attentions")
    ap.add_argument("tokens", type=str, help="Labels (tokens) separated by spaces")
    ap.add_argument("conll", type=str, help="Conll file for head selection.")

    ap.add_argument("-m", "--metric", type=str, default="DepAcc", help="Metric used to find optimal head ensembles.")
    ap.add_argument('-n', '--num-heads', type=int, default=4, help="Maximal number of heads in one ensemble")
    ap.add_argument("-j", "--json", type=str, default=None, help="Json with the head ensembles")
    ap.add_argument("-e", "--evaluate-only", action="store_true", help="Whether to only evaluate (preomputed Json with head ensembles needed)")
    # other arguments
    ap.add_argument("--report-result", type=str, default=None, help="File where to save the results.")
    ap.add_argument("-s", "--sentences", nargs='*', type=int, default=None,
                    help="Only use the specified sentences; 0-based")

    args = ap.parse_args()

    dependency_tree = Dependency(args.conll, args.tokens)
    bert_attns = AttentionWrapper(args.attentions, dependency_tree.wordpieces2tokens, args.sentences)

    if args.evaluate_only:
        if not args.json:
            raise ValueError("JSON with head ensembles required in evaluate only mode!")
        with open(args.json, 'r') as inj:
            head_ensembles = json.load(inj)
        head_ensembles = {rl: HeadEnsemble.from_dict(**he_dict) for rl, he_dict in head_ensembles.items()}
        
    else:
        head_ensembles = dict()

    results = defaultdict(dict)
    non_clausal_relations = ('adj-modifier', 'adv-modifier', 'auxiliary', 'compound', 'conjunct', 'determiner',
                         'noun-modifier', 'num-modifier', 'object', 'subject', 'case', 'mark')
    non_clausal_sum = 0.

    clausal_relations = ('adj-clause', 'adv-clause', 'clausal', 'clausal-subject', 'parataxis')
    clausal_sum = 0.
    
    metric = None
    for direction in ['p2d', 'd2p']:
        for relation_label in list(set(dependency_tree.label_map.values())) + [Dependency.LABEL_OTHER, Dependency.LABEL_ALL]:
            if args.metric.lower() == "depacc":
                metric = DepAcc(dependency_tree, relation_label, dependent2parent=(direction=='d2p'))
            else:
                raise ValueError("Unknown metric! Available metrics: DepAcc")
            relation_label_directional = relation_label + '-' + direction
            if not args.evaluate_only:
                head_ensembles[relation_label_directional] = HeadEnsemble(relation_label_directional, args.num_heads)
                print(f"Calculating metric for each head. Relation label: {relation_label_directional}")
                metric_grid = bert_attns.calc_metric_grid(metric)
                heads_idcs = np.argsort(metric_grid, axis=None)[-HEADS_TO_CHECK:][::-1]
                for candidate_id in tqdm(heads_idcs, desc=f"Candidates for ensemble!"):
                    candidate = np.unravel_index(candidate_id, metric_grid.shape)
                    head_ensembles[relation_label_directional].consider_candidate(candidate, metric, bert_attns)
                
                res_metric = head_ensembles[relation_label_directional].max_metric
            else:
                res_metric = head_ensembles[relation_label_directional].calc_metric(metric, bert_attns)
                
            results[relation_label][direction] = res_metric
            if relation_label in clausal_relations:
                clausal_sum += res_metric
            elif relation_label in non_clausal_relations:
                non_clausal_sum += res_metric

    if not args.evaluate_only and args.json:
        with open(args.json, 'w') as outj:
            json.dump({rl: he.__dict__ for rl, he in head_ensembles.items()}, fp=outj)
            
    if args.report_result:
        with open(args.report_result, 'w') as outr:
            outr.write('label\td2p\tp2d\n')
            for rel in clausal_relations:
                outr.write(f"{rel}\t{results[rel]['d2p']}\t{results[rel]['p2d']}\n")
            outr.write('\n')
            for rel in non_clausal_relations:
                outr.write(f"{rel}\t{results[rel]['d2p']}\t{results[rel]['p2d']}\n")
            outr.write('\n')
            outr.write(f'Clausal mean:\t{clausal_sum/len(clausal_relations)/2.}\n')
            outr.write(f'Non Cluasal mean:\t{non_clausal_sum / len(non_clausal_relations) / 2.}\n')
