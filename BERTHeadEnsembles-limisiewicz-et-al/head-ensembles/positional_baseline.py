import argparse
import numpy as np
import json
from tqdm import tqdm
from collections import defaultdict

from dependency import Dependency

if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument("conll", type=str, help="Conll file for head selection.")
    
    ap.add_argument("-j", "--json", type=str, default=None, help="Json with the head ensembles")
    ap.add_argument("-e", "--evaluate-only", action="store_true",
                    help="Whether to only evaluate (preomputed Json with head ensembles needed)")
    # other arguments
    
    ap.add_argument("--report-result", type=str, default=None, help="File where to save the results.")
    ap.add_argument("-s", "--sentences", nargs='*', type=int, default=None,
                    help="Only use the specified sentences; 0-based")
    
    args = ap.parse_args()
    
    dependency_tree = Dependency(args.conll)
    
    offset_modes = None
    
    if args.evaluate_only:
        if not args.json:
            raise ValueError("JSON with offset modes required in evaluate only mode!")

        with open(args.json, 'r') as inj:
            offset_modes = json.load(inj)
    
    else:
        offset_modes = dependency_tree.calc_offset_modes()
    
    results = defaultdict(dict)
    clausal_relations = ('adj-modifier', 'adv-modifier', 'auxiliary', 'compound', 'conjunct', 'determiner',
                         'noun-modifier', 'num-modifier', 'object', 'subject', 'case', 'mark')
    clausal_sum = 0.
    
    non_clausal_relations = ('adj-clause', 'adv-clause', 'clausal', 'clausal-subject', 'parataxis')
    non_clausal_sum = 0.
    
    if args.json:
        with open(args.json, 'w') as outj:
            json.dump(dependency_tree.calc_offset_modes(), fp=outj)
    
    if args.report_result:
        
        positional_baseline = dependency_tree.eval_positional_baseline(offset_modes)
        with open(args.report_result, 'w') as outr:
            outr.write('labe\tpositional_baseline\n')
            for rel in clausal_relations:
                if rel not in positional_baseline:
                    continue
                outr.write(f"{rel}\t{positional_baseline[rel]}\n")
                clausal_sum += positional_baseline[rel]
            outr.write('\n')
            for rel in non_clausal_relations:
                if rel not in positional_baseline:
                    continue
                outr.write(f"{rel}\t{positional_baseline[rel]}\n")
                non_clausal_sum += positional_baseline[rel]
            outr.write('\n')
            outr.write(f'Clausal mean:\t{clausal_sum / len(clausal_relations)}\n')
            outr.write(f'Non Clausal mean:\t{non_clausal_sum / len(non_clausal_relations)}\n')