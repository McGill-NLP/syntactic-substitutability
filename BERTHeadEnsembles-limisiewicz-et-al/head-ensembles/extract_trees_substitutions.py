#!/usr/bin/env python3

import numpy as np
import argparse
import json

from dependency import Dependency
from metrics import UAS, LAS, UUAS
from tqdm import tqdm
from attention_wrapper_substitutions import AttentionWrapper

import networkx as nx
from networkx.algorithms import tree

DEPACC_THRESHOLD = 0.6


def print_tikz(prediction_edges, sent_idcs, dependency , out_tikz_file):
	''' Turns edge sets on word (nodes) into tikz dependency LaTeX. '''
	uas_m = UAS(dependency)
	with open(out_tikz_file, 'w') as fout:
		for sent_preds, sid in zip(prediction_edges, sent_idcs):
			tokens = dependency.tokens[sid]
			uas_m.reset_state()
			uas_m([sid],[sent_preds])
			if len(tokens) < 10 and uas_m.result() > 0.6:

				sent_golds = dependency.unlabeled_relations[sid]

				string = """\\begin{dependency}[hide label, edge unit distance=.5ex]
		  \\begin{deptext}[column sep=0.05cm]
		  """
				string += "\\& ".join([x.replace('$', '\$').replace('&', '+') for x in tokens]) + " \\\\" + '\n'
				string += "\\end{deptext}" + '\n'
				for i_index, j_index in sent_golds:
					if i_index >= 0 and j_index >= 0:
						string += '\\depedge{{{}}}{{{}}}{{{}}}\n'.format(i_index + 1, j_index + 1, '.')
				for i_index, j_index in sent_preds:
					if i_index >= 0 and j_index >= 0:
						string += '\\depedge[edge style={{blue!60!}}, edge below]{{{}}}{{{}}}{{{}}}\n'.format(i_index +1,
																											  j_index +1, '.')
				string += '\\end{dependency}\n'
				fout.write('\n\n')
				fout.write(string)



def extract_trees(bert_attns, relation_heads_d2p, relation_heads_p2d, weights_d2p, weights_p2d, roots):
	"""Dependency extraction from BERT attenions
	Parameters:
		bert_attns (AttentionWrapper): object storing attention matrices and considered sentence indices
		relation_heads_d2p (dict): maps relation labels to list of tuples (layer_idx, head_idx) of heads selected in direction dependent -> parent
		relation_heads_p2d (dict): the same as above but in direction parent -> dependent
		weights_d2p (dict): maps relation label to metric of selected head ensemble in in direction dependent -> parent
		weights_p2d (dict): the same as above but in direction parent -> dependent
	Returns:
		extracted_unlabeled: list of tuples of extracted dependency relations (dependent_idx, parent_idx)
		extracted_unlabeled: list of tuples of extracted dependency relations (dependent_idx, parent_idx, relation_label)
	"""
	extracted_unlabeled = list()
	extracted_labeled = list()
	for idx, sent_idx in tqdm(enumerate(bert_attns.sentence_idcs), desc='Extracting trees from matrices'):
		root = roots[sent_idx]
		dependency_graph = nx.MultiDiGraph()
		dependency_graph.add_nodes_from(range(len(bert_attns.tokens_grouped[sent_idx])))

		edge2relation_label = dict()
		for relation in relation_heads_d2p.keys():

			layer_idx_d2p, head_idx_d2p = zip(*relation_heads_d2p[relation])
			ensemble_matrix_d2p = bert_attns.matrices[idx][layer_idx_d2p, head_idx_d2p, :, :].mean(axis=0).transpose()
			ensemble_matrix_d2p[:, root] = 0.001
			np.fill_diagonal(ensemble_matrix_d2p, 0.001)
			ensemble_matrix_d2p = np.clip(ensemble_matrix_d2p, 0.001, 0.999)

			layer_idx_p2d, head_idx_p2d = zip(*relation_heads_p2d[relation])
			ensemble_matrix_p2d = bert_attns.matrices[idx][layer_idx_p2d, head_idx_p2d, :, :].mean(axis=0)
			ensemble_matrix_p2d[:, root] = 0.001
			np.fill_diagonal(ensemble_matrix_p2d, 0.001)
			ensemble_matrix_p2d = np.clip(ensemble_matrix_p2d, 0.001, 0.999)

			weight_p2d = weights_p2d[relation] ** 5
			weight_d2p = weights_d2p[relation] ** 5
			ensemble_matrix = (weight_d2p * np.log(ensemble_matrix_d2p) + weight_p2d * np.log(ensemble_matrix_p2d)) / (
						weight_d2p + weight_p2d)

			ensemble_graph = nx.from_numpy_matrix(ensemble_matrix, create_using=nx.DiGraph)

			# Unfortunately this is necessary, because netwokx multigraph loses information about edges
			for u, v, d in ensemble_graph.edges(data=True):
				edge2relation_label[(u, v, d['weight'])] = relation

			dependency_graph.add_edges_from(ensemble_graph.edges(data=True), label=relation)

		dependency_aborescene = tree.branchings.maximum_spanning_arborescence(dependency_graph)

		extracted_unlabeled.append(
			[(dep, parent) for parent, dep in dependency_aborescene.edges(data=False)] + [(root, -1)])
		extracted_labeled.append([(dep, parent, edge2relation_label[(parent, dep, edge_data['weight'])])
								  for parent, dep, edge_data in dependency_aborescene.edges(data=True)] + [
									 (root, -1, 'root')])

	return extracted_unlabeled, extracted_labeled


if __name__ == '__main__':
	ap = argparse.ArgumentParser()
	ap.add_argument("attentions", type=str, help="NPZ file with attentions")
	ap.add_argument("tokens", type=str, help="Labels (tokens) separated by spaces")
	ap.add_argument("conll", type=str, help="Conll file for head selection.")
	ap.add_argument("json", type=str, help="Json file with head ensemble")
	
	# other arguments
	ap.add_argument("--report-result", type=str, default=None, help="File where to save the results.")
	ap.add_argument("-s", "--sentences", nargs='*', type=int, default=None,
	                help="Only use the specified sentences; 0-based")
	
	args = ap.parse_args()
	
	dependency = Dependency(args.conll, args.tokens)
	
	head_ensembles = dict()
	ensembles_d2p = dict()
	ensembles_p2d = dict()
	depacc_d2p = dict()
	depacc_p2d = dict()
	
	with open(args.json, 'r') as inj:
		head_ensembles = json.load(inj)
	
	# considered_relations = (Dependency.LABEL_ALL,)
	
	considered_relations = ('adj-modifier', 'adv-modifier', 'auxiliary', 'compound', 'conjunct', 'determiner',
							'noun-modifier', 'num-modifier', 'object', 'other', 'subject', 'cc', 'case', 'mark')

	for relation in considered_relations:
		ensembles_d2p[relation] = head_ensembles[relation + '-d2p']['ensemble']
		depacc_d2p[relation] = head_ensembles[relation + '-d2p']['max_metric']
		ensembles_p2d[relation] = head_ensembles[relation + '-p2d']['ensemble']
		depacc_p2d[relation] = head_ensembles[relation + '-p2d']['max_metric']
		
	bert_attns = AttentionWrapper(args.attentions, dependency.wordpieces2tokens, args.sentences)
	extracted_unlabeled, extracted_labeled = extract_trees(bert_attns, ensembles_d2p, ensembles_p2d, depacc_d2p, depacc_p2d, dependency.roots)
	
	uas_m = UAS(dependency)
	uas_m(bert_attns.sentence_idcs, extracted_unlabeled)
	uas_res, uas_rel_wise = uas_m.result()
	print("UAS: ")
	print(uas_res)

	uas_m = UUAS(dependency)
	uas_m(bert_attns.sentence_idcs, extracted_unlabeled)
	uas_res, uas_rel_wise = uas_m.result()
	print("UUAS: ")
	print(uas_res)

	las_m = LAS(dependency)
	las_m(bert_attns.sentence_idcs, extracted_labeled)
	las_rel_wise = las_m.rel_wise_prec
	las_res, las_rel_wise = las_m.result()
	print("LAS: ")
	print(las_res)
	

	if args.report_result:
		with open(args.report_result, 'w') as res_file:
			res_file.write(f"UUAS: {uas_res}\n")
			res_file.write(f"LAS: {las_res}\n")
			res_file.write("UUAS REL_WISE\n")
			for k in uas_rel_wise.keys():
				res_file.write(k + "\t" + str(uas_rel_wise[k]) + "\n")
			res_file.write("LAS REL_WISE\n")
			for k in las_rel_wise.keys():
				res_file.write(k + "\t" + str(las_rel_wise[k]) + "\n")
		#print_tikz(extracted_unlabeled, bert_attns.sentence_idcs, dependency, args.report_result+".tikz")		
		
		
	else:
		print(f"UAS result for extracted tree: {uas_res}, LAS: {las_res}")