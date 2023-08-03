
from tools import dependency, sentence_attentions

import argparse
from collections import defaultdict, namedtuple

import numpy as np



RelData = namedtuple('RelData','layers heads transpose d2p')



# soft pos mask (BEST)
# relation_rules  = {'adj-clause-p2d': RelData([3, 4, 7, 6, 5, 7], [3, 5, 6, 6, 9, 10],False, False),
#     'adj-modifier-d2p': RelData([3, 7, 4, 5], [9, 2, 5, 0],False, True),
#     'adv-clause-p2d': RelData([5, 3, 4, 5, 1], [8, 3, 8, 11, 5],False, False),
#     'adv-modifier-p2d': RelData([7, 5, 6, 4, 9, 4], [7, 1, 9, 7, 3, 10],False, True),
#     'apposition-p2d': RelData([3], [3],False, False),
#     'auxiliary-d2p': RelData([7, 3, 5, 4, 1, 1], [2, 9, 9, 11, 6, 1],False, True),
#     'clausal-p2d': RelData([5, 7, 7, 7, 4, 3, 5, 0, 6, 1], [8, 10, 1, 6, 5, 3, 11, 8, 6, 5],False, False),
#     'clausal-d2p': RelData([6, 5, 7, 7], [2, 3, 0, 9],False, True),
#     'compound-d2p': RelData([3, 7], [9, 2],False, True),
#     'conjunct-d2p': RelData([6, 9, 8, 4, 3, 11], [0, 6, 4, 10, 11, 9],False, True),
#     'determiner-d2p': RelData([7, 3, 4], [10, 9, 5],False, True),
#     # 'i object-d2p': RelData([5], [1],False, True),
#     'noun-modifier-d2p': RelData([6, 3, 7, 6], [9, 11, 9, 2],False, True),
#     'num-modifier-p2d': RelData([7, 4, 0, 6, 5], [11, 7, 2, 9, 9],False, False),
#     'object-d2p': RelData([6, 3, 6], [9, 11, 10],False, True),
#     'other-d2p': RelData([5, 4, 8, 6, 3, 5, 7, 7, 7, 5, 1, 1], [0, 5, 5, 6, 9, 7, 1, 2, 6, 9, 6, 1],False, True),
#     'punctuation-d2p': RelData([7, 8, 8, 5, 7, 8], [1, 5, 0, 7, 6, 4],False, True),
#     'subject-p2d': RelData([7, 4, 6], [4, 10, 4],False, False)}

# diagonal mask
# relation_rules = {'adj-clause-p2d': RelData([4, 7, 6, 0], [5, 6, 5, 8],False, False),
#     'adj-modifier-d2p': RelData([3, 7, 6, 5, 7, 8, 0, 2], [9, 10, 5, 7, 6, 5, 8, 11],False, True),
#     'adv-clause-d2p': RelData([4, 4, 5, 4, 2, 0], [9, 3, 4, 4, 7, 7],False, True),
#     'adv-modifier-d2p': RelData([7, 5, 6, 8, 7, 3, 10, 0, 6, 0], [6, 7, 5, 5, 10, 10, 10, 8, 4, 11],False, True),
#     'apposition-p2d': RelData([0, 9], [8, 0],False, False),
#     'auxiliary-d2p': RelData([3, 8, 7, 5, 4, 7, 10], [9, 5, 6, 0, 5, 10, 10],False, True),
#     'clausal subject-p2d': RelData([8, 0, 0, 0], [10, 8, 5, 1],False, False),
#     'clausal-d2p': RelData([7, 6, 4, 8, 5, 0, 0, 1, 0], [0, 2, 6, 8, 4, 5, 9, 11, 7],False, True),
#     'compound-d2p': RelData([3, 7, 6, 7, 0], [9, 6, 5, 10, 8],False, True),
#     'conjunct-d2p': RelData([4, 6, 4, 9, 5, 1, 0, 4, 6], [3, 0, 9, 6, 4, 10, 1, 4, 8],False, True),
#     'determiner-d2p': RelData([7, 3, 4, 8], [10, 9, 5, 10],False, True),
#     #'i object-d2p': RelData([6], [9],False, True),
#     'noun-modifier-p2d': RelData([4, 0, 9, 5, 3, 0, 0], [5, 8, 1, 8, 3, 1, 5],False, False),
#     'num-modifier-d2p': RelData([7, 6, 3, 8, 7, 6, 0, 10], [10, 5, 10, 5, 6, 4, 11, 10],False, True),
#     'object-d2p': RelData([7, 6, 4, 5, 3], [9, 9, 6, 3, 8],False, True),
#     'other-d2p': RelData([7, 4, 8, 6, 3, 0], [10, 5, 5, 5, 10, 8],False, True),
#     'punctuation-p2d': RelData([11, 10, 2, 11, 7, 7], [6, 7, 2, 2, 8, 7],False, False),
#     'subject-p2d': RelData([7, 4], [11, 10],False, False)}

# diagonal mask2
relation_rules  = {'adj-clause-p2d': RelData([4, 7, 6, 0], [5, 6, 5, 8],False, False),
    'adj-modifier-d2p': RelData([3, 7, 6, 5, 7, 8, 0, 2], [9, 10, 5, 7, 6, 5, 8, 11],False, True),
    'adv-clause-p2d': RelData([4, 5, 5, 0, 4, 5, 11, 8, 3, 0], [3, 4, 5, 8, 9, 8, 8, 7, 1, 4],False, False),
    'adv-modifier-d2p': RelData([7, 5, 6, 8, 7, 3, 10, 0, 6, 0], [6, 7, 5, 5, 10, 10, 10, 8, 4, 11],False, True),
    'apposition-p2d': RelData([0, 9], [8, 0],False, False),
    'auxiliary-d2p': RelData([3, 8, 7, 5, 4, 7, 10], [9, 5, 6, 0, 5, 10, 10],False, True),
    'clausal subject-p2d': RelData([8, 0, 0, 0], [10, 8, 5, 1],False, False),
    'clausal-p2d': RelData([5, 4, 7, 5, 0, 7, 4], [7, 5, 6, 8, 8, 1, 8],False, False),
    'compound-d2p': RelData([3, 7, 6, 7, 0], [9, 6, 5, 10, 8],False, True),
    'conjunct-d2p': RelData([4, 6, 4, 9, 5, 1, 0, 4, 6], [3, 0, 9, 6, 4, 10, 1, 4, 8],False, True),
    'determiner-d2p': RelData([7, 3, 4, 8], [10, 9, 5, 10],False, True),
    #'i object-d2p': RelData([6], [9],False, True),
    'noun-modifier-p2d': RelData([4, 0, 9, 5, 3, 0, 0], [5, 8, 1, 8, 3, 1, 5],False, False),
    'num-modifier-d2p': RelData([7, 6, 3, 8, 7, 6, 0, 10], [10, 5, 10, 5, 6, 4, 11, 10],False, True),
    'object-d2p': RelData([7, 6, 4, 5, 3], [9, 9, 6, 3, 8],False, True),
    'other-d2p': RelData([7, 4, 8, 6, 3, 0], [10, 5, 5, 5, 10, 8],False, True),
    'punctuation-d2p': RelData([4, 8, 3, 7, 3], [5, 5, 10, 5, 9],False, True),
    'subject-p2d': RelData([7, 4], [11, 10],False, False)
    }
# based on tiny set 10 examples
# relation_rules  = {'adj-clause-p2d': RelData([1], [1],False, False),
# 	'adj-modifier-d2p': RelData([3], [9],False, True),
# 	'adv-clause-p2d': RelData([1], [10],False, False),
# 	'adv-modifier-p2d': RelData([2], [10],False, False),
# 	'apposition-p2d': RelData([2], [1],False, False),
# 	'auxiliary-p2d': RelData([3], [8],False, False),
# 	'compound-d2p': RelData([5], [11],False, True),
# 	'conjunct-d2p': RelData([11, 4], [9, 7],False, True),
# 	'determiner-p2d': RelData([7], [9],False, False),
# 	'noun-modifier-p2d': RelData([3], [9],False, False),
# 	'num-modifier-p2d': RelData([11], [11],False, False),
# 	'object-d2p': RelData([9], [3],False, True),
# 	'other-d2p': RelData([6, 3, 5, 5], [6, 9, 9, 7],False, True),
# 	'punctuation-d2p': RelData([6, 5], [6, 0],False, True),
# 	'subject-p2d': RelData([4, 6], [10, 4],False, False)}


# more detailed relationship labels
# relation_rules  = {'acl-p2d': RelData([3, 4, 7, 6, 5, 7], [3, 5, 6, 6, 9, 10],False, False),
# 	'advcl-p2d': RelData([5, 3, 4, 5, 1], [8, 3, 8, 11, 5],False, False),
# 	'advmod-p2d': RelData([7, 5, 6, 4, 9, 4], [7, 1, 9, 7, 3, 10],False, False),
# 	'amod-d2p': RelData([3, 7, 4, 5], [9, 2, 5, 0],False, True),
# 	'appos-p2d': RelData([3], [3],False, False),
# 	'aux-d2p': RelData([7, 3, 5, 4, 1, 1], [2, 9, 9, 11, 6, 1],False, True),
# 	'case-p2d': RelData([0, 5, 3, 9, 1], [3, 10, 11, 3, 4],False, False),
# 	'cc-p2d': RelData([5, 3, 0, 6, 7, 7], [10, 11, 3, 9, 9, 4],False, False),
# 	'ccomp-d2p': RelData([6, 5, 7, 7], [2, 3, 0, 9],False, True),
# 	'compound-d2p': RelData([3, 7], [9, 2],False, True),
# 	'conj-p2d': RelData([5, 9, 7, 3, 10, 4, 11], [5, 6, 8, 3, 5, 8, 8],False, False),
# 	'csubj-p2d': RelData([7, 7], [1, 10],False, False),
# 	'det-p2d': RelData([0, 6, 2, 6, 1], [3, 9, 1, 3, 4],False, False),
# 	'discourse-p2d': RelData([11], [11],False, False),
# 	'expl-p2d': RelData([5], [9],False, False),
# 	'fixed-d2p': RelData([8], [6],False, True),
# 	'flat-p2d': RelData([8], [5],False, False),
# 	'iobj-p2d': RelData([3], [9],False, False),
# 	'mark-p2d': RelData([7, 6, 5, 0, 6, 7, 8], [9, 9, 6, 3, 10, 4, 11],False, False),
# 	'nmod-d2p': RelData([6, 3, 7, 6], [9, 11, 9, 2],False, True),
# 	'nsubj-d2p': RelData([7, 6, 5, 5, 1, 7, 5, 7, 4], [2, 6, 9, 0, 6, 1, 11, 8, 11],False, True),
# 	'nummod-p2d': RelData([7, 4, 0, 6, 5], [11, 7, 2, 9, 9],False, False),
# 	'obj-d2p': RelData([6, 3, 6], [9, 11, 10],False, True),
# 	'parataxis-p2d': RelData([5], [5],False, False),
# 	'punct-p2d': RelData([5, 6], [10, 11],False, False),
# 	'vocative-p2d': RelData([6], [9],False, False),
# 	'xcomp-d2p': RelData([6, 3, 5, 5], [9, 11, 6, 3],False, True)}


def rewrite_conllu(conllu_file, conllu_out_pred, conllu_out_gold, break_after=1000):
	
	CONLLU_ID = 0
	CONLLU_LABEL = 7
	CONLLU_HEAD = 6
	
	reverse_label_map = {value: key for key, value in dependency.label_map.items()}
	reverse_label_map['other'] = 'dep'
	
	print(reverse_label_map)
	
	lengths = []
	uas= []
	length_sent = 0
	out_lines = []
	out_lines_gold =[]
	with open(conllu_file, 'r') as in_conllu:
		sentid = 0
		for line in in_conllu:
			if sentid > break_after:
				break
			if line == '\n':
				out_lines.append(line.strip())
				out_lines_gold.append(line.strip())
				uas_sent = (np.array(list(map(int, gold.ravel() != 'no edge'))) * np.array(
					list(map(int, pred.ravel() != 'no edge')))).sum() / np.array(list(map(int,gold.ravel()!='no edge'))).sum()
				uas.append(uas_sent)
				lengths.append(length_sent)
				print(f"Processed sentence {sentid}", flush=True)
				sentid += 1
				
			elif line.startswith('#'):
				if line.startswith('# sent_id'):
					out_lines.append(line.strip() + '/pred')
					out_lines_gold.append(line.strip() + '/gold')
					pred, gold = multigraph_aborescene(sentid)
					length_sent = 0
				else:
					out_lines_gold.append(line.strip())
					out_lines.append(line.strip())
			else:
				fields = line.strip().split('\t')
				out_lines_gold.append(line.strip())
				if fields[CONLLU_ID].isdigit():
					length_sent += 1
					if fields[CONLLU_LABEL].strip() != 'root':
						col = pred.transpose()[int(fields[CONLLU_ID]) - 1]

						x = np.argwhere(col != 'no edge')
						x = x.item()
						lab = reverse_label_map[col[x][:-4]]
						
						fields[CONLLU_HEAD] = str(x + 1)
						fields[CONLLU_LABEL] = lab
				
				out_lines.append('\t'.join(fields))
	
	with open(conllu_out_pred, 'w') as out_conllu:
		out_conllu.write('\n'.join(out_lines))
		
	with open(conllu_out_gold, 'w') as out_conllu:
		out_conllu.write('\n'.join(out_lines_gold))

	print("mena uas:")
	print(np.mean(np.array(uas)))
	print("length uas corr coef:")
	print(np.corrcoef(np.array(uas), np.array(lengths)))


def multigraph_aborescene(sentence_index, dependency):
	matrices, sentence_id = next(attention_gen)

	assert sentence_index == sentence_id
	
	words_list = common_tokens[sentence_index]
	words = ' '.join(words_list)
	
	edge_labeled = {(h, d): l for d, h, l, p in dependency_rels[sentence_index] if l != 'root'}
	root_ord = 0
	for d, h, l, p in dependency_rels[sentence_index]:
		if l == 'root':
			root_ord = d
			break
	

	DG = nx.DiGraph()
	DG.add_edges_from(edge_labeled.keys())
	
	labels = {}
	for node in DG.nodes():
		labels[node] = words_list[node]

	MultiAttention = nx.MultiDiGraph()
	MultiAttention.add_nodes_from(DG.nodes())
	
	multi_edge2label = dict()
	for relation, rules in relation_rules.items():
		aggr_matrix = np.mean(np.array(matrices)[rules.layers, rules.heads, :, :], axis=0)

		if rules.d2p == True:
			aggr_matrix = aggr_matrix.transpose()
		aggr_matrix[:, root_ord] = 0.001
		np.fill_diagonal(aggr_matrix, 0.001)
		aggr_matrix = np.log(aggr_matrix/(1-aggr_matrix))

		
		AG = nx.from_numpy_matrix(aggr_matrix, create_using=nx.DiGraph)
		
		for u, v, d in AG.edges(data=True):
			multi_edge2label[(u, v, d['weight'])] = relation
		# incldue statistical info about pos:
		
		MultiAttention.add_edges_from(AG.edges(data=True), label=relation)
	
	AttentionAborescene = tree.branchings.maximum_spanning_arborescence(MultiAttention)
	espanning = AttentionAborescene.edges(data=True)
	weights = [max(d['weight'] * 20, 1) for _, _, d in espanning]
	attention_labels = {(u, v): multi_edge2label[(u, v, d['weight'])] for u, v, d in espanning}
	espanning = [(u, v) for (u, v, d) in espanning]
	posA = nx.spring_layout(AttentionAborescene)

	alabelm = np.full((len(aggr_matrix), len(aggr_matrix)), 'no edge', dtype='U24')
	dlabelm = np.full((len(aggr_matrix), len(aggr_matrix)), 'no edge', dtype='U24')
	
	for aedge, ael in attention_labels.items():

		alabelm[aedge[0], aedge[1]] = ael
	#         else:
	#             alabelm[aedge[1],aedge[0]] = ael
	
	for dedge, deel in edge_labeled.items():
		
		deel = dependency.transform_label(deel)
		
		if deel + '-d2p' in relation_rules:
			dlabelm[dedge[0], dedge[1]] = deel + '-d2p'
		elif deel + '-p2d' in relation_rules:
			dlabelm[dedge[0], dedge[1]] = deel + '-p2d'
		
		elif 'other-d2p' in relation_rules:
			dlabelm[dedge[0], dedge[1]] = 'other-d2p'
		elif 'other-p2d' in relation_rules:
			dlabelm[dedge[1], dedge[0]] = 'other-p2d'
	
	return alabelm, dlabelm

	
if __name__ == '__main__':
	ap = argparse.ArgumentParser()
	ap.add_argument("-a", "--attentions", required=True, help="NPZ file with attentions")
	ap.add_argument("-t", "--tokens", required=True, help="Labels (tokens) separated by spaces")
	
	ap.add_argument("-T", "--train-conllu", help="Conllu file for training POS",
	                default='/net/projects/LSD/attention_tomasz/lsd/attention-analysis/naacl2019/graph-extraction/entrain.conllu')
	
	ap.add_argument("-c", "--conllu", help="Eval against the given conllu file")
	
	ap.add_argument("-o", "--output-pred")
	ap.add_argument("-g", "--output-gold")
	
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
	dependency_rels = dependency.read_conllu_labeled(args.conllu)
	
	grouped_tokens, common_tokens  = dependency.group_wordpieces(tokens_loaded, args.conllu)
	
	attention_gen = sentence_attentions.generate_matrices(attentions_loaded, grouped_tokens, args.eos, args.no_softmax,
	                                                      args.maxlen, args.sentences)
	
	pos_frame = dependency.conllu2freq_frame(args.train_conllu)
	
	rewrite_conllu(args.conllu, args.output_pred, args.output_gold, break_after=300)
