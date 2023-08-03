import numpy as np
import sys
from itertools import chain


def pos_soft_mask(sentence_rels, relation_label, pos_frame):
	pos_mask = np.zeros((len(sentence_rels), len(sentence_rels)))
	for idx in range(len(sentence_rels)):
		for jdx in range(idx+1,len(sentence_rels)):
			i_pos = sentence_rels[idx][3]
			j_pos = sentence_rels[jdx][3]
			if relation_label not in pos_frame:
				pos_mask[idx,jdx] = 0
				pos_mask[jdx, idx] = 0
			else:
				pos_mask[idx,jdx] = pos_frame[relation_label][(i_pos, j_pos)]
				pos_mask[jdx, idx] = pos_frame[relation_label][(j_pos, i_pos)]
		
	return pos_mask


def pos_hard_mask(sentence_rels, relation_label, pos_frame, thr=0.005):
	pos_mask = np.zeros((len(sentence_rels), len(sentence_rels)))
	for idx in range(len(sentence_rels)):
		for jdx in range(idx + 1, len(sentence_rels)):
			i_pos = sentence_rels[idx][3]
			j_pos = sentence_rels[jdx][3]
			if relation_label in pos_frame and pos_frame[relation_label][(i_pos, j_pos)] >= thr:
				pos_mask[idx, jdx] = 1.0
			if relation_label in pos_frame and pos_frame[relation_label][(j_pos, i_pos)] >= thr:
				pos_mask[jdx, idx] = 1.0
	return pos_mask


def diagonal_mask(sentnce_rels):
	mask = np.ones((len(sentnce_rels), len(sentnce_rels)))
	np.fill_diagonal(mask, 0)
	return mask


def aggregate_subtoken_matrix(attention_matrix, tokens_grouped):
	# this functions connects subtokens and aggregates their attention.
	midres_matrix = np.zeros((len(tokens_grouped), attention_matrix.shape[0]))
	
	for tok_id, wp_ids in enumerate(tokens_grouped):
		midres_matrix[tok_id, :] = np.mean(attention_matrix[wp_ids, :], axis=0)
	
	res_matrix = np.zeros((len(tokens_grouped), len(tokens_grouped)))
	
	for tok_id, wp_ids in enumerate(tokens_grouped):
		res_matrix[:, tok_id] = np.sum(midres_matrix[:, wp_ids], axis=1)

	return res_matrix


def generate_matrices(attentions_loaded, tokens_grouped, eos=True, no_softmax=False, maxlen=1000, sentences=None):
	sentences_count = len(tokens_grouped)
	layers_count = attentions_loaded['arr_0'].shape[0]
	heads_count = attentions_loaded['arr_0'].shape[1]
	for sentence_index in range(sentences_count):
		
		if sentences and sentence_index not in sentences:
			continue

		sentence_id = 'arr_' + str(sentence_index)
		tokens_count = attentions_loaded[sentence_id].shape[2]
		
		if eos:
			tokens_count -= 1
		groups_list = tokens_grouped[sentence_index]
		
		if groups_list is None:
			print('Token mismatch sentence skipped', sentence_index, file=sys.stderr)
			yield None, sentence_index
			continue
		
		# check maxlen
		if not len(groups_list) <= maxlen:
			print('Too long sentence, skipped', sentence_index, file=sys.stderr)
			yield None, sentence_index
			continue
		
		ungrouped_list = list(chain.from_iterable(groups_list))
		# NOTE sentences truncated to 64 tokens
		# assert len(tokens_list) == tokens_count, "Bad no of tokens in sent " + str(sentence_index)
		assert len(ungrouped_list) >= tokens_count, "Bad no of tokens in sent " + str(sentence_index)
		if len(ungrouped_list) > tokens_count:
			print('Too long sentence, skipped', sentence_index, file=sys.stderr)
			yield None, sentence_index
			continue
		
		words_count = len(groups_list)
		
		# for visualisation -- vis[layer][head]
		matrices = list()
		
		for layer in range(layers_count):
			layer_deps = list()  # for vis
			for head in range(heads_count):
				matrix = attentions_loaded[sentence_id][layer][head]
				if eos:
					matrix = matrix[:-1, :-1]
				# the max trick -- for each row subtract its max
				# from all of its components to get the values into (-inf, 0]
				if not no_softmax:
					matrix = matrix - np.max(matrix, axis=1, keepdims=True)
					# softmax
					exp_matrix = np.exp(matrix)
					deps = exp_matrix / np.sum(exp_matrix, axis=1, keepdims=True)
				else:
					deps = matrix / np.sum(matrix, axis=1, keepdims=True)
				deps = aggregate_subtoken_matrix(deps, groups_list)
				layer_deps.append(deps)
			# layer_matrix = layer_matrix + deps
			matrices.append(layer_deps)
		yield matrices, sentence_index
