import sys
from transformers import BertTokenizerFast, BertModel 
import torch
from torch.nn.functional import normalize
import numpy as np
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
import pickle
import copy
from tqdm import tqdm
from stanza.utils.conll import CoNLL

def get_attentions(sents, model, tokenizer):
    with torch.no_grad():
        inputs = tokenizer(sents, return_tensors='pt', padding=True).to('cuda')
        padding = inputs['attention_mask']
        padding = [np.where(p.cpu() == 0)[0][0] if len(np.where(p.cpu() == 0)[0]) > 0 else -1 for p in padding]
        attention = model(**inputs)
        attention = attention.attentions
        unbatched_atts = [torch.unbind(a, 0) for a in attention]
        atts_list = [[layer[i] for layer in unbatched_atts] for i in range(len(unbatched_atts[0]))]
        atts_list = [[torch.unsqueeze(a, 0).cpu()if padding[i] == -1 else torch.unsqueeze(a, 0).cpu()[:, :, :padding[i], :padding[i]] for a in s] for i, s in enumerate(atts_list)]
        atts_list = [tuple(a) for a in atts_list]
    return atts_list

def delete_cls(attention, normalize=False):
    attention_new = []
    for arr in attention:
        a = arr[:, :, 1:-1, 1:-1]
        if normalize:
            a = a / np.linalg.norm(a, ord=1, axis=2, keepdims=True)
        attention_new.append(a)
    return tuple(attention_new)

# gets the splits for each individual sentence so that attention weights can be averaged if a word is broken up into different tok
def get_spans(s, tokenizer):
    s = s.split()
    tokenized = tokenizer(s, is_split_into_words=True, return_offsets_mapping=True)
    tokenized = tokenized['offset_mapping'][1:-1]
    #print(tokenized)
    #print(s)
    j = 0
    spans = []
    while j < len(tokenized):
        if tokenized[j][0] == 0:
            spans.append(j)
            j += 1
        else:
            span = [j]
            j += 1
            while j < len(tokenized) and tokenized[j][0] != 0:
                span.append(j)
                j += 1
            spans[-1] = (spans[-1],) + tuple(span)
    return spans

def average_between_sentences(attentions, layer=7, p=0):
    not_included = []
    try:
        #averaged_heads = [np.sum(np.stack(attentions[i]), 0) for i in range(len(attentions))]
        if p < 0:
            averaged_heads = [attentions[i][layer] for i in range(len(attentions)) if i != 0]
            if len(averaged_heads) == 0:
                return None
        else:
            averaged_heads = [attentions[i][layer] for i in range(len(attentions))]
        averaged_layers = [np.sum(np.stack(averaged_heads[i]), 1) for i in range(len(averaged_heads))]
        averaged_sentences = np.sum(np.stack(averaged_layers), 0)
        averaged_sentences = averaged_sentences / np.linalg.norm(averaged_sentences, ord=1, axis=2, keepdims=True)
    except ValueError: # there should not be any errors here now since the spans are based on splitting sentences on whitespace
        return
    return averaged_sentences

# takes the attention and sentence spans to return a merged version of the attention
def combine_subwords(attention, spans):
    new_attention_1 = []
    for layer in attention:
        #print(layer.size())
        start = 0
        end = 0
        splits = []
        for sp in spans:
            if type(sp) == tuple:
                start = end
                end = sp[-1] + 1
                layer[:, :, :, sp[0]] = layer[:, :, :, sp].mean(3)
                splits.append(layer[:, :, :, start:sp[0] + 1])
        if end < int(layer.shape[-1]):
            splits.append(layer[:, :, :, end:])
        new_layer = torch.cat(splits, 3)
        new_attention_1.append(new_layer)
    
    new_attention_2 = []
    for j, layer in enumerate(new_attention_1):
        start = 0
        end = 0
        splits = []
        for sp in spans:
            if type(sp) == tuple:
                start = end
                end = sp[-1] + 1
                layer[:, :, sp[0]] = normalize(layer[:, :, sp].sum(2), p=1, dim=-1)
                splits.append(layer[:, :, start:sp[0] + 1])
        if end < int(layer.shape[-2]):
            splits.append(layer[:, :, end:])
        new_layer = torch.cat(splits, 2)
        new_attention_2.append(new_layer)
    return tuple(new_attention_2)

# gets all the attentions and outputs a dictionary of the attentions for each sentence, including the original
# deletes CLS and SEP tokens and combines subword attentions
# averages between sentences
def get_all_atts(sents, model, tokenizer, l=7):
    """
    Parameters:
            sents (dict of str : list of (int, [str])): The list of sentences and perturbations
            model (transformers.Model): A pretrained language model
            tokenize (transformers.Tokenizer): A tokenizer
            l (int)

    Returns:
            original_sents (dict of str : list of (int, Tensor)) : The attentions of each of the original sentences which should all be the same
            perturbed_sents (dict of str : list of (int, Tensor)) : The attentions of the sentences perturbed at each position

    """
    new_process = {}    
    original_sents = {}
    perturbed_sents = {}
    
    for i, k in enumerate(tqdm(sents.keys())):
        perturbed_sents[k] = []
        original_sents[k] = []
        for word_position in sents[k]:
            position = word_position[0]
            position_attentions = []
            attentions = get_attentions(word_position[1], model, tokenizer)
            for i, s in enumerate(word_position[1]):
                t = copy.deepcopy(attentions[i])
                t = delete_cls(t, normalize=True)
                spans = get_spans(s, tokenizer)
                t = combine_subwords(t, spans)
                position_attentions.append(t)
            new_process[k] = position_attentions
            original_s = [copy.deepcopy(position_attentions[0])]
            if len(position_attentions) >= 1:
                perturbed_sents[k].append((position, average_between_sentences(position_attentions, layer=l, p=position)))
                original_sents[k].append((position, average_between_sentences(original_s, layer=l, p=position)))
            else:
                continue
    return original_sents, perturbed_sents

def np_to_edge_list(matrix):
    e_list = []
    for i in range(len(list(matrix[0]))):
        for j in range(len(list(matrix[0]))):
            e_list.append((i, j, matrix[i][j]))
    return e_list

def get_graphs(avg_atts, trees=False):
    graphs = {}
    for k in avg_atts.keys():
        graphs[k] = []
        for word_position in avg_atts[k]:
            position = word_position[0]
            if not word_position[1] is None:
                a = np.zeros_like(word_position[1][0])
                a[position] = word_position[1][0][:, position]                 
                a_elist = np_to_edge_list(a)
                graph_4 = nx.MultiGraph()
                graph_4.add_weighted_edges_from(a_elist)
            else:
                graph_4 = nx.MultiGraph()
            if trees:
                graph = get_max_trees(graph)
            graphs[k].append((position, graph_4))
    return graphs

def get_max_trees(graph):
    G_tree = nx.maximum_spanning_tree(graph)
    return G_tree

def rb_baseline(gold_standard_rels, graphs):
    sentences = graphs.keys()
    num_words = sum([len(sen) + 1 for sen in gold_standard_rels.values()])
    recall = {}
    precision = {}
    for s in sentences:
        s_rec = []
        s_prec = []
        gold_edges = gold_standard_rels[s]
        gold_edges = [sorted(e[:-1]) for e in gold_edges]
        for w in graphs[s]:
            position = w[0]
            if position == len(graphs[s]):
                pred_edges = [(position, position - 1)]
            elif position == 0:
                pred_edges = [(position, position + 1)]
            else:
                pred_edges = [(position, position + 1), (position, position - 1)]
            pred_edges_w = [sorted((e[0]+1, e[1]+1)) for e in pred_edges if position in e]
            pred_edges_w = set([tuple(e) for e in pred_edges_w])
            gold_edges_w = [e for e in gold_edges if position + 1 in e]
            gold_edges_w = set([tuple(e) for e in gold_edges_w])
            intersect = pred_edges_w.intersection(gold_edges_w)
            if len(gold_edges_w) != 0:
                s_rec.append((position, len(intersect)/len(gold_edges_w)))
            s_prec.append((position, len(intersect)/len(pred_edges_w)))
        recall[s] = s_rec
        precision[s] = s_prec
    total_recall = sum([sum([s_r[1] for s_r in r]) for r in recall.values()])/num_words
    total_precision = sum([sum([s_p[1] for s_p in p]) for p in precision.values()])/num_words
    return total_precision, total_recall

# this function does ALL the heavy lifting, it combines graphs, parses trees, calculates UUAS 
def get_uuas(gold_standard_rels, graphs, eval=True):
    if not eval:
        final_graphs = {}
        for i in graphs.keys():
            graph_combine = nx.MultiGraph()
            for g_list in graphs[i]:
                graph_combine.add_weighted_edges_from(g_list[1].edges.data("weight"))
            graph_combine = nx.maximum_spanning_tree(graph_combine, algorithm='prim')
            edges_at_layer = list(graph_combine.edges())
            final_graphs[i] = graph_combine
        return None, final_graphs
    
    sentences = graphs.keys()
    num_rels = []
    dep_counts = {}
    adj_total = 0
    adj_prec_count = 0
    non_adj_total = 0
    non_adj_prec_count = 0
    adj_count = 0
    non_adj_count = 0
    uuas_dict = {}
    rels_dict = {}
    total = 0
    final_graphs = {}
    num_pred = []
    for i in tqdm(sentences):
        gold_rels_i = i
        num_rels.append(len(gold_standard_rels[gold_rels_i]) - len([d for d in gold_standard_rels[gold_rels_i] if d[2] == 'grand']))
        for d in gold_standard_rels[gold_rels_i]: 
            if d[2] in dep_counts.keys():
                dep_counts[d[2]] += 1
            else:
                dep_counts[d[2]] = 1
                rels_dict[d[2]] = 0
            if abs(d[0] - d[1]) == 1:
                adj_total += 1
            else:
                non_adj_total += 1
        graph_combine = nx.MultiGraph()
        for g_list in graphs[i]:
            graph_combine.add_weighted_edges_from(g_list[1].edges.data("weight"))
        graph_combine = nx.maximum_spanning_tree(graph_combine, algorithm='prim')
        edges_at_layer = list(graph_combine.edges())
        final_graphs[i] = graph_combine
        gold_edge_list = gold_standard_rels[gold_rels_i]
        num_pred.append(len(edges_at_layer))
        for rel in gold_edge_list:
            cop = (rel[0] - 1, rel[1] - 1)
            if cop in edges_at_layer or (cop[1], cop[0]) in edges_at_layer: #undirectedness
                total += 1
                if rel[2] in rels_dict.keys():
                    rels_dict[rel[2]] += 1
                else:
                    rels_dict[rel[2]] = 1
                if abs(rel[0] - rel[1]) == 1:
                    adj_count += 1
                else:
                    non_adj_count += 1
        for pred_edge in edges_at_layer:
            if abs(pred_edge[0] - pred_edge[1]) == 1:
                adj_prec_count += 1
            else:
                non_adj_prec_count += 1
    num_rels = sum(num_rels)
    num_pred = sum(num_pred)
    uuas_dict['Layer All'] = total/num_rels
    uuas_dict['Precision'] = total/num_pred
    #this calculates recall
    #uuas_dict['Adjacent_Recall'] = adj_count/adj_total
    #uuas_dict['Adjacent_Precision'] = adj_count/adj_prec_count
    uuas_dict['Non-adjacent_Recall'] = non_adj_count/non_adj_total
    uuas_dict['Non-adjacent_Precision'] = non_adj_count/non_adj_prec_count
    uuas_dict['Dependencies'] = {k : rels_dict[k]/dep_counts[k] for k in rels_dict.keys()}
    uuas_dict['If_Adj'] = adj_total/num_rels
    return uuas_dict, final_graphs

def get_parses(conll_file, ned=False):
    converted_parses = CoNLL.conll2doc(conll_file).sentences
    parses = {}
    for i, s in enumerate(converted_parses):
        sent_dict = s.to_dict()
        original_sent = ''.join(['' if type(w['id']) == tuple or w['upos'] == 'PUNCT' else w['text'] + ' ' for i, w in enumerate(sent_dict)])
        parses[(i, original_sent)] = sent_dict

    target_sents_deps_labeled = {}
    for sent in parses.keys():
        deplist = [(word['id'], word['head'], word['deprel']) for word in parses[sent]]
        deplist = [dep for dep in deplist if dep[2] != 'root']
        updated_positions = [i for i in range(1, len(deplist) + 2)]
        for i, dep in enumerate(deplist):
            if dep[2] == 'punct' and i != len(deplist) - 1:
                for j, pos_id in enumerate(updated_positions[i+1:]):
                    updated_positions[j + i + 1] -= 1
        fixed_deplist = []
        for dep in deplist:
            if dep[2] != 'punct':
                pos_1 = updated_positions[dep[0] - 1]
                pos_2 = updated_positions[dep[1] - 1]
                fixed_deplist.append([pos_1, pos_2, dep[2]])
        target_sents_deps_labeled[sent] = fixed_deplist
    # adds the grandparents if we want it
    if ned:
        for k in target_sents_deps_labeled.keys():
            deps = target_sents_deps_labeled[k]
            grandparents = []
            children = [d[0] for d in deps]
            heads = [d[1] for d in deps]
            for d in deps:
                head = d[1]
                if head in children:
                    grandparents.append((d[0], heads[children.index(head)], 'grand'))
            target_sents_deps_labeled[k] = [*deps, *grandparents]
    return target_sents_deps_labeled

def dict_to_dataframe(uuas_dict, trial_name):
    l = []
    for k in uuas_dict.keys():
        if type(uuas_dict[k]) != dict:
            d = {'Deprel' : k, trial_name : uuas_dict[k]}
            l.append(d)
        elif type(uuas_dict[k]) == dict:
            l_d = dict_to_dataframe(uuas_dict[k], trial_name)
            l = l + l_d
    return l

def main():
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    model_version = 'bert-base-uncased'
    model = BertModel.from_pretrained(model_version, output_attentions=True)
    model.eval()
    tokenizer = BertTokenizerFast.from_pretrained(model_version)
    model = model.to(device)
    split = str(sys.argv[3])
    sent_f = str(sys.argv[1])
    n = int(sys.argv[4])
    #conll formatted depparses
    parses = str(sys.argv[5])

    layers_to_report = range(9, 10)
    
    with open(sent_f, 'rb') as f:
        sentences = pickle.load(f)

    combined_dicts = []
    
    loaded_parses = get_parses(parses, ned=False)

    for i in layers_to_report:
        only_target_atts, perturbed_atts = get_all_atts(sentences, model, tokenizer, l=i)
        only_target_graphs = get_graphs(only_target_atts, trees=False)
        perturbed_graphs = get_graphs(perturbed_atts, trees=False)

        perturbed_total_uuas, fixed_graphs = get_uuas(loaded_parses, perturbed_graphs)
        only_target_total_uuas, target_graphs = get_uuas(loaded_parses, only_target_graphs)
        
        trial = 'Layer' + str(i + 1)
       
        perturbed_total_uuas = dict_to_dataframe(perturbed_total_uuas, trial+"_pert")
        only_target_total_uuas = dict_to_dataframe(only_target_total_uuas, trial+"_target")
        
        combined_dicts.append(pd.DataFrame.from_dict(perturbed_total_uuas))
        combined_dicts.append(pd.DataFrame.from_dict(only_target_total_uuas))
    
    dataframe = combined_dicts[0]
    for d in combined_dicts[1:]:
        dataframe = dataframe.merge(d, on='Deprel')
    
    dataframe.to_csv('./out/' + split +'/uuas_results' + str(n) + '.csv')
    return    

if __name__=="__main__":
    main()