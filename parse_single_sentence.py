import sys
import stanza
from transformers import BertTokenizerFast, AutoModelForMaskedLM, BertModel
import torch
import generate_substitutions
import parse_eval
import networkx as nx
import random

def convert_to_dict(sent):
    tokenizer = stanza.Pipeline(lang='en', processors='tokenize')
    doc = list(tokenizer(sent).sentences)
    return doc

def generate_json(sent_list, edge_list):
    sent_dict = {
        "words" : [], 
        "arcs" : []
        }
    for w in sent_list:
        sent_dict["words"].append({"text" : w})
    for e in edge_list:
        sent_dict["arcs"].append({"start" : e[0], "end" : e[1]})
    return sent_dict

def main():
    print('Loading model: ')
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    model_version = 'bert-base-uncased'
    model = AutoModelForMaskedLM.from_pretrained(model_version)
    model = model.to(device)
    tokenizer = BertTokenizerFast.from_pretrained(model_version)
    pos_tagger = stanza.Pipeline(lang='en', processors='tokenize,mwt,pos')
    model_2 = BertModel.from_pretrained(model_version, output_attentions=True)
    model_2.eval()
    model_2 = model_2.to(device)
    print('Done loading!')
    if len(sys.argv) > 1:
        num_sent = int(sys.argv[1])
    else:
        num_sent = 5
    parser_active = True
    while parser_active:
        sent_to_parse = input('Please enter a sentence to parse: ')
        if sent_to_parse == "cs":
            num_sent = int(input('Change the number of substitutions: '))
            continue
        sent_dict = convert_to_dict(sent_to_parse)

        #this can now be passed into the normal parser
        subs_dict = generate_substitutions.fill_sentences(sent_dict, model, number_sentences=num_sent, perturbed_categories = ['ADJ', 'ADV', 'NOUN', 'VERB', 'PROPN', 'ADP', 'DET'], use_bert=True, tokenizer=tokenizer, nlp=pos_tagger, need_pos=False, have_pos=False)    
        print('Here are some of the substitutions generated: ')
        subs_at_pos = list(subs_dict.values())[0]
        for p in subs_at_pos:
            sent_list=p[1]
            print(random.choice(sent_list[1:]))
        print("-----------------------------")

        layer = 9 #default is layer 10 as used in the paper

        only_target_atts, perturbed_atts = parse_eval.get_all_atts(subs_dict, model_2, tokenizer, l=layer)
        only_target_graphs = parse_eval.get_graphs(only_target_atts, trees=False)
        perturbed_graphs = parse_eval.get_graphs(perturbed_atts, trees=False)

        _, ssud_graphs = parse_eval.get_uuas(None, perturbed_graphs, eval=False)
        _, target_graphs = parse_eval.get_uuas(None, only_target_graphs, eval=False)
        sentence_index = list(ssud_graphs.keys())[0]
        predicted_edges_ssud = list(nx.dfs_tree(ssud_graphs[sentence_index]).edges())
        print("This is the induced SSUD parse (k = " + str(num_sent) + "):")
        print(predicted_edges_ssud)
        s_list = sentence_index[1].split()
        for e in predicted_edges_ssud:
           print(s_list[e[0]] + ' <--> ' + s_list[e[1]])
        print("-----------------------------")
        print("This is the induced target only parse (k = 0):")
        predicted_edges_target = list(nx.dfs_tree(target_graphs[sentence_index]).edges())
        print(predicted_edges_target)
        s_list = sentence_index[1].split()
        for e in predicted_edges_target:
           print(s_list[e[0]] + ' <--> ' + s_list[e[1]])

        json_formatted = (generate_json(s_list, predicted_edges_ssud), generate_json(s_list, predicted_edges_target))
    return json_formatted

if __name__ == '__main__':
    main()