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
    num_sent = int(sys.argv[1])
    parser_active = True
    while parser_active:
        sent_to_parse = input('Please enter a sentence to parse: ')
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
        predicted_edges = list(nx.dfs_tree(ssud_graphs[sentence_index]).edges())
        print("This is the induced SSUD parse:")
        print(predicted_edges)
        s_list = sentence_index[1].split()
        for e in predicted_edges:
           print(s_list[e[0]] + ' <--> ' + s_list[e[1]])
        print("-----------------------------")
        print("This is the induced target only parse:")
        predicted_edges = list(nx.dfs_tree(target_graphs[sentence_index]).edges())
        print(predicted_edges)
        s_list = sentence_index[1].split()
        for e in predicted_edges:
           print(s_list[e[0]] + ' <--> ' + s_list[e[1]])  
    return

if __name__ == '__main__':
    main()