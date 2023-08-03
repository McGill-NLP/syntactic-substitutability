import sys
import stanza
from transformers import BertTokenizerFast, AutoModelForMaskedLM, BertModel
import torch
import generate_substitutions
import parse_eval

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
    
    print('Done loading!')
    sent_to_parse = str(sys.argv[1])
    #out = str(sys.argv[2])
    #pathlib.Path(out).mkdir(parents=True, exist_ok=True)
    num_sent = int(sys.argv[2])

    sent_dict = convert_to_dict(sent_to_parse)

    #this can now be passed into the normal parser
    subs_dict = generate_substitutions.fill_sentences(sent_dict, model, number_sentences=num_sent, perturbed_categories = ['ADJ', 'ADV', 'NOUN', 'VERB', 'PROPN', 'ADP', 'DET'], use_bert=True, tokenizer=tokenizer, nlp=pos_tagger, need_pos=False, have_pos=False)    
    
    print(subs_dict)

    model = BertModel.from_pretrained(model_version, output_attentions=True)
    model.eval()
    model = model.to(device)

    layer = 9 #offset by 1

    only_target_atts, perturbed_atts = parse_eval.get_all_atts(subs_dict, model, tokenizer, l=layer)
    only_target_graphs = parse_eval.get_graphs(only_target_atts, trees=False)
    perturbed_graphs = parse_eval.get_graphs(perturbed_atts, trees=False)

    _, fixed_graphs = parse_eval.get_uuas(None, perturbed_graphs, eval=False)
    _, target_graphs = parse_eval.get_uuas(None, only_target_graphs, eval=False)

    return

if __name__ == '__main__':
    main()