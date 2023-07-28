import sys
import stanza
from stanza.utils.conll import CoNLL
import pickle
import copy
import pathlib
from transformers import BertTokenizerFast, FillMaskPipeline, AutoModelForMaskedLM
import torch
from tqdm import tqdm
import generate_substitutions

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
    print(sent_dict)

    #this can now be passed into the normal parser
    subs_dict = generate_substitutions.fill_sentences(sent_dict, model, number_sentences=num_sent, perturbed_categories = ['ADJ', 'ADV', 'NOUN', 'VERB', 'PROPN', 'ADP', 'DET'], use_bert=True, tokenizer=tokenizer, nlp=pos_tagger, need_pos=False, have_pos=False)    
    """for pos in subs_dict[list(subs_dict.keys())[0]]:
        for s in pos[1]:
            print(s)"""

    return

if __name__ == '__main__':
    main()