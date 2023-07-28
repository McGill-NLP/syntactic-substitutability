import sys
import stanza
from stanza.utils.conll import CoNLL
import pickle
import copy
import pathlib
from transformers import BertTokenizerFast, FillMaskPipeline, AutoModelForMaskedLM
import torch
from tqdm import tqdm

def parse_conll(filename):
    doc = CoNLL.conll2doc(filename)
    return doc

def fill_sentences(sentences, word_model, number_sentences=10, perturbed_categories=['ADJ', 'ADV', 'NOUN', 'VERB'], use_bert=True, tokenizer=None, nlp=None, need_pos=True, have_pos=True):
    filled = {}
    for k, s in enumerate(tqdm(sentences)):
        sent_dict = s.to_dict()
        #sent_dict = s
        if have_pos:
            original_sentence = ''.join(['' if type(w['id']) == tuple or w['upos'] == 'PUNCT' else w['text'] + ' ' for i, w in enumerate(sent_dict)])
        else:
            original_sentence = ''.join(['' if type(w['id']) == tuple else w['text'] + ' ' for i, w in enumerate(sent_dict)])
        pert = []
        position = 0
        for i in range(len(sent_dict)):
            token_counter = 0
            if 'upos' in sent_dict[i].keys() and sent_dict[i]['upos'] in perturbed_categories and replace(sent_dict[i]):
                dict_copy = copy.deepcopy(sent_dict)
                #adds original sentence to the beginning of the list
                replacements = [''.join(['' if type(w['id']) == tuple or w['upos'] == 'PUNCT' else w['text'] + ' ' for w in dict_copy]).strip()]
                if use_bert:
                    results = get_replacements_mask(dict_copy, i, number_words=number_sentences+10, transformer_model=word_model, tokenizer=tokenizer)
                if len(results) == 0:
                    #we've already added the original sentence to the beginning of the list
                    continue
                else:
                    j = 0
                    count = 0
                    #results = random.sample(results, len(results))
                    while j < len(results) and count < number_sentences:
                        if sent_dict[i]['text'].lower() != "":
                            dict_copy[i]['text'] = results[j]
                            new_sent = ''.join(['' if type(w['id']) == tuple else w['text'] + ' ' for w in dict_copy])
                            new_sent = new_sent.strip()
                            if need_pos: 
                                if check_pos(new_sent, sent_dict, position, nlp):
                                    new_sent = ''.join(['' if type(w['id']) == tuple or w['upos'] == 'PUNCT' else w['text'] + ' ' for w in dict_copy])
                                    replacements.append(new_sent)
                                    count += 1
                        j += 1
                pert.append((position, replacements))               
                position += 1
            elif type(sent_dict[i]['id']) != tuple and 'upos' in sent_dict[i].keys() and sent_dict[i]['upos'] != 'PUNCT':
                pert.append((position, [original_sentence]))
                position += 1
            elif not have_pos:
                dict_copy = copy.deepcopy(sent_dict)
                #adds original sentence to the beginning of the list
                replacements = [''.join(['' if type(w['id']) == tuple else w['text'] + ' ' for w in dict_copy]).strip()]
                if use_bert:
                    results = get_replacements_mask(dict_copy, i, number_words=number_sentences+10, transformer_model=word_model, tokenizer=tokenizer)
                if len(results) == 0:
                    #we've already added the original sentence to the beginning of the list
                    continue
                else:
                    j = 0
                    count = 0
                    #results = random.sample(results, len(results))
                    while j < len(results) and count < number_sentences:
                        if sent_dict[i]['text'].lower() != "":
                            dict_copy[i]['text'] = results[j]
                            new_sent = ''.join(['' if type(w['id']) == tuple else w['text'] + ' ' for w in dict_copy])
                            new_sent = new_sent.strip()
                            replacements.append(new_sent)
                            count += 1
                        j += 1
                pert.append((position, replacements))               
                position += 1
        filled[(k, original_sentence)] = pert
    return filled

def get_replacements_mask(sent_dict, position, transformer_model=None, tokenizer=None, number_words=10):
    unmasker = FillMaskPipeline(model=transformer_model, tokenizer=tokenizer, device=0)
    unmasker.model = transformer_model
    unmasker.tokenizer = tokenizer
    masked_sentence = ''.join(['' if type(w['id']) == tuple else '[MASK] ' if i == position else w['text'] + ' ' for i, w in enumerate(sent_dict)])
    filled_list = unmasker([masked_sentence], top_k=number_words)
    filled_list = [word['token_str'].replace(" ", "") for word in filled_list if word['token_str'].replace(" ", "").isalpha()]
    return filled_list
    
def check_pos(filled_sent, sent_dict, position, pos_tagger):
    original_upos = sent_dict[position]['upos']
    tagged_sent = pos_tagger(filled_sent)
    try:
        if len(tagged_sent.sentences) == 1:
            upos = tagged_sent.sentences[0].words[position].upos
        else:
            combined_list = tagged_sent.sentences[0].words
            for s in tagged_sent.sentences[1:]:
                combined_list += s.words
            upos = combined_list[position].upos
    except IndexError: 
        print(filled_sent)
        print(position)
        upos = ''
    return upos == original_upos

def replace(word):
    # excepts common functional words from replacement
    aux_cop = ['have', 'has', 'had', "'d", 'having', 'being', 'be', 'is', 'am', 'are', 'was', "'s", "'m", "'re", 'will', "'ll"]
    modal = ['must', 'need', 'needs', 'should', 'would', 'want', 'wants', 'can', 'might']
    return word['text'] not in aux_cop + modal

def main():
    print('Loading model: ')
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    model_version = 'bert-base-uncased'
    model = AutoModelForMaskedLM.from_pretrained(model_version)
    model = model.to(device)
    tokenizer = BertTokenizerFast.from_pretrained(model_version)
    pos_tagger = stanza.Pipeline(lang='en', processors='tokenize,mwt,pos')
    print('Done loading!')
    num_sent = int(sys.argv[4])
    conll_file = str(sys.argv[1])
    sent_dict = parse_conll(conll_file)
    sent_dict = sent_dict.sentences

    subs_dict = fill_sentences(sent_dict, model, number_sentences=num_sent, perturbed_categories = ['ADJ', 'ADV', 'NOUN', 'VERB', 'PROPN', 'ADP', 'DET'], use_bert=True, tokenizer=tokenizer, nlp=pos_tagger, need_pos=True)    
    out = str(sys.argv[2])
    pathlib.Path(out).mkdir(parents=True, exist_ok=True)
    out_pkl = out + '/'+ str(sys.argv[3]) + ".pos_" + str(num_sent) + ".pkl"
    with open(out_pkl, 'wb') as f:
        pickle.dump(subs_dict, f)
    out_txt = out + '/'+ str(sys.argv[3]) + ".pos_" + str(num_sent) + ".txt"

    with open(out_txt, 'w') as f:
        for s in subs_dict.values():
            for position in s:
                for sub in position[1]:
                    f.write(sub + '\n')
    
    return

if __name__ == '__main__':
    main()