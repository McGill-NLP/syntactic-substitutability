import sys
import stanza
from stanza.utils.conll import CoNLL
import pickle
import copy
import pathlib
from transformers import BertTokenizerFast, FillMaskPipeline, AutoModelForMaskedLM
import torch
import random
from tqdm import tqdm

def parse_conll(filename):
    doc = CoNLL.conll2doc(filename)
    return doc

def fill_sentences(sentences, word_model, number_sentences=10, perturbed_categories=['ADJ', 'ADV', 'NOUN', 'VERB'], use_bert=False, tokenizer=None, nlp=None, need_pos=True):
    filled = {}
    already_calculated = {}
    for k, s in enumerate(tqdm(sentences)):
        sent_dict = s.to_dict()
        #sent_dict = s
        original_sentence = ''.join(['' if type(w['id']) == tuple or w['upos'] == 'PUNCT' else w['text'] + ' ' for i, w in enumerate(sent_dict)])
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
                    #results = get_predictions_bert_token(dict_copy, i, number_words=15, transformer_model=w2v_model, tokenizer=tokenizer)
                else:
                    word = dict_copy[i]['text']
                    if word not in already_calculated.keys():
                        results = get_replacements(word, word_model, n=number_sentences)
                        already_calculated[word] = results
                    else:
                        results = already_calculated[word]
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
                #token_counter += 1
            elif type(sent_dict[i]['id']) != tuple and sent_dict[i]['upos'] != 'PUNCT':
                pert.append((position, [original_sentence]))
                position += 1
                #token_counter += 1
        filled[(k, original_sentence)] = pert
    return filled

def fill_full_sentences(sentences, w2v_model, number_sentences=10, perturbed_categories=['ADJ', 'ADV', 'NOUN', 'VERB'], use_bert=False, tokenizer=None, need_pos=True):
    filled = {}
    already_calculated = {}
    for k, s in enumerate(sentences):
        if k % 50 == 0:
            print(k)
        sent_dict = s.to_dict()
        original_sentence = ''.join(['' if type(w['id']) == tuple or w['upos'] == 'PUNCT' else w['text'] + ' ' for i, w in enumerate(sent_dict)])
        pert = []
        position = 0
        full_pos_list = []
        for i in range(len(sent_dict)):
            token_counter = 0
            if 'upos' in sent_dict[i].keys() and sent_dict[i]['upos'] in perturbed_categories and replace(sent_dict[i]):
                dict_copy = copy.deepcopy(sent_dict)
                #adds original sentence to the beginning of the list
                replacements = [''.join(['' if type(w['id']) == tuple or w['upos'] == 'PUNCT' else w['text'] + ' ' for w in dict_copy]).strip()]
                if use_bert:
                    results = get_replacements_mask(dict_copy, i, number_words=number_sentences, transformer_model=w2v_model, tokenizer=tokenizer)
                    #results = get_predictions_bert_token(dict_copy, i, number_words=20, transformer_model=w2v_model, tokenizer=tokenizer)               
                position += 1
                if len(results) == 0:
                    results = dict_copy[i]['text']
                full_pos_list.append((i, results))
                #token_counter += 1
            elif type(sent_dict[i]['id']) != tuple and sent_dict[i]['upos'] != 'PUNCT':
                position += 1
                #token_counter += 1
        filled[original_sentence] = generate_new_sentences(full_pos_list)
    return filled

def generate_new_sentences(pos_word_list):
    final = []
    for i in range(len(pos_word_list)):
        new_s = ''.join([str(random.choice(x[1])) + " " for x in pos_word_list])
        print(new_s)
        #print(pos_word_list)
        final.append((i, [new_s]))
    return final

def get_replacements(word, w2v_model, n=10):
    try:
        results = w2v_model.most_similar(word, topn=n+5)
        results = [r[0] for r in results]
    except KeyError:
        results = [word]
    return results

def get_replacements_mask(sent_dict, position, transformer_model=None, tokenizer=None, number_words=10):
    unmasker = FillMaskPipeline(model=transformer_model, tokenizer=tokenizer, device=0)
    unmasker.model = transformer_model
    unmasker.tokenizer = tokenizer
    masked_sentence = ''.join(['' if type(w['id']) == tuple else '[MASK] ' if i == position else w['text'] + ' ' for i, w in enumerate(sent_dict)])
    filled_list = unmasker([masked_sentence], top_k=number_words)
    filled_list = [word['token_str'].replace(" ", "") for word in filled_list if word['token_str'].replace(" ", "").isalpha()]
    #filled_list = ['[MASK]']
    return filled_list

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
    #print(spans)
    return spans
    
def get_predictions_bert_token(sent_dict, position, transformer_model=None, tokenizer=None, number_words=10):
    # Tokenize input
    text = "[CLS] %s [SEP]"%''.join(['' if type(w['id']) == tuple else w['text'] + ' ' for i, w in enumerate(sent_dict)])
    tokenized_text = tokenizer.tokenize(text)
    spans = get_spans(text, tokenizer)
    position_index = spans[position+1]
    if type(position_index) == tuple:
        position_index = position_index[0]
    indexed_tokens = tokenizer.convert_tokens_to_ids(tokenized_text)
    tokens_tensor = torch.tensor([indexed_tokens])
    tokens_tensor = tokens_tensor.to('cuda:0')    # if you have gpu

    # Predict all tokens
    with torch.no_grad():
        outputs = transformer_model(tokens_tensor)
        predictions = outputs[0]

    probs = torch.nn.functional.softmax(predictions[0, position_index], dim=-1)
    top_k_weights, top_k_indices = torch.topk(probs, number_words, sorted=True)
    filled_tokens = []
    for i, pred_idx in enumerate(top_k_indices):
        predicted_token = tokenizer.convert_ids_to_tokens([pred_idx])[0]
        if predicted_token.isalpha():
            filled_tokens.append(predicted_token)
    return filled_tokens

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
    #print(upos)
    #print(original_upos)
    return upos == original_upos

def replace(word):
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
    n = int(sys.argv[4])
    f = str(sys.argv[1])
    d = parse_conll(f)
    d = d.sentences
    """with open(f, 'rb') as file:
        d = pickle.load(file)"""
    d1 = fill_sentences(d, model, number_sentences=n, perturbed_categories = ['ADJ', 'ADV', 'NOUN', 'VERB', 'PROPN', 'ADP', 'DET'], use_bert=True, tokenizer=tokenizer, nlp=pos_tagger, need_pos=True)    
    out = str(sys.argv[2])
    pathlib.Path(out).mkdir(parents=True, exist_ok=True)
    out_pkl = out + '/'+ str(sys.argv[3]) + ".pos_" + str(n) + ".pkl"
    with open(out_pkl, 'wb') as f:
        pickle.dump(d1, f)
    out_txt = out + '/'+ str(sys.argv[3]) + ".pos_" + str(n) + ".txt"
    s_list = []
    for s in d1.values():
        for position in s:
            s_list += position[1]
    with open(out_txt, 'w') as f:
        for l in s_list:
            #print(l)
            f.write(l)
            f.write('\n')
    return

if __name__ == '__main__':
    main()