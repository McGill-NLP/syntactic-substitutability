"""
Takes in a json file of [dict]
dict contains dict['text'] = sentence
returns for each dict, dict['text'] and dict['substitutions'] = [(0,['substituted']), (1, ['substituted']), ...., (n-1, ['substituted'])]
"""
import os

os.environ["USE_TF"] = "0"
os.environ["USE_TORCH"] = "1"

import numpy as np
import torch

import utils
from transformers import BertTokenizerFast, FillMaskPipeline, AutoModelForMaskedLM
from tqdm import tqdm

def generate(json_file, number_sentences=10, filename=None):
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    model_version = 'bert-base-uncased'
    model = AutoModelForMaskedLM.from_pretrained(model_version)
    tokenizer = BertTokenizerFast.from_pretrained(model_version)
    word2vec = model.to(device)
    final = []
    final_listed = []
    print("GENERATING SUBSTITUTIONS")
    for features in tqdm(utils.load_json(json_file)):
        sentence_substitutions = []
        substitutions_listed = []
        sentence = features['text']
        sentence_split = sentence.split()
        sentence_length = len(sentence.split())
        for i in range(sentence_length):
            word = sentence_split[i]
            if word == "the":
                sentence_substitutions.append((i, []))
            elif word.isalpha():
                results = get_replacements_mask(sentence_split, i, transformer_model=word2vec, tokenizer=tokenizer, number_words=number_sentences)
                substitutions_at_i = []
                for replacement in results:
                    if replacement.lower() == word.lower():
                        continue
                    new_sentence = ''.join([replacement + " " if j == i else w + " " for j, w in enumerate(sentence_split)])
                    substitutions_at_i.append(new_sentence.strip())
                    substitutions_listed.append(new_sentence.strip())
                sentence_substitutions.append((i, substitutions_at_i))
            else:
                sentence_substitutions.append((i, []))
        final.append({'text' : sentence, 'substitutions' : sentence_substitutions})
        final_listed += [{'text' : sentence}] + [{'text' : s_sub} for s_sub in substitutions_listed]
    print("This is the number of sentences: " + str(len(final_listed)))   
    return final, final_listed

def get_replacements_mask(sent_dict, position, transformer_model=None, tokenizer=None, number_words=10):
    unmasker = FillMaskPipeline(model=transformer_model, tokenizer=tokenizer, device=0)
    unmasker.model = transformer_model
    unmasker.tokenizer = tokenizer
    masked_sentence = ''.join(['[MASK] ' if i == position else w + ' ' for i, w in enumerate(sent_dict)])
    filled_list = unmasker([masked_sentence], top_k=number_words)
    filled_list = [word['token_str'].replace(" ", "") for word in filled_list if word['token_str'].replace(" ", "").isalpha()]
    #filled_list = ['[MASK]']
    return filled_list
