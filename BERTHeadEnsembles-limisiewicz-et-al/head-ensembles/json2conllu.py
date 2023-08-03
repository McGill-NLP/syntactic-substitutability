#!/usr/bin/env python3
import argparse
import json

from dependency import Dependency

if __name__ == '__main__':
	ap = argparse.ArgumentParser()
	ap.add_argument("json4bert", type=str, help="Json file with Bert input ")
	ap.add_argument("conll", type=str, help="Conllu to be converted to Bert input.")
	args = ap.parse_args()
	
	with open(args.json4bert, 'r') as f:
		bert_json = json.load(f)
	
	with open(args.conll, 'w') as f:
		for d in bert_json:
			sentence = d['text'].split()
			for i, w in enumerate(sentence):
				f.writelines(str(i+1) + '\t' + w + '\t' + w + '\t' + 'DET' + '\t' + 'DT' + '\t' + '_' + '\t' + '1' + '\t' + 'det' + '\t' + '_' + '\t' + '_' + '\n')
			f.writelines('\n')
		f.close()
