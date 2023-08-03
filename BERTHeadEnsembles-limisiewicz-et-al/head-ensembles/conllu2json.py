#!/usr/bin/env python3
import argparse
import json

from dependency import Dependency

if __name__ == '__main__':
	ap = argparse.ArgumentParser()
	ap.add_argument("conll", type=str, help="Conllu to be converted to Bert input.")
	ap.add_argument("json4bert", type=str, help="Json file with Bert input ")
	args = ap.parse_args()
	
	read_dependency = Dependency(args.conll)
	
	with open(args.json4bert, 'w') as outj:
		json.dump([{'text': ' '.join(sent_toks)} for sent_toks in read_dependency.tokens], outj)
