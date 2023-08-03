from dependency import Dependency
from metrics import UAS

language='Korean'
langcode='ko'
PUD_CONLL = f"/net/data/universal-dependencies-2.6/UD_{language}-PUD/{langcode}_pud-ud-test.conllu"
PUD_TOKENS = f"/net/data/universal-dependencies-2.6/UD_{language}-PUD/{langcode}_pud-ud-test.txt"

dependency_trees = Dependency(PUD_CONLL, PUD_TOKENS)

uas_metric = UAS(dependency_trees)

left_branching = [list(zip(range(1,len(toks)),range(0, len(toks)-1))) + [(0,-1)] for toks in dependency_trees.tokens]
right_branching = [list(zip(range(0,len(toks)-1),range(1,len(toks)))) + [(len(toks)-1, -1)] for toks in dependency_trees.tokens]

uas_metric.reset_state()
uas_metric(range(1000), left_branching)
print("UAS left branching: ", uas_metric.result()*100)
uas_metric.reset_state()
uas_metric(range(1000), right_branching)
print("UAS right branching: ", uas_metric.result()*100)

gr_right_branching = right_branching
gr_left_branching = left_branching

for idx, root in enumerate(dependency_trees.roots):
	last_tok_id = len(dependency_trees.tokens[idx]) -1
	if root != last_tok_id :
		gr_right_branching[idx].remove((root,root+1))
		gr_right_branching[idx].append((root, -1))
		gr_right_branching[idx].remove((last_tok_id , -1))
		gr_right_branching[idx].append((last_tok_id , 0))
		
	if root != 0:
		gr_left_branching[idx].remove((root, root-1))
		gr_left_branching[idx].append((root, -1))
		gr_left_branching[idx].remove((0, -1))
		gr_left_branching[idx].append((0, last_tok_id))

uas_metric.reset_state()
uas_metric(range(1000), gr_left_branching)
print("UAS left branching (with gold root): ", uas_metric.result()*100)
uas_metric.reset_state()
uas_metric(range(1000), gr_right_branching)
print("UAS right branching (with gold root): ", uas_metric.result()*100)
