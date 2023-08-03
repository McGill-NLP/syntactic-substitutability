from collections import defaultdict, OrderedDict

from unidecode import unidecode


class Dependency():

    pos_labels = ('ADJ', 'ADP', 'ADV', 'AUX', 'CCONJ', 'DET', 'INTJ', 'NOUN', 'NUM',
                  'PART', 'PRON', 'PROPN', 'PUNCT', 'SCONJ', 'SYM', 'VERB', 'X')

    label_map = OrderedDict({'acl': 'adj-clause',
                             'advcl': 'adv-clause',
                             'advmod': 'adv-modifier',
                             'amod': 'adj-modifier',
                             'appos': 'apposition',
                             'aux': 'auxiliary',
                             'xcomp': 'clausal',
                             'ccomp': 'clausal',
                             'parataxis': 'parataxis',
                             'compound': 'compound',
                             'conj': 'conjunct',
                             'cc': 'cc',
                             'csubj': 'clausal-subject',
                             'det': 'determiner',
                             'nmod': 'noun-modifier',
                             'nsubj': 'subject',
                             'nummod': 'num-modifier',
                             'iobj': 'object',
                             'obj': 'object',
                             'punct': 'punctuation',
                             'case': 'case',
                             'mark': 'mark',
                             'root': 'root'})

    reverse_label_map = {v: k for k, v in label_map.items()}

    LABEL_OTHER = 'other'
    LABEL_ALL = 'all'

    CONLLU_ID = 0
    CONLLU_ORTH = 1
    CONLLU_POS = 3
    CONLLU_HEAD = 6
    CONLLU_LABEL = 7

    def __init__(self, conll_file, bert_wordpiece_file=None):

        self.tokens = []
        self.relations = []
        self.labeled_relations = []
        self.roots = []
        self.wordpieces2tokens = []

        self.read_conllu(conll_file)
        if bert_wordpiece_file:
            self.group_wordpieces(bert_wordpiece_file)

    @classmethod
    def transform_label(cls, label):
        """Maps UD labels to labels specified in label_map. Returns 'other' for unknown labels"""
        label = label.split(':')[0]  # to cope with nsubj:pass for instance
        if label in cls.label_map:
            label = cls.label_map[label]
        else:
            label = cls.LABEL_OTHER
        return label
    
    @property
    def unlabeled_relations(self):
        """Returns lists of tuples of dependent and its head or each sentence."""
        return [[rel for rel in sent_relations[self.LABEL_ALL]] for sent_relations in self.relations]

    def read_conllu(self, conll_file_path):
        sentence_relations = defaultdict(list)
        sentence_labeled_relations = []
        sentence_tokens = []

        with open(conll_file_path, 'r') as in_conllu:
            sentid = 0
            for line in in_conllu:
                if line == '\n':
                    self.relations.append(sentence_relations)
                    sentence_relations = defaultdict(list)
                    self.labeled_relations.append(sentence_labeled_relations)
                    sentence_labeled_relations = []
                    self.tokens.append(sentence_tokens)
                    sentence_tokens = []
                    sentid += 1
                elif line.startswith('#'):
                    continue
                else:
                    fields = line.strip().split('\t')
                    if fields[self.CONLLU_ID].isdigit():
                        head_id = int(fields[self.CONLLU_HEAD]) - 1
                        dep_id = int(fields[self.CONLLU_ID]) - 1
                        original_label = fields[self.CONLLU_LABEL]
                        label = self.transform_label(original_label)

                        sentence_relations[label].append((dep_id, head_id))
                        sentence_relations[self.LABEL_ALL].append((dep_id, head_id))

                        sentence_labeled_relations.append((dep_id, head_id, original_label))
                        if head_id < 0:
                            self.roots.append(int(fields[self.CONLLU_ID]) -1)

                        sentence_tokens.append(fields[self.CONLLU_ORTH])

    def group_wordpieces(self, wordpieces_file):
        '''
        Joins wordpices of tokens, so that they correspond to the tokens in conllu file.

        :param wordpieces_all: lists of BPE pieces for each sentence
        :return: group_ids_all list of grouped token ids, e.g. for a BPE sentence:
        "Mr. Kowal@@ ski called" joined to "Mr. Kowalski called" it would be [[0], [1, 2], [3]]
        '''

        with open(wordpieces_file, 'r') as in_file:
            wordpieces = [wp_sentence.strip().split() for wp_sentence in in_file.readlines()]

        grouped_ids_all = []
        tokens_out_all = []
        idx = 0
        for wordpieces, conllu_tokens in zip(wordpieces, self.tokens):
            conllu_id = 0
            curr_token = ''
            grouped_ids = []
            tokens_out = []
            wp_ids = []
            for wp_id, wp in enumerate(wordpieces):
                wp_ids.append(wp_id)
                if wp.endswith('@@'):
                    curr_token += wp[:-2]
                else:
                    curr_token += wp
                if unidecode(curr_token).lower() == unidecode(conllu_tokens[conllu_id]).lower():
                    grouped_ids.append(wp_ids)
                    wp_ids = []
                    tokens_out.append(curr_token)
                    curr_token = ''
                    conllu_id += 1
            try:
                assert conllu_id == len(conllu_tokens), f'{idx} \n' \
                                                        f'bert count {conllu_id} tokens{tokens_out} \n' \
                                                        f'conllu count {len(conllu_tokens)}, tokens {conllu_tokens}'
            except AssertionError:
                self.wordpieces2tokens.append(None)
            else:
                self.wordpieces2tokens.append(grouped_ids)
            idx += 1

    def calc_offset_modes(self):
        offsets = defaultdict(list)
        for sent_relations in self.relations:
            for rel, dep_edges in sent_relations.items():
                for dep, head in dep_edges:
                    if head != -1:
                        offsets[rel].append(head - dep)

        offset_modes = {rel: max(set(off), key=off.count) for rel, off in offsets.items()}
        return offset_modes

    def eval_positional_baseline(self, offset_modes=None):
        offset_modes = offset_modes or self.calc_offset_modes()

        results = defaultdict(list)

        for sent_relations in self.relations:
            for rel, dep_edges in sent_relations.items():
                for dep, head in dep_edges:
                    if head != -1:
                        results[rel].append(head - dep == offset_modes.get(rel, 0))

        return {rel: sum(res)/len(res) for rel, res in results.items()}
