import numpy as np
import sys
from itertools import chain
from tqdm import tqdm


class AttentionWrapper:
    # Those values are used in all the experiments. Parameters could be superfluous.
    MAX_LEN = 1000  # maximum number of tokens in the sentence
    WITH_EOS = False  # whether attention matrix contain EOS token.
    WITH_CLS = False # whether attention matrix contain CLS token.
    NO_SOFTMAX = False  # whether to conduct softmax on loaded attention matrices.

    def __init__(self, attention_file, tokens_grouped, selected_sentences=None, whole_words=False):

        # loads all the attention matrices and tokens
        attention_loaded = np.load(attention_file)
        self.layer_count = attention_loaded['arr_0'].shape[0]
        self.head_count = attention_loaded['arr_0'].shape[1]

        self.matrices = list()
        self.tokens_grouped = tokens_grouped
        self.sentence_idcs = selected_sentences or list(range(len(attention_loaded.files)))
        self.whole_words = whole_words

        self.preprocess_matrices(attention_loaded)

    def calc_metric_grid(self, metric):
        metric_res = np.zeros((self.layer_count, self.head_count))
        for l in range(self.layer_count):
            for h in range(self.head_count):
                metric.reset_state()
                metric(self.sentence_idcs, [np.squeeze(sent_matrices[l,h,:,:]) for sent_matrices in self.matrices])
                metric_res[l,h] = metric.result()
        return metric_res

    def calc_metric_ensemble(self, metric, layer_idx, head_idx):
        metric.reset_state()
        metric(self.sentence_idcs,
               [sent_matrices[layer_idx, head_idx, :,:].mean(axis=0) for sent_matrices in self.matrices])
        return metric.result()

    def __getitem__(self, idx):
        return self.sentence_idcs[idx], self.matrices[idx]
    
    def __iter__(self):
        for sent_id, mats in zip(self.sentence_idcs, self.matrices):
            yield sent_id, mats

    def check_wordpieces(self, item, attention_loaded):
        matrix_id = 'arr_' + str(item)
        attention_rank = attention_loaded[matrix_id].shape[2] - int(self.WITH_EOS) - int(self.WITH_CLS)
        item_wordpieces_grouped = self.tokens_grouped[item]
        if item_wordpieces_grouped is None:
            print('Token mismatch sentence skipped', item, file=sys.stderr)
            return False

        item_wordpieces = list(chain.from_iterable(item_wordpieces_grouped))
        if attention_rank <= 1:
            print('Too short sentence, skipped', item, file=sys.stderr)
            return False
        
        # check maxlen
        if not len(item_wordpieces_grouped) <= self.MAX_LEN:
            print('Too long sentence, skipped', item, file=sys.stderr)
            return False
        # NOTE sentences truncated to 64 tokens
        if len(item_wordpieces) != attention_rank:
            print('Sentence does not match attention rank, skipped', item, file=sys.stderr)
            return False
        return True

    def aggregate_wordpiece_matrices(self, attention_matrices, tokens_grouped):
        # this functions connects wordpieces and aggregates their attention.
        midres_matrices = np.zeros((self.layer_count, self.head_count, len(tokens_grouped), attention_matrices.shape[3]))

        for tok_id, wp_ids in enumerate(tokens_grouped):
            midres_matrices[:,:,tok_id, :] = np.mean(attention_matrices[:,:,wp_ids, :], axis=2)

        res_matrices= np.zeros((self.layer_count, self.head_count, len(tokens_grouped), len(tokens_grouped)))

        for tok_id, wp_ids in enumerate(tokens_grouped):
            res_matrices[:,:,:, tok_id] = np.sum(midres_matrices[:, :, :, wp_ids], axis=3)

        return res_matrices

    def preprocess_matrices(self, attention_loaded):
        for sent_idx in tqdm(self.sentence_idcs[:], desc="Preprocessing attention for sentences"):
            if not self.check_wordpieces(sent_idx, attention_loaded):
                self.sentence_idcs.remove(sent_idx)
                continue

            matrices_id = 'arr_' + str(sent_idx)
            sent_matrices = np.array(attention_loaded[matrices_id])
            if self.WITH_EOS:
                sent_matrices = sent_matrices[:,:,:-1, :-1]
            if self.WITH_CLS:
                sent_matrices = sent_matrices[:, :, 1:, 1:]
            # the max trick -- for each row subtract its max
            # from all of its components to get the values into (-inf, 0]
            if not self.NO_SOFTMAX:
                sent_matrices = sent_matrices - np.max(sent_matrices, axis=3, keepdims=True)
                exp_matrix = np.exp(sent_matrices)
                sent_matrices = exp_matrix / np.sum(exp_matrix, axis=3, keepdims=True)
            else:
                sent_matrices = sent_matrices / np.sum(sent_matrices, axis=3, keepdims=True)
            sent_matrices = self.aggregate_wordpiece_matrices(sent_matrices, self.tokens_grouped[sent_idx])
            self.matrices.append(sent_matrices)
