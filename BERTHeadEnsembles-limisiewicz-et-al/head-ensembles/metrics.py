import numpy as np
from abc import abstractmethod
from collections import defaultdict


class Metric:

    def __init__(self, dependency, *args, **kwargs):
        self.dependency = dependency

    @abstractmethod
    def __call__(self, *args, **kwargs):
        pass

    @abstractmethod
    def reset_state(self):
        pass

    @abstractmethod
    def update_state(self, *args, **kwargs):
        pass

    @abstractmethod
    def result(self):
        pass


class DepAcc(Metric):

    def __init__(self, dependency, relation_label, dependent2parent=True):
        """ Dependency accuracy for a given relation label and direction. """
        self.relation_label = relation_label
        self.dependent2parent = dependent2parent

        self.retrieved = 0
        self.total = 0
        super().__init__(dependency)

    def __call__(self, sent_idcs, matrices):
        """ Takes indices of sentences in parsed dependency and the list of corresponding matrices."""
        for sent_id, matrix in zip(sent_idcs, matrices):
            self.update_state(sent_id, matrix)

    def reset_state(self):
        self.retrieved = 0
        self.total = 0

    def update_state(self, sent_id, matrix):
        if matrix is not None:
            np.fill_diagonal(matrix, 0.)
            max_row = matrix.argmax(axis=1)

            rel_pairs = self.dependency.relations[sent_id][self.relation_label]
            if not self.dependent2parent:
                rel_pairs = list(map(tuple, map(reversed, rel_pairs)))
            self.retrieved += sum([max_row[attending] == attended for attending, attended in rel_pairs])
            self.total += len(rel_pairs)

    def result(self):
        if not self.total:
            return 0.
        return self.retrieved / self.total


class UAS(Metric):
    """ Unlabeled Attachment Score """
    def __init__(self, dependency):
        self.all_correct = 0
        self.all_predicted = 0
        self.all_gold = 0
        self.rel_wise_gold = {}
        self.rel_wise_pred = {}
        self.rel_wise_prec = {}
        super().__init__(dependency)

    def __call__(self, sent_idcs, predicted_relations):
        for sent_id, sent_predicted_relations in zip(sent_idcs, predicted_relations):
            self.update_state(sent_id, sent_predicted_relations)
        
        for k in self.rel_wise_gold.keys():
            if k not in self.rel_wise_pred.keys():
                self.rel_wise_prec[k] = 0
            else:
                self.rel_wise_prec[k] = self.rel_wise_pred[k]/self.rel_wise_gold[k]
        print(self.rel_wise_prec)

    def reset_state(self):
        self.all_correct = 0
        self.all_predicted = 0
        self.all_gold = 0

    def update_state(self, sent_id, sent_predicted_relations):
        if sent_predicted_relations is not None:
            rel_pairs = self.dependency.unlabeled_relations[sent_id]
            rel_pairs_labeled = self.dependency.labeled_relations[sent_id]
            self.all_gold += len(rel_pairs)
            self.all_predicted += len(sent_predicted_relations)
            self.all_correct += len(set(rel_pairs).intersection(set(sent_predicted_relations)))

            for rel in rel_pairs_labeled:
                rel_name = rel[2]
                if rel_name in self.rel_wise_gold.keys():
                    self.rel_wise_gold[rel_name] += 1
                else:
                    self.rel_wise_gold[rel_name] = 1

            correct_edges = set(rel_pairs).intersection(set(sent_predicted_relations))

            for i, rel in enumerate(rel_pairs):
                if rel in correct_edges:
                    rel_name = rel_pairs_labeled[i][2]
                    if rel_name in self.rel_wise_pred.keys():
                        self.rel_wise_pred[rel_name] += 1
                    else:
                        self.rel_wise_pred[rel_name] = 1


    def result(self):
        if not self.all_correct:
            return 0.
        return 2. / (self.all_predicted / self.all_correct + self.all_gold / self.all_correct), self.rel_wise_prec

class UUAS(Metric):
    """ Unlabeled Attachment Score """
    def __init__(self, dependency):
        self.all_correct = 0
        self.all_predicted = 0
        self.all_gold = 0
        self.rel_wise_gold = {}
        self.rel_wise_pred = {}
        self.rel_wise_prec = {}
        super().__init__(dependency)

    def __call__(self, sent_idcs, predicted_relations):
        for sent_id, sent_predicted_relations in zip(sent_idcs, predicted_relations):
            self.update_state(sent_id, sent_predicted_relations)
        
        for k in self.rel_wise_gold.keys():
            if k not in self.rel_wise_pred.keys():
                self.rel_wise_prec[k] = 0
            else:
                self.rel_wise_prec[k] = self.rel_wise_pred[k]/self.rel_wise_gold[k]
        print(self.rel_wise_prec)

    def reset_state(self):
        self.all_correct = 0
        self.all_predicted = 0
        self.all_gold = 0

    def update_state(self, sent_id, sent_predicted_relations):
        if sent_predicted_relations is not None:
            rel_pairs = self.dependency.unlabeled_relations[sent_id]
            rel_pairs_undirected = [(r[1], r[0]) for r in self.dependency.unlabeled_relations[sent_id]]
            rel_pairs_undirected = rel_pairs + rel_pairs_undirected
            rel_pairs_labeled = self.dependency.labeled_relations[sent_id]
            self.all_gold += len(rel_pairs)
            self.all_predicted += len(sent_predicted_relations)
            self.all_correct += len(set(rel_pairs_undirected).intersection(set(sent_predicted_relations)))

            for rel in rel_pairs_labeled:
                rel_name = rel[2]
                if rel_name in self.rel_wise_gold.keys():
                    self.rel_wise_gold[rel_name] += 1
                else:
                    self.rel_wise_gold[rel_name] = 1

            correct_edges = set(rel_pairs).intersection(set(sent_predicted_relations))

            for i, rel in enumerate(rel_pairs):
                if rel in correct_edges:
                    rel_name = rel_pairs_labeled[i][2]
                    if rel_name in self.rel_wise_pred.keys():
                        self.rel_wise_pred[rel_name] += 1
                    else:
                        self.rel_wise_pred[rel_name] = 1


    def result(self):
        if not self.all_correct:
            return 0.
        return 2. / (self.all_predicted / self.all_correct + self.all_gold / self.all_correct), self.rel_wise_prec


class LAS(Metric):
    """ Labeled Attachment Score """
    def __init__(self, dependency):
        self.all_correct = 0
        self.all_predicted = 0
        self.all_gold = 0
        self.rel_wise_gold = {}
        self.rel_wise_pred = {}
        self.rel_wise_prec = {}
        self.labeled_det = {}
        super().__init__(dependency)

    def __call__(self, sent_idcs, predicted_relations):
        for sent_id, sent_predicted_relations in zip(sent_idcs, predicted_relations):
            self.update_state(sent_id, sent_predicted_relations)
        
        for k in self.rel_wise_gold.keys():
            if k not in self.rel_wise_pred.keys():
                self.rel_wise_prec[k] = 0
            else:
                self.rel_wise_prec[k] = self.rel_wise_pred[k]/self.rel_wise_gold[k]
        print(self.rel_wise_prec)
        print(self.labeled_det)

    def reset_state(self):
        self.all_correct = 0
        self.all_predicted = 0
        self.all_gold = 0

    def update_state(self, sent_id, sent_predicted_relations):
        if sent_predicted_relations is not None:
            rel_pairs = self.dependency.labeled_relations[sent_id]
            sent_predicted_relations = \
                [(d, p, self.dependency.reverse_label_map.get(l, 'dep')) for d, p, l in sent_predicted_relations]
            self.all_gold += len(rel_pairs)
            self.all_predicted += len(sent_predicted_relations)
            self.all_correct += len(set(rel_pairs).intersection(set(sent_predicted_relations)))

            for i, rel in enumerate(rel_pairs):
                rel_name = rel[2]
                if rel_name in self.rel_wise_gold.keys():
                    self.rel_wise_gold[rel_name] += 1
                else:
                    self.rel_wise_gold[rel_name] = 1
                if rel_name == 'det' or rel_name == 'determiner':
                    predicted_label = sent_predicted_relations[i][2]
                    if predicted_label in self.labeled_det.keys():
                        self.labeled_det[predicted_label] += 1
                    else:
                        self.labeled_det[predicted_label] = 1
            
            for rel in list(set(rel_pairs).intersection(set(sent_predicted_relations))):
                rel_name = rel[2]
                if rel_name in self.rel_wise_pred.keys():
                    self.rel_wise_pred[rel_name] += 1
                else:
                    self.rel_wise_pred[rel_name] = 1


    def result(self):
        if not self.all_correct:
            return 0.
        return 2. / (self.all_predicted / self.all_correct + self.all_gold / self.all_correct), self.rel_wise_prec


