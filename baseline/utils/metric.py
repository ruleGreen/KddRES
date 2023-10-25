from typing import List
from .util import seq2tuples


class SpanF1:
    def __init__(self):
        self.reset()

    def reset(self):
        self.n_label = 0.0
        self.n_pred = 0.0
        self.n_correct = 0.0

    def compute(self):
        p = self.n_correct / (self.n_pred + 1e-8)
        r = self.n_correct / (self.n_label + 1e-8)
        f1 = (2 * p * r) / (p + r + 1e-8)
        return f1

    def __call__(self, y_pred: List[List[str]], y_true: List[List[str]]):
        """
        Args:
            y_pred: List of List of predicted truth BIO labels.
            y_true: List of List of ground truth BIO labels.

        Return:
            float: The average metric of all seen examples.

        """
        n_pred = 0
        n_label = 0
        n_correct = 0
        for b in range(len(y_pred)):
            pred_chunks = set(seq2tuples(["O"] + y_pred[b] + ["O"]))
            true_chunks = set(seq2tuples(["O"] + y_true[b] + ["O"]))
            n_pred += len(pred_chunks)
            n_label += len(true_chunks)
            n_correct += len(pred_chunks & true_chunks)
        p = (n_correct / n_pred) if n_pred > 0 else 1
        r = (n_correct / n_label) if n_label > 0 else 1
        f1 = ((2 * p * r) / (p + r)) if p + r > 0 else 0

        self.n_pred += n_pred
        self.n_label += n_label
        self.n_correct += n_correct
        return f1
