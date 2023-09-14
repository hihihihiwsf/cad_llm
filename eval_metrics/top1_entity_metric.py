from torchmetrics import Metric
import torch


class Top1EntityMetric(Metric):

    def __init__(self):
        super().__init__()
        self.add_state("correct", default=torch.tensor(0), dist_reduce_fx="sum")
        self.add_state("total", default=torch.tensor(0), dist_reduce_fx="sum")

    def update(self, pred, true):
        """
        pred and true contain point entities
        """
        self.total += 1
        if pred and any(p and p in true for p in pred):
            self.correct += 1

    def compute(self):
        return self.correct.float() / self.total
