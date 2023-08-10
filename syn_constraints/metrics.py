from torchmetrics import Metric
import torch


class PartIoU(Metric):
    def __init__(self):
        super().__init__()
        self.add_state("numerator", default=torch.tensor(0.), dist_reduce_fx="sum")
        self.add_state("denominator", default=torch.tensor(0), dist_reduce_fx="sum")

    def update(self, pred_set, target_set):
        if not pred_set and not target_set:
            return

        self.numerator += len(pred_set.intersection(target_set))
        self.denominator += len(pred_set.union(target_set))

    def compute(self):
        if self.denominator == 0:
            return torch.tensor(0.)
        return self.numerator / self.denominator


class PartAccuracy(Metric):
    def __init__(self):
        super().__init__()
        self.add_state("numerator", default=torch.tensor(0.), dist_reduce_fx="sum")
        self.add_state("denominator", default=torch.tensor(0), dist_reduce_fx="sum")

    def update(self, pred_set, target_set):
        if not pred_set and not target_set:
            return

        self.numerator += len(pred_set.intersection(target_set))
        self.denominator += len(target_set)

    def compute(self):
        if self.denominator == 0:
            return torch.tensor(0.)
        return self.numerator / self.denominator
