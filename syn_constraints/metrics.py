from torchmetrics import Metric
import torch


class IoU(Metric):
    def __init__(self):
        super().__init__()
        self.add_state("iou", default=torch.tensor(0.), dist_reduce_fx="sum")
        self.add_state("total", default=torch.tensor(0), dist_reduce_fx="sum")

    def update(self, pred_set, target_set):
        if not pred_set and not target_set:
            self.iou += 1
        else:
            self.iou += len(pred_set.intersection(target_set)) / len(pred_set.union(target_set))
        self.total += 1

    def compute(self):
        if self.total == 0:
            return 0
        return self.iou / self.total
