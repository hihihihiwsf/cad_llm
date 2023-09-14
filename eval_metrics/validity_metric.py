import torch
from torchmetrics import Metric

from geometry.parse import get_curve


class ValidityMetric(Metric):
    def __init__(self):
        super().__init__()
        self.add_state("valid", default=torch.tensor(0), dist_reduce_fx="sum")
        self.add_state("total", default=torch.tensor(0), dist_reduce_fx="sum")

    def update(self, pred):
        """
        pred contains point entities
        """
        for point_entity in set(pred):
            self.total += 1
            curve = get_curve(point_entity)
            if curve and curve.good:
                self.valid += 1

    def compute(self):
        return self.valid / self.total
