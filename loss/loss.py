import torch
import torch.nn.functional as F
from torch import nn

from geomloss import SamplesLoss  # See also ImagesLoss, VolumesLoss


class SetCriterion(nn.Module):
    def __init__(self, reduction='mean'):
        super().__init__()
        self.criterion_CE = nn.CrossEntropyLoss(reduction=reduction)



    def forward(self, outputs, targets, type):
        if type == 'CE':
            loss = self.criterion_CE(outputs, targets)

        elif type == 'EMD':
            loss = SamplesLoss(loss="sinkhorn", p=2, blur=0.3)
            # outputs = F.softmax(outputs, dim=1)
            # targets = F.softmax(targets, dim=1)
            loss = loss(outputs, targets)
            loss=torch.mean(loss)

        return loss


if __name__ == '__main__':
    x_input = torch.randn(128, 9, 1)
    y_input = torch.randn(128, 9, 1)
    x = SetCriterion()

    y = x(x_input, y_input, 'EMD')
    print(y)
