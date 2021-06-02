from __future__ import absolute_import

from torch import nn


class SoftmaxLoss(nn.Module):
    def __init__(self):
        super(SoftmaxLoss, self).__init__()
        self.crossentropy = nn.CrossEntropyLoss()

    def forward(self, inputs, targets):
        return self.crossentropy(inputs, targets) # including softmax~~~