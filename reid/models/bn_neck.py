from torch import nn
from torch.nn import init


class BN_neck(nn.Module):
    def __init__(self, add_feature=2048):
        super(BN_neck, self).__init__()
        self.add_feature = add_feature
        if add_feature > 0 :
            self.feat = nn.Linear(2048, add_feature)
            init.kaiming_normal_(self.feat.weight, mode='fan_out')
            init.constant_(self.feat.bias, 0)
            self.feat_bn = nn.BatchNorm1d(add_feature)
        else:
            self.feat_bn = nn.BatchNorm1d(2048)

        self.feat_bn.bias.requires_grad_(False)
        init.constant_(self.feat_bn.weight, 1)
        init.constant_(self.feat_bn.bias, 0)

        self.relu = nn.ReLU()

    def forward(self, x):
        if self.add_feature > 0:
            x = self.feat(x)
            x = self.feat_bn(x)
            x = self.relu(x)
        else:
            x = self.feat_bn(x)
        return x