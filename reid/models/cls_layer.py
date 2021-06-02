from __future__ import print_function

from torch import nn
from torch.nn import functional as F
from torch.autograd import Function
from torch.nn import init
import torch
import numpy as np

from reid.loss.sofmax_loss import SoftmaxLoss

from reid.models.mish import Mish
from reid.models.proxy_table import Proxy_table
from reid.models.bn_neck import BN_neck



class OldClsTarget(nn.Module):
    def __init__(self,
                 target_pids,
                 in_dim,
                 t_cam_ids,
                 t_alpha,
                 t_beta,
                 t_switch):
        super(OldClsTarget, self).__init__()
        # target
        self.t_alpha = t_alpha
        self.t_beta = t_beta
        self.t_num_cam = len(t_cam_ids)
        t_cumsum = np.cumsum(t_cam_ids)
        self.t_cumsum = np.append(t_cumsum, 0)
        self.t_switch = t_switch
        self.softmax = SoftmaxLoss()
        self.norm = F.normalize # todo important
        self.drop = nn.Dropout(p=0.5) # todo important
        em_list = []

        if in_dim > 0:
            in_dim = in_dim
        else:
            in_dim = 2048

        # hyper parameter
        self.use_single = False

        for c, cls in enumerate(t_cam_ids):
            em_cls = nn.Parameter(torch.zeros(cls, in_dim))
            setattr(self, 'em_c{}'.format(c), em_cls)
            em_list.append(em_cls)

        setattr(self, 'em_all', torch.cat(em_list, dim=0).cuda())

    def forward(self, x, pids, img_index=None, cams=None, labels=None, epoch=None):
        # target
        index_single = {}
        alpha = self.t_alpha
        out = {}
        x = self.norm(x)
        x = self.drop(x)
        if not isinstance(labels, torch.Tensor):
            for c in range(self.t_num_cam):
                index_c = torch.nonzero(cams == c).squeeze(1)
                index_single[c] = index_c
                if index_c.shape[0]:
                    target_c = pids[index_c]
                    em_c = getattr(self, 'em_c{}'.format(c))
                    outputs = Proxy_table(em_c, alpha) \
                        (x[index_c], target_c)

                    out[c] = outputs
            loss = 0.
            for c, v in out.items():
                v = v / self.t_beta
                loss += 1 / self.t_num_cam * self.softmax(v, pids[index_single[c]])
        else:
            loss = 0.

            # hyper parameter
            if self.use_single:
                labels = labels / (torch.max(labels, dim=1, keepdim=True)[0] + 1e-20)
                outputs = Proxy_table(self.em_all, alpha=alpha)(x, img_index)
                outputs = outputs / self.t_beta
                outputs = F.log_softmax(outputs, dim=1)
                loss = (-(labels*outputs)).sum(1).mean(0)
            else:

                for c in range(self.t_num_cam):
                    y = labels[:, self.t_cumsum[c - 1]:self.t_cumsum[c]]
                    y = y / (torch.max(y, dim=1, keepdim=True)[0]+1e-20) # good
                    index_neighbor = torch.zeros_like(cams)
                    index_neighbor -= 1.
                    index_c = torch.nonzero(cams == c).squeeze(1)
                    index_neighbor[index_c] = pids[index_c]
                    em_c = getattr(self, 'em_c{}'.format(c))
                    outputs = Proxy_table(em_c, alpha=alpha)(x, index_neighbor)
                    outputs = outputs / self.t_beta
                    outputs = F.log_softmax(outputs, dim=1)
                    loss_c = (-(y * outputs)).sum(1).mean(0)
                    loss += 1 / self.t_num_cam * loss_c

                # try one camera
                # if epoch > 8:
                #     labels = labels / (torch.max(labels, dim=1, keepdim=True)[0]+1e-20)
                #     # labels = labels / labels.sum(dim=1)[:, None]
                #     outputs = Proxy_table(self.em_all, alpha=alpha)(x, img_index)
                #     outputs = outputs / self.t_beta
                #     outputs = F.log_softmax(outputs, dim=1)
                #     loss = loss + (-(labels*outputs)).sum(1).mean(0)

        return loss