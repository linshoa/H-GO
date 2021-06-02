import time
import torch
import numpy as np

from reid.evaluation.meters import AverageMeter
from reid.evaluation.classification import accuracy
from reid.lib.serialization import to_torch
from reid.loss.triplet_loss import TripletLoss


class Trainer(object):
    def __init__(self, backbone, cls_layer1):
        self.backbone = backbone
        self.share_layer = None
        self.cls_layer1 = cls_layer1
        self.cls_layer2 = None
        self.iteration = None
        self.xi = None
        self.graph_target = None
        self.margin = 0.5
        self.triplet = TripletLoss(margin=self.margin)


    def parse_with_graph_target(self, x):
        imgs, _, pids, camids, indices = x  # img, fpath, pid, camid, index
        inputs = imgs.cuda()
        pids = pids.cuda()
        camids = camids.cuda()
        if not isinstance(self.graph_target, np.ndarray):
            graph_target = '_'
        else:
            graph_target = torch.Tensor(self.graph_target[indices]).cuda()
        indices = indices.cuda()
        return inputs, pids, indices, camids, graph_target

    def target_train(self, optimizer, t_train_loader, epoch):
        batch_time = AverageMeter()
        data_time = AverageMeter()
        losses = AverageMeter()

        end = time.time()
        print_freq = len(t_train_loader) // 4
        self.backbone.train()
        self.cls_layer1.train()
        for i, x_target in enumerate(t_train_loader):
            data_time.update(time.time() - end)
            x_target, y_pid, y_idx, camids, graph_target = \
                self.parse_with_graph_target(x_target)
            x_target, x_target_bn = self.backbone(x_target)
            loss = self.cls_layer1(x_target_bn, y_pid, y_idx, camids, graph_target, epoch)

            losses.update(loss.item(), x_target.size(0))

            optimizer.zero_grad()
            loss.backward()

            optimizer.step()
            batch_time.update(time.time() - end)
            end = time.time()

            if (i + 1) % print_freq == 0:
                log = "Epoch: [{}][{}/{}], " \
                      "Time {:.3f} ({:.3f}), " \
                      "Data {:.3f} ({:.3f}), " \
                      "Loss {:.3f} ({:.3f}), " \
                    .format(epoch, i + 1, len(t_train_loader),
                            batch_time.val, batch_time.avg,
                            data_time.val, data_time.avg,
                            losses.val, losses.avg)
                print(log)


