from __future__ import print_function, absolute_import
import time

from reid.loss.triplet_loss import TripletLoss, SoftTripletLoss
from reid.loss.sofmax_loss import SoftmaxLoss
from reid.loss.CrossEntropyLabelSmooth import CrossEntropyLabelSmooth
from reid.evaluation.meters import AverageMeter
from reid.evaluation.classification import accuracy


class PreTrainer(object):
    def __init__(self, model, num_classes, margin=0.0):
        super(PreTrainer, self).__init__()
        self.model = model
        self.criterion_ce = CrossEntropyLabelSmooth(num_classes).cuda()
        self.criterion_triple = TripletLoss(margin=margin).cuda()

    def train(self, epoch, data_loader_source, data_loader_target,  optimizer, train_iters=200, print_freq=1):
        self.model.train()

        batch_time = AverageMeter()
        data_time = AverageMeter()
        losses_ce = AverageMeter()
        losses_tr = AverageMeter()
        precisions = AverageMeter()

        end = time.time()

        for i in range(train_iters):
            source_inputs = data_loader_source.next()
            target_inputs = data_loader_target.next()
            data_time.update(time.time() - end)

            s_x, targets = self._parse_data(source_inputs)
            t_x, _ = self._parse_data(target_inputs)
            s_x, s_cls_out = self.model(s_x)

            # target samples: only forward
            t_x, _ = self.model(t_x)

            # backward main #
            loss_ce, loss_tr, prec1 = self._forward(s_x, s_cls_out, targets)
            loss = loss_ce + loss_tr

            losses_ce.update(loss_ce.item())
            losses_tr.update(loss_tr.item())
            precisions.update(prec1)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            batch_time.update(time.time() - end)
            end = time.time()

            if ((i + 1) % print_freq == 0):
                print('Epoch: [{}][{}/{}]\t'
                      'Time {:.3f} ({:.3f})\t'
                      'Data {:.3f} ({:.3f})\t'
                      'Loss_ce {:.3f} ({:.3f})\t'
                      'Loss_tr {:.3f} ({:.3f})\t'
                      'Prec {:.2%} ({:.2%})'
                      .format(epoch, i + 1, train_iters,
                              batch_time.val, batch_time.avg,
                              data_time.val, data_time.avg,
                              losses_ce.val, losses_ce.avg,
                              losses_tr.val, losses_tr.avg,
                              precisions.val, precisions.avg))

    def _parse_data(self, inputs):
        imgs, _, pids, _, _ = inputs
        inputs = imgs.cuda()
        targets = pids.cuda()
        return inputs, targets

    def _forward(self, s_features, s_outputs, targets):
        loss_ce = self.criterion_ce(s_outputs, targets)
        if isinstance(self.criterion_triple, SoftTripletLoss):
            loss_tr = self.criterion_triple(s_features, s_features, targets)
        elif isinstance(self.criterion_triple, TripletLoss):
            loss_tr, _ = self.criterion_triple(s_features, targets)
        prec, = accuracy(s_outputs.data, targets.data)
        prec = prec[0]

        return loss_ce, loss_tr, prec