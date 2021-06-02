from __future__ import print_function, absolute_import
import time
from collections import OrderedDict, defaultdict
import torch
import numpy as np
import os.path as osp
import scipy.io as sci
from sklearn.metrics import euclidean_distances
from sklearn.metrics.pairwise import cosine_similarity

from reid.evaluation.ranking import cmc, mean_ap, map_cmc
from reid.evaluation.meters import AverageMeter
from reid.lib.serialization import to_torch, mkdir_if_missing


def extract_cnn_feature(backbone, inputs, layer=None):
    inputs = to_torch(inputs).cuda()
    with torch.no_grad():
        outputs = backbone(inputs)
        outputs = outputs.data.cpu()
    # if isinstance(layer, int):
    #     outputs = outputs[layer].data.cpu()
    # else:
    #     outputs = outputs[-1].data.cpu()  # norm feature
    return outputs


def extract_features(model, data_loader):

    model.eval()
    batch_time = AverageMeter()
    data_time = AverageMeter()

    features = OrderedDict()
    labels = OrderedDict()

    end = time.time()
    for i, (imgs, fpaths, pids, cams, indices) in enumerate(data_loader):
        data_time.update(time.time() - end)

        outputs = extract_cnn_feature(model, imgs)
        for fpath, output, pid in zip(fpaths, outputs, pids):
            features[fpath] = output
            labels[fpath] = pid

        batch_time.update(time.time() - end)
        end = time.time()

    return features, labels

def get_gause_sim(q_f, g_f, delta=1.):
    distance = euclidean_distances(q_f, g_f, squared=True)
    distance /= 2 *delta**2
    dist = np.exp(distance)
    return dist

def get_cossim(q_f, g_f):
    dist = -cosine_similarity(q_f, g_f)
    return dist

def get_euclidean(q_f, g_f):
    m, n = q_f.size(0), g_f.size(0)
    dist = torch.pow(q_f, 2)
    dist = dist.sum(dim=1, keepdim=True)
    dist = dist.expand(m, n)

    dist = torch.pow(q_f, 2).sum(dim=1, keepdim=True).expand(m, n) + \
           torch.pow(g_f, 2).sum(dim=1, keepdim=True).expand(n, m).t()
    dist.addmm_(1, -2, q_f, g_f.t())
    # We use clamp to keep numerical stability
    dist = torch.clamp(dist, 1e-8, np.inf)
    return dist



def pairwise_distance(query_features, gallery_features, query=None, gallery=None):
    # fpath, pid, camera
    x = torch.cat([query_features[f].unsqueeze(0) for f, _, _ in query], 0)
    y = torch.cat([gallery_features[f].unsqueeze(0) for f, _, _ in gallery], 0)

    m, n = x.size(0), y.size(0)
    x = x.view(m, -1)
    y = y.view(n, -1)

    # print("euclidean")
    # dist = get_euclidean(x, y)



    # print("get_cossim")
    # dist = get_cossim(x.numpy(), y.numpy())

    print("guase")
    dist = get_gause_sim(x.numpy(), y.numpy(), delta=1.)
    print(dist[0, :5])

    return dist


def evaluate_all(distmat, query=None, gallery=None,
                 query_ids=None, gallery_ids=None,
                 query_cams=None, gallery_cams=None,
                 cmc_topk=(1, 5, 10, 20)):
    if query is not None and gallery is not None:
        query_ids = [pid for _, pid, _ in query]
        gallery_ids = [pid for _, pid, _ in gallery]
        query_cams = [cam for _, _, cam in query]
        gallery_cams = [cam for _, _, cam in gallery]
    else:
        assert (query_ids is not None and gallery_ids is not None
                and query_cams is not None and gallery_cams is not None)

    # Evaluation
    mAP, all_cmc = map_cmc(distmat, query_ids, gallery_ids, query_cams, gallery_cams)
    print('Mean AP: {:4.1%}'.format(mAP))
    print('CMC Scores')
    for k in cmc_topk:
        print('  top-{:<4}{:12.1%}'
              .format(k, all_cmc[k - 1]))
    return mAP


def save_feature(features, fpaths, labels, camids, save_name):
    result = {'features': np.squeeze(features),
              'fpaths': np.squeeze(fpaths),
              'labels': np.squeeze(labels),
              'camids': np.squeeze(camids)
              }
    save_dir = '/hdd/sdb/lsc/pytorch/PGM/logs/features_visualization'
    mkdir_if_missing(save_dir)
    sci.savemat(osp.join(save_dir, save_name+'_result.mat'),
                result)


class Evaluator(object):
    def __init__(self, backbone):
        super(Evaluator, self).__init__()
        self.backbone = backbone
        self.backbone.eval()
        self.visualize = False

    # for train
    def evaluate(self, query_loader, gallery_loader, query, gallery):
        query_features, _ = extract_features(self.backbone, query_loader)
        gallery_features, _ = extract_features(self.backbone, gallery_loader)
        distmat = pairwise_distance(query_features, gallery_features, query, gallery)

        return evaluate_all(distmat, query=query, gallery=gallery)

    def extract_tgt_train_features(self, train_loader, save_name=None, layer=None):
        features, paths, labels, camids = [], [], [], []
        for i, (imgs, fpaths, pids, cams, _) in enumerate(train_loader):
            outputs = extract_cnn_feature(self.backbone, imgs, layer)
            for fpath, feature, pid, cam in zip(fpaths, outputs, pids, cams):
                features.append(feature.numpy())
                paths.append(fpath)
                labels.append(pid.numpy())
                camids.append(cam.numpy())
        features, fpaths, labels, camids = \
            np.array(features), np.array(paths), np.array(labels), np.array(camids)
        if save_name:
            save_feature(features, fpaths, labels, camids, save_name)
        return features, fpaths, labels, camids

    @staticmethod
    def extract_pids(data_loader):
        labels = []
        for i, (_, _, pids, _, _) in enumerate(data_loader):
            for pid in pids:
                labels.append(pid)

        labels = np.array(labels)
        return labels

    @staticmethod
    def load_feature(name):
        file_dir = '/hdd/sdb/lsc/pytorch/PGM/logs/features_visualization'

        result = sci.loadmat(osp.join(file_dir, name+'_result.mat'))
        features = result['features']
        print(features.shape)
        labels = result['labels']
        fpaths = result['fpaths']
        camids = result['camids']
        labels = np.squeeze(labels)
        fpaths = np.squeeze(fpaths)
        camids = np.squeeze(camids)
        return features, fpaths, labels, camids,
