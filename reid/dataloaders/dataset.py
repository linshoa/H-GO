import os
import os.path as osp
import re
from glob import glob
from collections import defaultdict
import numpy as np


class Person(object):
    def __init__(self, data_dir, target, source):
        self.target = target
        self.source = source

        # source / target image root
        self.target_root = osp.join(data_dir, self.target)
        self.source_root = osp.join(data_dir, self.source)
        self.train_path = 'bounding_box_train'
        self.camstyle_train_path = 'bounding_box_train_camstyle'
        self.gallery_path = 'bounding_box_test'
        self.query_path = 'query'

        # source / target misc
        self.target_train_fake, self.target_gallery, self.target_query = [], [], []
        self.target_train_real = []
        self.source_train, self.source_gallery, self.source_query = [], [], []

        self.t_train_pids, self.t_gallery_pids, self.t_query_pids = 0, 0, 0
        self.s_train_pids, self.s_gallery_pids, self.s_query_pids = 0, 0, 0
        self.t_train_cam_2_imgs, self.s_train_cam_2_imgs = [], []
        self.target_cam_nums = self.set_cam_dict(self.target)
        self.source_cam_nums = self.set_cam_dict(self.source)

        self.load_dataset()

    @staticmethod
    def set_cam_dict(name):
        cam_dict = {}
        cam_dict['market'] = 6
        cam_dict['duke'] = 8
        cam_dict['msmt17'] = 15
        cam_dict['cuhk03'] = 2
        return cam_dict[name]

    def preprocess(self, root, name_path, relable=True):
        pattern = re.compile(r'([-\d]+)_c([-\d]+)')
        pids_dict = {}
        ret = []
        if name_path == self.train_path:
            cam_2_imgs = defaultdict(int)
        else:
            cam_2_imgs = None

        if 'cuhk03' in root:
            fpaths = sorted(glob(osp.join(root, name_path, '*.png')))
        else:
            fpaths = sorted(glob(osp.join(root, name_path, '*.jpg')))

        for fpath in fpaths:
            fname = osp.basename(fpath)
            pid, cam = map(int, pattern.search(fname).groups())
            cam -= 1  # start from 0
            if pid == -1: continue
            if relable:
                # relable
                if pid not in pids_dict:
                    pids_dict[pid] = len(pids_dict)
                # for train
                cam_2_imgs[cam] += 1
            else:
                # not relabel
                if pid not in pids_dict:
                    pids_dict[pid] = pid

            pid = pids_dict[pid]
            ret.append([fpath, pid, cam])

        if cam_2_imgs:
            cam_2_imgs = list(np.asarray(
                sorted(cam_2_imgs.items(), key=lambda e: e[0]))[:, 1])
            print(root, cam_2_imgs)
        return ret, len(pids_dict), cam_2_imgs

    def preprocess_cam(self, root, name_path, fake, relable=True):
        pattern = re.compile(r'([-\d]+)_c([-\d]+)')
        pids_dict = {}
        ret = []
        cam_2_imgs = defaultdict(list)

        if 'cuhk03' in root:
            fpaths = sorted(glob(osp.join(root, name_path, '*.png')))
        else:
            fpaths = sorted(glob(osp.join(root, name_path, '*.jpg')))

        for i, fpath in enumerate(fpaths):
            fname = osp.basename(fpath)
            pid, cam = map(int, pattern.search(fname).groups())
            cam -= 1  # start from 0
            if pid == -1: continue
            cam_2_imgs[cam].append([fpath, pid, cam])

        # index start from camera 0-c
        if fake:
            # fake pid
            for c in sorted(cam_2_imgs.keys()):
                if c not in pids_dict:
                    pids_dict[c] = {}
                for i, (fpath, pid, cam) in enumerate(cam_2_imgs[c]):
                    # relable
                    pids_dict[cam][i] = len(pids_dict[cam]) # within camera, pid start from 0
                    pid = pids_dict[cam][i]
                    ret.append([fpath, pid, cam])
            cam_2_imgs = [len(pids_dict[e]) for e in sorted(pids_dict.keys())]
            print(root, cam_2_imgs)
        else:
            for c in sorted(cam_2_imgs.keys()):
                for i, (fpath, pid, cam) in enumerate(cam_2_imgs[c]):
                    if pid not in pids_dict:
                        pids_dict[pid] = len(pids_dict)
                    pid = pids_dict[pid]
                    ret.append([fpath, pid, cam])
        return ret, len(pids_dict), cam_2_imgs

    def load_dataset(self):
        # train dataset order is not different !!!!
        self.target_train_real, self.t_train_pids, _ = self.preprocess_cam(self.target_root, self.train_path, fake=False, relable=True)
        self.target_train_fake, _, self.t_train_cam_2_imgs = self.preprocess_cam(self.target_root, self.train_path, fake=True, relable=True)
        self.source_train, self.s_train_pids, self.s_train_cam_2_imgs = self.preprocess(self.source_root, self.train_path, True)
        self.target_gallery, self.t_gallery_pids, _ = self.preprocess(self.target_root, self.gallery_path, False)
        self.source_gallery, self.s_gallery_pids, _ = self.preprocess(self.source_root, self.gallery_path, False)
        self.target_query, self.t_query_pids, _ = self.preprocess(self.target_root, self.query_path, False)
        self.source_query, self.s_query_pids, _ = self.preprocess(self.source_root, self.query_path, False)

        print("  subset     | # ids | # imgs")
        print('  --------------------------')
        print("{} train   | {:5d} | {:8d}".format(self.source, self.s_train_pids, len(self.source_train)))
        print("{} query   | {:5d} | {:8d}".format(self.source, self.s_query_pids, len(self.source_query)))
        print("{} gallery | {:5d} | {:8d}".format(self.source, self.s_gallery_pids, len(self.source_gallery)))
        print('  --------------------------')
        print("{} train   | {:5d} | {:8d}".format(self.target, self.t_train_pids, len(self.target_train_fake)))
        print("{} query   | {:5d} | {:8d}".format(self.target, self.t_query_pids, len(self.target_query)))
        print("{} gallery | {:5d} | {:8d}".format(self.target, self.t_gallery_pids, len(self.target_gallery)))


