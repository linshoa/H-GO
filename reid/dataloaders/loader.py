from __future__ import absolute_import
import os.path as osp
from PIL import Image
import torch


class Preprocessor(object):
    def __init__(self, dataset, transform=None):
        super(Preprocessor, self).__init__()
        self.dataset = dataset
        self.transform = transform

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, indices):
        if isinstance(indices, (tuple, list)):
            return [self._get_single_item(index) for index in indices]
        return self._get_single_item(indices)

    def _get_single_item(self, index):
        fpath, pid, camid = self.dataset[index]
        img = Image.open(fpath).convert('RGB')
        if self.transform is not None:
            img = self.transform(img)
        return img, fpath, pid, camid, index


class CamStylePreprocessor(object):
    def __init__(self, dataset, num_cam=6, transform=None, use_camstyle=True):
        super(CamStylePreprocessor, self).__init__()
        self.dataset = dataset
        self.transform = transform
        self.num_cam = num_cam
        self.use_camstyle = use_camstyle
        print('Use camstyle : ', self.use_camstyle)

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, indices):
        if isinstance(indices, (tuple, list)):
            return [self._get_single_item(index) for index in indices]
        return self._get_single_item(indices)

    def _get_single_item(self, index):
        fpath, pid, camid = self.dataset[index]
        fpath_dir, fname = osp.split(fpath)
        if self.use_camstyle:
            fpath_dir = fpath_dir.replace('bounding_box_train',
                                          'bounding_box_train_camstyle')
            sel_cam = torch.randperm(self.num_cam)[0]
            if sel_cam == camid:
                img = Image.open(fpath).convert('RGB')
            else:
                if 'msmt' in fpath_dir:
                    fname = fname[:-4] + '_fake_' + str(sel_cam.numpy() + 1) + '.jpg'
                else:
                    fname = fname[:-4] + '_fake_' + str(camid + 1) + 'to' + str(sel_cam.numpy() + 1) + '.jpg'
                fpath = osp.join(fpath_dir, fname)
                img = Image.open(fpath).convert('RGB')
        else:
            img = Image.open(fpath).convert('RGB')

        if self.transform is not None:
            img = self.transform(img)
        # index can be used as pseudo label
        return img, fpath, pid, camid, index


class IterLoader:
    def __init__(self, loader, length=None):
        self.loader = loader
        self.length = length
        self.iter = None

    def __len__(self):
        if (self.length is not None):
            return self.length
        return len(self.loader)

    def new_epoch(self):
        self.iter = iter(self.loader)

    def next(self):
        try:
            return next(self.iter)
        except:
            self.iter = iter(self.loader)
            return next(self.iter)