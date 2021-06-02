from __future__ import print_function, absolute_import
import argparse
import os.path as osp
import argparse
import sys
import time
import os
import torch.nn as nn
from torch.backends import cudnn
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import StepLR

from reid.lib.logging import Logger
from reid.lib.serialization import load_checkpoint, save_checkpoint, to_torch
from reid.lib.lr_scheduler import WarmupMultiStepLR
from reid.dataloaders.loader import *
import reid.dataloaders.transforms as T
from reid.dataloaders.dataset import Person
from reid import models
from reid.models.cls_layer import *
from reid.evaluation.evaluator import Evaluator
from reid.trainer.trainer import Trainer
from reid.detach_enhancement.perception_graph import HG
from reid.lib.serialization import copy_state_dict


def get_data(opt):
    dataset = Person(opt.data_dir, opt.target, opt.source)
    t_pids = dataset.t_train_pids
    s_pids = dataset.s_train_pids
    s_train = dataset.source_train
    t_train_fake = dataset.target_train_fake
    t_train_real = dataset.target_train_real
    s_query = dataset.source_query
    t_query = dataset.target_query
    s_gallery = dataset.source_gallery
    t_gallery = dataset.target_gallery
    t_cam_2_imgs = dataset.t_train_cam_2_imgs
    s_cam_2_imgs = dataset.s_train_cam_2_imgs

    height, width = opt.height, opt.width
    batch_size, workers = opt.batch_size, opt.workers

    normalizer = T.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )

    train_transformer = T.Compose([
        T.Resize((height, width), interpolation=3),
        T.RandomHorizontalFlip(p=0.5),
        T.Pad(10),
        T.RandomCrop((height, width)),
        T.ToTensor(),
        normalizer
    ])

    test_graph_transformer = T.Compose([
        T.Resize((height, width), interpolation=3),
        T.ColorJitter(brightness=opt.brightness, saturation=opt.saturation, hue=opt.hue),
        T.ToTensor(),
        normalizer,
    ])

    test_transformer = T.Compose([
        T.Resize((height, width), interpolation=3),
        T.ToTensor(),
        normalizer,
    ])

    t_train_loader = DataLoader(
        dataset=CamStylePreprocessor(t_train_fake, num_cam=len(t_cam_2_imgs),
                                     transform=train_transformer, use_camstyle=opt.use_camstyle), # todo next try
        batch_size=batch_size,
        num_workers=workers,
        shuffle=True,
        pin_memory=True,
        drop_last=True
    )
    t_traintest_loader = DataLoader(
        dataset=Preprocessor(t_train_real, test_graph_transformer),
        batch_size=batch_size,
        shuffle=False,
        num_workers=workers,
        pin_memory=True,
        drop_last=False
    )

    t_query_loader = DataLoader(
        dataset=Preprocessor(t_query, test_transformer),
        batch_size=batch_size,
        shuffle=False,
        num_workers=workers,
        pin_memory=True,
        drop_last=False
    )
    t_gallery_loader = DataLoader(
        dataset=Preprocessor(t_gallery, test_transformer),
        batch_size=batch_size,
        shuffle=False,
        num_workers=workers,
        pin_memory=True,
        drop_last=False
    )
    s_train_loader = DataLoader(
        dataset=Preprocessor(s_train, train_transformer),
        batch_size=batch_size,
        num_workers=workers,
        shuffle=True,
        pin_memory=True,
        drop_last=True
    )
    s_traintest_loader = DataLoader(
        dataset=Preprocessor(s_train, test_transformer),
        batch_size=batch_size,
        shuffle=False,
        num_workers=workers,
        pin_memory=False,
        drop_last=False
    )
    s_query_loader = DataLoader(
        dataset=Preprocessor(s_query, test_transformer),
        batch_size=batch_size,
        shuffle=False,
        num_workers=workers,
        pin_memory=False,
        drop_last=False
    )
    s_gallery_loader = DataLoader(
        dataset=Preprocessor(s_gallery, test_transformer),
        batch_size=batch_size,
        shuffle=False,
        num_workers=workers,
        pin_memory=False,
        drop_last=False
    )

    return t_pids, s_pids, t_train_loader, \
           t_query_loader, t_gallery_loader,\
           t_query, t_gallery, s_train_loader,\
           s_query_loader, s_gallery_loader, \
           s_query, s_gallery, \
           t_traintest_loader, s_traintest_loader, \
           t_train_fake, s_train, t_cam_2_imgs, s_cam_2_imgs


def main(opt):
    start = time.time()
    if opt.seed != -1:
        SEED = opt.seed
        np.random.seed(SEED)
        # this set the model state of all parameters
        torch.manual_seed(SEED)
        torch.cuda.manual_seed_all(SEED)
        cudnn.deterministic = True


    # For fast training.
    cudnn.benchmark = True
    torch.cuda.set_device(opt.gpu_ids)
    # Redirect print ot both console and log file
    if not opt.evaluate:
        sys.stdout = Logger(osp.join(opt.log_dir, 'log.txt'))
    print(time.ctime(), 'log-dir=', opt.log_dir)

    # Print logs
    print(opt)

    # Create data loaders
    t_pids, s_pids, t_train_loader, \
    t_query_loader, t_gallery_loader, \
    t_query, t_gallery, s_train_loader, \
    s_query_loader, s_gallery_loader, \
    s_query, s_gallery, \
    t_traintest_loader, s_traintest_loader, \
    t_train, s_train, t_cam_2_imgs, s_cam_2_imgs \
        = get_data(opt)

    # Create model
    backbone = models.create(args.arch, num_features=args.features, dropout=args.dropout, num_classes=0)
    old_cls_tgt = OldClsTarget(
                            len(t_train),
                            opt.features,
                            t_cam_2_imgs,
                            opt.alpha,
                            opt.tao,
                            opt.switch)
    # hyper parameter
    old_cls_tgt.use_single = opt.use_single

    # Load from checkpoint
    start_epoch = 0
    pre_path = osp.join('./logs/pretrain',args.arch)
    if opt.source == 'market' and opt.target == 'duke':
        checkpoint = load_checkpoint(osp.join(pre_path, 'market2duke/checkpoint.pth.tar'))
        copy_state_dict(checkpoint['backbone_dict'], backbone)
    elif opt.source == 'duke' and opt.target == 'market':
        checkpoint = load_checkpoint(osp.join(pre_path, 'duke2market/checkpoint.pth.tar'))
        copy_state_dict(checkpoint['backbone_dict'], backbone)
    elif opt.source == 'duke' and opt.target == 'msmt17':
        checkpoint = load_checkpoint(osp.join(pre_path, 'duke2msmt17/checkpoint.pth.tar'))
        copy_state_dict(checkpoint['backbone_dict'], backbone)
    elif opt.source == 'market' and opt.target == 'msmt17':
        checkpoint = load_checkpoint(osp.join(pre_path, 'market2msmt17/checkpoint.pth.tar'))
        copy_state_dict(checkpoint['backbone_dict'], backbone)

    # Set model
    backbone = nn.DataParallel(backbone, device_ids=[opt.gpu_ids])
    old_cls_tgt = old_cls_tgt.cuda()

    # Evaluator
    evaluator = Evaluator(backbone)
    if opt.evaluate:
        print('Test source {} : '.format(opt.source))
        evaluator.evaluate(s_query_loader, s_gallery_loader, s_query, s_gallery)
        print('Test target {} : '.format(opt.target))
        evaluator.evaluate(t_query_loader, t_gallery_loader, t_query, t_gallery)
        return

    # Optimizer
    params = []
    for key, value in backbone.named_parameters():
        if not value.requires_grad:
            continue
        params += [{"params": [value], "lr": args.lr, "weight_decay": args.weight_decay}]

    optimizer = torch.optim.Adam(params)
    scheduler = WarmupMultiStepLR(optimizer,
                                  [6],
                                  gamma=0.1,
                        warmup_factor=0.01,
                            warmup_iters=6)

    # Trainer
    trainer = Trainer(backbone, old_cls_tgt)
    # Start training
    for epoch in range(start_epoch, opt.epochs):
        if epoch % opt.switch == 0 and opt.switch > 0:

            evaluator = Evaluator(backbone)
            features, fpaths, labels, camids = \
                evaluator.extract_tgt_train_features(
                                    t_traintest_loader)
            graph = HG(opt.lamd, features, labels, camids)

            # hyper parameter
            graph.general_graph = opt.general_graph
            graph.homo_ap = opt.homo_ap

            start1 = time.time()
            if opt.only_graph:
                graph_target = graph.only_graph(ks=opt.ks, kd=opt.kd, k2=opt.k2)
            else:
                 if opt.cos_sim:
                     graph_target = graph.old_cos_propagation(ks=opt.ks, kd=opt.kd, k2=opt.k2)
                 else:
                    if opt.tracklet:
                        graph_target = graph.old_tracklet_propagation(ks=opt.ks, kd=opt.kd, k2=opt.k2)
                    else:
                        graph_target = graph.old_delta_propagation(ks=opt.ks, kd=opt.kd, k2=opt.k2)

            print('###graph time: ', time.time()-start1)
            trainer.graph_target = graph_target

        scheduler.step(epoch)
        trainer.target_train(optimizer,
                           t_train_loader,
                           epoch)

        save_checkpoint({
            'backbone_dict': backbone.module.state_dict(),
            'epoch': epoch + 1,
        }, fpath=osp.join(opt.log_dir, 'checkpoint.pth.tar'))
        print('\n * Finished epoch {:3d} \n'.
              format(epoch))

    print('Test target {} : '.format(opt.target))
    evaluator.evaluate(t_query_loader, t_gallery_loader, t_query, t_gallery)
    print(time.ctime(), ' Need Time:', time.time() - start)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="PGM")
    # gpu ids
    parser.add_argument('--gpu_ids', type=int, default=7)
    # random seed
    parser.add_argument('--seed', type=int, default=1)
    # source
    parser.add_argument('-s', '--source', type=str, default='duke',
                        choices=['market', 'duke', 'msmt17',])
    # target
    parser.add_argument('-t', '--target', type=str, default='market',
                        choices=['market', 'duke', 'msmt17', ])
    # imgs setting
    parser.add_argument('-b', '--batch-size', type=int, default=64)
    parser.add_argument('-j', '--workers', type=int, default=4)
    parser.add_argument('--height', type=int, default=256,
                        help="input height, default: 256")
    parser.add_argument('--width', type=int, default=128,
                        help="input width, default: 128")
    # model
    parser.add_argument('-a', '--arch', type=str, default='resnet50',
                        choices=models.names())
    parser.add_argument('--features', type=int, default=0)
    parser.add_argument('--dropout', type=float, default=0)
    # optimizer
    parser.add_argument('--lr', type=float, default=0.00035)
    parser.add_argument('--momentum', type=float, default=0.9)
    parser.add_argument('--weight-decay', type=float, default=5e-4)
    # training configs
    parser.add_argument('--resume', type=str, metavar='PATH')
    parser.add_argument('--evaluate', action='store_true', default=False,
                        help="evaluation only")
    parser.add_argument('--epochs', type=int, default=8)
    parser.add_argument('--epochs_decay', type=int, default=40)
    # misc
    working_dir = osp.dirname(osp.abspath(__file__))
    parser.add_argument('--data-dir', type=str, metavar='PATH',
                        default=osp.join(working_dir, 'data'))
    parser.add_argument('--log-dir', type=str, metavar='PATH',
                        default=osp.join(working_dir, 'logs/old/d2m'))

    parser.add_argument('--re', type=float, default=0.5)

    parser.add_argument('--alpha', type=float, default=0.01)
    parser.add_argument('--tao', type=float, default=0.05)
    parser.add_argument('--switch', type=int, default=1)
    parser.add_argument('--lamd', type=float, default=0.99)
    parser.add_argument('--ks', type=int, default=2)
    parser.add_argument('--kd', type=int, default=4)
    parser.add_argument('--k2', type=int, default=14)
    parser.add_argument('--use-camstyle', action='store_true', default=False)
    parser.add_argument('--general-graph', action='store_true', default=False)
    parser.add_argument('--use-single', action='store_true', default=False)
    parser.add_argument('--only-graph', action='store_true', default=False)
    parser.add_argument('--cos-sim', action='store_true', default=False)
    parser.add_argument('--homo-ap', action='store_true', default=False)

    parser.add_argument('--brightness', type=float, default=0.2)
    parser.add_argument('--saturation', type=float, default=0.0)
    parser.add_argument('--hue', type=float, default=0.0)

    parser.add_argument("--tracklet", action="store_true", default=False)

    parser.add_argument('--describe', type=str)
    args = parser.parse_args()
    main(args)
