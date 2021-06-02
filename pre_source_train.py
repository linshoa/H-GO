from torch.backends import cudnn
from torch.utils.data import DataLoader
import numpy as np
import argparse
import sys
import os.path as osp


from reid.lib.logging import Logger
from reid.dataloaders.loader import Preprocessor,IterLoader
from reid.lib.serialization import load_checkpoint, save_checkpoint, copy_state_dict

import reid.dataloaders.transforms as T
from reid.dataloaders.dataset import Person
from reid.dataloaders.sampler import RandomMultipleGallerySampler

from reid.lib.lr_scheduler import WarmupMultiStepLR
from reid import models
from reid.models.cls_layer import *

from reid.evaluation.evaluator import Evaluator

from reid.trainer.pre_trainer import PreTrainer

start_epoch = best_mAP = 0


def get_data(opt, source, target, iters=200):

    dataset = Person(opt.data_dir, target, source)
    s_pids = dataset.s_train_pids
    s_train = dataset.source_train
    s_query = dataset.source_query
    s_gallery = dataset.source_gallery

    height, width = opt.height, opt.width
    batch_size, workers = opt.batch_size, opt.workers
    num_instances = opt.num_instances

    normalizer = T.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])

    train_transformer = T.Compose([
             T.Resize((height, width), interpolation=3),
             T.RandomHorizontalFlip(p=0.5),
             T.Pad(10),
             T.RandomCrop((height, width)),
             T.ToTensor(),
             normalizer
         ])

    test_transformer = T.Compose([
             T.Resize((height, width), interpolation=3),
             T.ToTensor(),
             normalizer
         ])

    rmgs_flag = num_instances > 0
    if rmgs_flag:
        sampler = RandomMultipleGallerySampler(s_train, num_instances)
    else:
        sampler = None

    s_train_loader = IterLoader(
                DataLoader(Preprocessor(s_train,
                                        train_transformer),
                            batch_size=batch_size,
                           num_workers=workers,
                           sampler=sampler,
                            shuffle=not rmgs_flag,
                           pin_memory=True,
                           drop_last=True),
                    length=iters)


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

    return s_pids, s_train_loader, s_query_loader, s_gallery_loader, s_query, s_gallery


def main(args):
    torch.multiprocessing.set_sharing_strategy('file_system')
    if args.seed is not None:
        np.random.seed(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        cudnn.deterministic = True
    main_worker(args)


def main_worker(args):
    global start_epoch, best_mAP

    cudnn.benchmark = True
    log_dir = osp.join(working_dir, 'logs/pretrain', args.arch, args.source+"2"+args.target)
    if not args.evaluate:
        # sys.stdout = Logger(osp.join(args.log_dir, 'log.txt'))
        sys.stdout = Logger(osp.join(log_dir, 'log.txt'))
    else:
        # log_dir = osp.dirname(args.resume)
        # sys.stdout = Logger(osp.join(log_dir, 'log_test.txt'))
        sys.stdout = Logger(
            osp.join(log_dir, 'log_test.txt'))
    print("==========\nArgs:{}\n==========".format(args))

    # Create data loaders
    iters = args.iters if (args.iters>0) else None
    s_pids, s_train_loader, s_query_loader, s_gallery_loader, s_query, s_gallery = \
        get_data(args, args.source, args.target, iters)

    t_pids, t_train_loader, t_query_loader, t_gallery_loader, t_query, t_gallery = \
        get_data(args, args.target, args.source, iters)

    # Create model
    device = torch.device('cuda:' + str(args.gpu_ids))
    torch.cuda.set_device(device)
    model = models.create(args.arch, num_features=args.features, dropout=args.dropout, num_classes=s_pids)

    model = nn.DataParallel(model, device_ids=[args.gpu_ids])


    if args.resume:
        checkpoint = load_checkpoint(args.resume)
        copy_state_dict(checkpoint['backbone_dict'], model)

    # Evaluator
    evaluator = Evaluator(model)
    if args.evaluate:
        print('Test target {} : '.format(args.target))
        evaluator.evaluate(t_query_loader, t_gallery_loader, t_query, t_gallery)

        return

    evaluator.evaluate(t_query_loader, t_gallery_loader, t_query, t_gallery)

    params = []
    for key, value in model.named_parameters():
        if not value.requires_grad:
            continue
        params += [{"params": [value], "lr": args.lr, "weight_decay": args.weight_decay}]

    optimizer = torch.optim.Adam(params)
    lr_scheduler = WarmupMultiStepLR(optimizer, args.milestones, gamma=0.1, warmup_factor=0.01, warmup_iters=args.warmup_step)

    # Trainer
    trainer = PreTrainer(model,
                         s_pids, margin=args.margin)

    # Start training
    for epoch in range(start_epoch, args.epochs):
        lr_scheduler.step()
        s_train_loader.new_epoch()
        t_train_loader.new_epoch()

        trainer.train(epoch, s_train_loader, t_train_loader, optimizer,
                    train_iters=len(s_train_loader), print_freq=args.print_freq)

        if ((epoch+1)%args.eval_step==0 or (epoch==args.epochs-1)):

            mAP = evaluator.evaluate(s_query_loader, s_gallery_loader, s_query, s_gallery)

            is_best = mAP > best_mAP
            best_mAP = max(mAP, best_mAP)
            if is_best:
                evaluator.evaluate(t_query_loader, t_gallery_loader, t_query, t_gallery)
                save_checkpoint({
                    'backbone_dict': model.module.state_dict(),
                    'epoch': epoch + 1,
                }, fpath=osp.join(log_dir, 'checkpoint.pth.tar'))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Pre-training on the source domain")
    # gpu ids
    parser.add_argument('--gpu_ids', type=int, default=1)
    # random seed
    parser.add_argument('--seed', type=int, default=1)
    # source
    parser.add_argument('-s', '--source', type=str, default='duke',
                        choices=['market', 'duke', 'msmt17'])
    # target
    parser.add_argument('-t', '--target', type=str, default='market',
                        choices=['market', 'duke', 'msmt17'])
    parser.add_argument('-b', '--batch-size', type=int, default=64)
    parser.add_argument('-j', '--workers', type=int, default=4)
    parser.add_argument('--height', type=int, default=256, help="input height")
    parser.add_argument('--width', type=int, default=128, help="input width")
    parser.add_argument('--num-instances', type=int, default=4,
                        help="each minibatch consist of "
                             "(batch_size // num_instances) identities, and "
                             "each identity has num_instances instances, "
                             "default: 0 (NOT USE)")
    # model
    parser.add_argument('-a', '--arch', type=str, default='resnet50',
                        choices=models.names())
    parser.add_argument('--features', type=int, default=0)
    parser.add_argument('--dropout', type=float, default=0)
    # optimizer
    parser.add_argument('--lr', type=float, default=0.00035,
                        help="learning rate of new parameters, for pretrained ")
    parser.add_argument('--momentum', type=float, default=0.9)
    parser.add_argument('--weight-decay', type=float, default=5e-4)
    parser.add_argument('--warmup-step', type=int, default=10)
    parser.add_argument('--milestones', nargs='+', type=int, default=[40, 70],
                        help='milestones for the learning rate decay')
    # training configs
    parser.add_argument('--resume', type=str, default='', metavar='PATH')
    parser.add_argument('--evaluate', action='store_true', default=False,
                        help="evaluation only")
    parser.add_argument('--eval-step', type=int, default=40)

    parser.add_argument('--epochs', type=int, default=80)
    parser.add_argument('--iters', type=int, default=200)
    parser.add_argument('--print-freq', type=int, default=10)
    parser.add_argument('--margin', type=float, default=0.0, help='margin for the triplet loss with batch hard')
    # path
    working_dir = osp.dirname(osp.abspath(__file__))
    parser.add_argument('--data-dir', type=str, metavar='PATH',
                        default=osp.join(working_dir, 'data'))
    # parser.add_argument('--log-dir', type=str, metavar='PATH',
    #                     default=osp.join(working_dir, 'logs/old/market2msmt17'))

    opt = parser.parse_args()
    main(opt)