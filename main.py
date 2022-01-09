from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import _init_paths

import os
import numpy as np
import torch
import torch.utils.data
from opts import opts
from models.model import create_model, load_model, save_model
from models.data_parallel import DataParallel
from trains.ctdet import CtdetLoss
from logger import Logger
from datasets.dataset_factory import get_dataset
from trains.train_factory import train_factory
from exemplar_create import create_exempalr
from mrc_utils.preprocess import process


def set_requires_grad(model, requires_grad=True):
    for param in model.parameters():
        param.requires_grad = requires_grad


def main(opt):
    torch.manual_seed(opt.seed)
    torch.backends.cudnn.benchmark = not opt.not_cuda_benchmark and not opt.test
    Dataset = get_dataset(opt.dataset, opt.task)
    opt = opts().update_dataset_info_and_set_heads(opt, Dataset)
    #print(opt)
    task = 1
    logger = Logger(opt)

    os.environ['CUDA_VISIBLE_DEVICES'] = opt.gpus_str
    opt.device = torch.device('cuda' if opt.gpus[0] >= 0 else 'cpu')

    print('Preprocessing data...')
    process(opt)

    print('Creating model...')
    model = create_model(opt.arch, opt.heads, opt.head_conv, opt.pretrained_model)
    model1 = create_model(opt.arch, opt.heads, opt.head_conv, opt.pretrained_model)
    optimizer = torch.optim.Adam(model.parameters(), opt.lr)
    start_epoch = 0
    # print(model)
    if opt.load_model != '':
        model, optimizer, start_epoch = load_model(
            model, opt.load_model, optimizer, opt.resume, opt.lr, opt.lr_step)
        model1, _, _ = load_model(
            model1, opt.load_model, optimizer, opt.resume, opt.lr, opt.lr_step)
        if (opt.continual):
            task = -1
    set_requires_grad(model1, requires_grad=False)
    Trainer = train_factory[opt.task]
    trainer = Trainer(opt, model, model1, task, optimizer)
    trainer.set_device(opt.gpus, opt.chunk_sizes, opt.device)
    print('Setting up data...')
    # val_loader = torch.utils.data.DataLoader(
    #     Dataset(opt, 'val'),
    #     batch_size=1,
    #     shuffle=False,
    #     num_workers=1,
    #     pin_memory=True
    # )
    #
    # if opt.test:
    #     _, preds = trainer.val(0, val_loader)
    #     val_loader.dataset.run_eval(preds, opt.save_dir)
    #     return

    train_loader = torch.utils.data.DataLoader(
        Dataset(opt, 'train'),
        batch_size=opt.batch_size,
        shuffle=True,
        num_workers=opt.num_workers,
        pin_memory=True,
        drop_last=False
    )
    old_loader = None
    if os.path.exists(opt.load_exemplar):
        N = len(os.listdir(opt.load_exemplar)) - 1
        N = min(N, opt.batch_size)
    if (task == -1):
        old_loader = torch.utils.data.DataLoader(
            Dataset(opt, 'exemplar'),
            batch_size=N,
            shuffle=True,
            num_workers=opt.num_workers,
            pin_memory=True,
            drop_last=False,
        )

    params = {n: p for n, p in model.named_parameters() if p.requires_grad}
    _means = {}
    for n, p in params.items():
        _means[n] = p.clone().detach()

    print('Starting training...')

    #best = 1e10
    for epoch in range(start_epoch + 1, opt.num_epochs + 1):
        mark = epoch if opt.save_all else 'last'
        log_dict_train, _ = trainer.train(epoch, train_loader, old_loader, _means)
        logger.write('epoch: {} |'.format(epoch))
        for k, v in log_dict_train.items():
            logger.scalar_summary('train_{}'.format(k), v, epoch)
            logger.write('{} {:8f} | '.format(k, v))
        #if epoch % 5 == 0:
        #    save_model(os.path.join(opt.save_dir, 'model_{}.pth'.format(epoch)),
        #               epoch, model, optimizer)
            #with torch.no_grad():
            #     log_dict_val, preds = trainer.val(epoch, val_loader, old_loader, _means)
            # for k, v in log_dict_val.items():
            #     logger.scalar_summary('val_{}'.format(k), v, epoch)
            #     logger.write('{} {:8f} | '.format(k, v))
            # if log_dict_val[opt.metric] < best:
            #     best = log_dict_val[opt.metric]
            #     save_model(os.path.join(opt.save_dir, 'model_best.pth'),
            #                epoch, model)
        # else:
        save_model(os.path.join(opt.save_dir, 'model_last.pth'),
                   epoch, model, optimizer)
        logger.write('\n')
        if epoch in opt.lr_step:
            #save_model(os.path.join(opt.save_dir, 'model_{}.pth'.format(epoch)),
            #           epoch, model, optimizer)
            lr = opt.lr * (0.1 ** (opt.lr_step.index(epoch) + 1))
            print('Drop LR to', lr)
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr
    if opt.continual:
        create_exempalr()
    if not opt.save_debug_files:
        os.system('rm -rf {}'.format(os.path.join('./'+opt.exp_id, 'annotations')))
        os.system('rm -rf {}'.format(os.path.join('./'+opt.exp_id, 'images')))
    logger.close()
    

if __name__ == '__main__':
    opt = opts().parse()
    main(opt)

