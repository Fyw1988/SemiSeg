import argparse
import logging
import os
import random
import shutil
import sys
import time

import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from tensorboardX import SummaryWriter
from torch.nn import BCEWithLogitsLoss
from torch.nn.modules.loss import CrossEntropyLoss
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.utils import make_grid
from tqdm import tqdm

from dataloaders.la_heart import *

from networks.net_factory import net_factory
from utils import losses, metrics, ramps, test_patch

parser = argparse.ArgumentParser()
parser.add_argument('--root_path', type=str,
                    default='./', help='Name of Experiment')
parser.add_argument('--dataset_name', type=str,
                    default='Prostate', help='dataset_name')
parser.add_argument('--model_name', type=str,
                    default='SUP', help='model_name')
parser.add_argument('--model_type', type=str,
                    default='vnet_trans', help='model_name')
parser.add_argument('--max_iterations', type=int,
                    default=12000, help='maximum epoch number to train')
parser.add_argument('--batch_size', type=int, default=4,
                    help='batch_size per gpu')
parser.add_argument('--deterministic', type=int,  default=1,
                    help='whether use deterministic training')
parser.add_argument('--base_lr', type=float,  default=0.01,
                    help='segmentation network learning rate')
parser.add_argument('--labelnum', type=int, default=160,
                    help='trained samples')

parser.add_argument('--seed', type=int,  default=1337, help='random seed')

args = parser.parse_args()


def train(args, snapshot_path):
    base_lr = args.base_lr
    train_data_path = args.root_path
    batch_size = args.batch_size
    max_iterations = args.max_iterations
    num_classes = 2
    if args.dataset_name == "LA":
        args.patch_size = (112, 112, 80)
        args.root_path = args.root_path + 'datasets/LA'
        args.max_samples = 100
    elif args.dataset_name == "Pancreas":
        args.patch_size = (96, 96, 96)
        args.root_path = args.root_path + 'datasets/Pancreas-CT'
        args.max_samples = 60
    elif args.dataset_name == "BraTS":
        args.patch_size = (96, 96, 96)
        args.root_path = args.root_path + 'datasets/BraTS'
        args.max_samples = 250
    elif args.dataset_name == "Prostate":
        args.patch_size = (112, 112, 80)
        args.root_path = args.root_path + 'datasets/Prostate'
        args.max_samples = 160
    train_data_path = args.root_path

    model = net_factory(net_type=args.model_type, in_chns=1, class_num=num_classes)

    if args.dataset_name == "LA":
        db_train = LA(base_dir=train_data_path,
                           split='train',
                           num=args.labelnum,
                           transform=transforms.Compose([
                               RandomRotFlip(),
                               RandomCrop(args.patch_size),
                               ToTensor(),
                           ]))
    elif args.dataset_name == "Pancreas":
        db_train = Pancreas(base_dir=train_data_path,
                            split='train',
                            num=args.labelnum,
                            transform=transforms.Compose([
                                RandomCrop(args.patch_size),
                                ToTensor(),
                            ]))
    elif args.dataset_name == "BraTS":
        db_train = BraTS(base_dir=train_data_path,
                            split='train',
                            num=args.labelnum,
                            transform=transforms.Compose([
                                RandomCrop(args.patch_size),
                                ToTensorBra(),
                            ]))
    elif args.dataset_name == "Prostate":
        db_train = Prostate(base_dir=train_data_path,
                            split='train',
                            num=args.labelnum,
                            transform=transforms.Compose([
                                RandomCrop(args.patch_size),
                                ToTensor(),
                            ]))
    def worker_init_fn(worker_id):
        random.seed(args.seed + worker_id)
    # import pdb; pdb.set_trace()
    trainloader = DataLoader(db_train, batch_size=batch_size, shuffle=True,
                             num_workers=4, pin_memory=True, worker_init_fn=worker_init_fn)

    model.train()

    optimizer = optim.SGD(model.parameters(), lr=base_lr,
                          momentum=0.9, weight_decay=0.0001)

    writer = SummaryWriter(snapshot_path + '/log')
    logging.info("{} iterations per epoch".format(len(trainloader)))

    iter_num = 0
    max_epoch = max_iterations // len(trainloader) + 1
    best_performance = 0.0
    iterator = tqdm(range(max_epoch), ncols=70)
    for epoch_num in iterator:
        for i_batch, sampled_batch in enumerate(trainloader):

            volume_batch, label_batch = sampled_batch['image'], sampled_batch['label']
            volume_batch, label_batch = volume_batch.cuda(), label_batch.cuda()

            outputs = model(volume_batch)
            outputs_soft = torch.softmax(outputs, dim=1)

            loss_ce = F.cross_entropy(outputs, label_batch)
            loss_dice = losses.dice_loss(outputs_soft[:, 1, :, :, :], label_batch == 1)

            loss = 0.5 * (loss_dice + loss_ce)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            iter_num = iter_num + 1
            # writer.add_scalar('info/lr', lr_, iter_num)
            writer.add_scalar('info/total_loss', loss, iter_num)
            writer.add_scalar('info/loss_ce', loss_ce, iter_num)
            writer.add_scalar('info/loss_dice', loss_dice, iter_num)

            logging.info(
                'iteration %d : loss : %f, loss_ce: %f, loss_dice: %f' %
                (iter_num, loss.item(), loss_ce.item(), loss_dice.item()))
            writer.add_scalar('loss/loss', loss, iter_num)

            # if iter_num % 20 == 0:
            if iter_num > 800 and iter_num % 200 == 0:
                model.eval()
                if args.dataset_name == "LA":
                    dice_sample = test_patch.var_all_case(model, num_classes=num_classes, patch_size=args.patch_size,
                                                          stride_xy=18, stride_z=4, dataset_name='LA')
                elif args.dataset_name == "Pancreas":
                    dice_sample = test_patch.var_all_case(model, num_classes=num_classes, patch_size=args.patch_size,
                                                          stride_xy=16, stride_z=16, dataset_name='Pancreas')
                elif args.dataset_name == "BraTS":
                    dice_sample = test_patch.var_all_case(model, num_classes=num_classes, patch_size=args.patch_size,
                                                          stride_xy=64, stride_z=64, dataset_name='BraTS')
                elif args.dataset_name == "Prostate":
                    dice_sample = test_patch.var_all_case(model, num_classes=num_classes, patch_size=args.patch_size,
                                                          stride_xy=18, stride_z=4, dataset_name='Prostate')
                if dice_sample > best_performance:
                    best_performance = dice_sample
                    save_mode_path = os.path.join(snapshot_path,
                                                  'iter_{}_dice_{}.pth'.format(
                                                      iter_num, best_performance))
                    save_best = os.path.join(snapshot_path,
                                             '{}_best_model.pth'.format(args.model_name))
                    torch.save(model.state_dict(), save_mode_path)
                    torch.save(model.state_dict(), save_best)

                writer.add_scalar('info/Dice',
                                  dice_sample, iter_num)
                writer.add_scalar('info/Best_dice',
                                  best_performance, iter_num)
                model.train()

            if iter_num % 3000 == 0:
                save_mode_path = os.path.join(
                    snapshot_path, 'iter_' + str(iter_num) + '.pth')
                torch.save(model.state_dict(), save_mode_path)
                logging.info("save model to {}".format(save_mode_path))

            if iter_num >= max_iterations:
                break
        if iter_num >= max_iterations:
            iterator.close()
            break
    writer.close()
    return "Training Finished!"


if __name__ == "__main__":
    if not args.deterministic:
        cudnn.benchmark = True
        cudnn.deterministic = False
    else:
        cudnn.benchmark = False
        cudnn.deterministic = True

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)

    snapshot_path = "./model/SUP/{}_{}_{}".format(args.dataset_name, args.model_type, args.labelnum)
    if not os.path.exists(snapshot_path):
        os.makedirs(snapshot_path)

    logging.basicConfig(filename=snapshot_path+"/log.txt", level=logging.INFO,
                        format='[%(asctime)s.%(msecs)03d] %(message)s', datefmt='%H:%M:%S')
    logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))
    logging.info(str(args))
    train(args, snapshot_path)