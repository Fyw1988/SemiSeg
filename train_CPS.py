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

from dataloaders import utils
from networks.net_factory import net_factory
from dataloaders.la_heart import *
from utils import losses, metrics, ramps, test_patch

parser = argparse.ArgumentParser()
parser.add_argument('--root_path', type=str,
                    default='./', help='Name of Experiment')
parser.add_argument('--dataset_name', type=str,
                    default='Prostate', help='dataset_name')
parser.add_argument('--model_name', type=str,
                    default='CPS', help='model_name')    
parser.add_argument('--model_type', type=str,
                    default='vnet', help='model_type')             
parser.add_argument('--max_iterations', type=int,
                    default=12000, help='maximum epoch number to train')
parser.add_argument('--batch_size', type=int, default=4,
                    help='batch_size per gpu')
parser.add_argument('--deterministic', type=int, default=1,
                    help='whether use deterministic training')
parser.add_argument('--base_lr', type=float, default=0.01,
                    help='segmentation network learning rate')
parser.add_argument('--seed', type=int, default=1337, help='random seed')

# label and unlabel
parser.add_argument('--labeled_bs', type=int, default=2,
                    help='labeled_batch_size per gpu')
parser.add_argument('--labeled_num', type=int, default=32,
                    help='labeled data')

# costs
parser.add_argument('--ema_decay', type=float, default=0.99, help='ema_decay')
parser.add_argument('--consistency_type', type=str,
                    default="mse", help='consistency_type')
parser.add_argument('--consistency', type=float,
                    default=0.1, help='consistency')
parser.add_argument('--consistency_rampup', type=float,
                    default=40.0, help='consistency_rampup')
args = parser.parse_args()

def get_current_consistency_weight(epoch):
    # Consistency ramp-up from https://arxiv.org/abs/1610.02242
    return args.consistency * ramps.sigmoid_rampup(epoch, args.consistency_rampup)


def update_ema_variables(model, ema_model, alpha, global_step):
    # Use the true average until the exponential average is more correct
    alpha = min(1 - 1 / (global_step + 1), alpha)
    for ema_param, param in zip(ema_model.parameters(), model.parameters()):
        ema_param.data.mul_(alpha).add_(1 - alpha, param.data)


def kaiming_normal_init_weight(model):
    for m in model.modules():
        if isinstance(m, nn.Conv3d):
            torch.nn.init.kaiming_normal_(m.weight)
        elif isinstance(m, nn.BatchNorm3d):
            m.weight.data.fill_(1)
            m.bias.data.zero_()
    return model


def xavier_normal_init_weight(model):
    for m in model.modules():
        if isinstance(m, nn.Conv3d):
            torch.nn.init.xavier_normal_(m.weight)
        elif isinstance(m, nn.BatchNorm3d):
            m.weight.data.fill_(1)
            m.bias.data.zero_()
    return model


def train(args, snapshot_path):
    base_lr = args.base_lr
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

    def create_model(ema=False):
        # Network definition
        net = net_factory(net_type=args.model_type, in_chns=1, class_num=num_classes)
        model = net.cuda()
        if ema:
            for param in model.parameters():
                param.detach_()  # 参数冻结
        return model

    net1 = create_model()
    net2 = create_model()
    model1 = kaiming_normal_init_weight(net1)
    model2 = xavier_normal_init_weight(net2)
    model1.train()
    model2.train()

    # Create Dataset
    if args.dataset_name == "LA":
        db_train = LA(base_dir=train_data_path,
                           split='train',
                           transform=transforms.Compose([
                               RandomRotFlip(),
                               RandomCrop(args.patch_size),
                               ToTensor(),
                           ]))
    elif args.dataset_name == "Pancreas":
        db_train = Pancreas(base_dir=train_data_path,
                            split='train',
                            transform=transforms.Compose([
                                RandomCrop(args.patch_size),
                                ToTensor(),
                            ]))
    elif args.dataset_name == "BraTS":
        db_train = BraTS(base_dir=train_data_path,
                            split='train',
                            transform=transforms.Compose([
                                RandomCrop(args.patch_size),
                                ToTensorBra(),
                            ]))
    elif args.dataset_name == "Prostate":
        db_train = Prostate(base_dir=train_data_path,
                            split='train',
                            transform=transforms.Compose([
                                RandomCrop(args.patch_size),
                                ToTensor(),
                            ]))

    def worker_init_fn(worker_id):
        random.seed(args.seed + worker_id)

    labeled_idxs = list(range(0, args.labeled_num))
    unlabeled_idxs = list(range(args.labeled_num, args.max_samples))
    batch_sampler = TwoStreamBatchSampler(
        labeled_idxs, unlabeled_idxs, batch_size, batch_size - args.labeled_bs)

    trainloader = DataLoader(db_train, batch_sampler=batch_sampler,
                             num_workers=4, pin_memory=True, worker_init_fn=worker_init_fn)

    optimizer1 = optim.SGD(model1.parameters(), lr=base_lr,
                           momentum=0.9, weight_decay=0.0001)
    optimizer2 = optim.SGD(model2.parameters(), lr=base_lr,
                           momentum=0.9, weight_decay=0.0001)
    best_performance1 = 0.0
    best_performance2 = 0.0
    iter_num = 0
    ce_loss = CrossEntropyLoss()
    dice_loss = losses.DiceLoss(num_classes)

    writer = SummaryWriter(snapshot_path + '/log')
    logging.info("{} iterations per epoch".format(len(trainloader)))

    max_epoch = max_iterations // len(trainloader) + 1
    iterator = tqdm(range(max_epoch), ncols=70)
    for epoch_num in iterator:
        for i_batch, sampled_batch in enumerate(trainloader):

            volume_batch, label_batch = sampled_batch['image'], sampled_batch['label']
            volume_batch, label_batch = volume_batch.cuda(), label_batch.cuda()

            outputs1 = model1(volume_batch)
            outputs_soft1 = torch.softmax(outputs1, dim=1)

            outputs2 = model2(volume_batch)
            outputs_soft2 = torch.softmax(outputs2, dim=1)
            consistency_weight = get_current_consistency_weight(iter_num // 150)

            loss1 = 0.5 * (ce_loss(outputs1[:args.labeled_bs],
                                   label_batch[:][:args.labeled_bs].long()) + dice_loss(
                outputs_soft1[:args.labeled_bs], label_batch[:args.labeled_bs].unsqueeze(1)))
            loss2 = 0.5 * (ce_loss(outputs2[:args.labeled_bs],
                                   label_batch[:][:args.labeled_bs].long()) + dice_loss(
                outputs_soft2[:args.labeled_bs], label_batch[:args.labeled_bs].unsqueeze(1)))

            pseudo_outputs1 = torch.argmax(outputs_soft1[args.labeled_bs:].detach(), dim=1, keepdim=False)
            pseudo_outputs2 = torch.argmax(outputs_soft2[args.labeled_bs:].detach(), dim=1, keepdim=False)

            pseudo_supervision1 = ce_loss(outputs1[args.labeled_bs:], pseudo_outputs2)
            pseudo_supervision2 = ce_loss(outputs2[args.labeled_bs:], pseudo_outputs1)

            model1_loss = loss1 + consistency_weight * pseudo_supervision1
            model2_loss = loss2 + consistency_weight * pseudo_supervision2

            loss = model1_loss + model2_loss

            optimizer1.zero_grad()
            optimizer2.zero_grad()

            loss.backward()

            optimizer1.step()
            optimizer2.step()

            iter_num = iter_num + 1
            # 更新学习率
            # lr_ = base_lr * (1.0 - iter_num / max_iterations) ** 0.9
            # for param_group1 in optimizer1.param_groups:
            #     param_group1['lr'] = lr_
            # for param_group2 in optimizer2.param_groups:
            #     param_group2['lr'] = lr_

            # writer.add_scalar('lr', lr_, iter_num)
            writer.add_scalar(
                'consistency_weight/consistency_weight', consistency_weight, iter_num)
            writer.add_scalar('loss/model1_loss',
                              model1_loss, iter_num)
            writer.add_scalar('loss/model2_loss',
                              model2_loss, iter_num)
            logging.info(
                'iteration %d : model1 loss : %f model2 loss : %f' % (iter_num, model1_loss.item(), model2_loss.item()))

            # if iter_num % 50 == 0:
            #     image = volume_batch[0, 0:1, :, :, 20:61:10].permute(
            #         3, 0, 1, 2).repeat(1, 3, 1, 1)
            #     grid_image = make_grid(image, 5, normalize=True)
            #     writer.add_image('train/Image', grid_image, iter_num)

            #     image = outputs_soft1[0, 0:1, :, :, 20:61:10].permute(
            #         3, 0, 1, 2).repeat(1, 3, 1, 1)
            #     grid_image = make_grid(image, 5, normalize=False)
            #     writer.add_image('train/Model1_Predicted_label',
            #                      grid_image, iter_num)

            #     image = outputs_soft2[0, 0:1, :, :, 20:61:10].permute(
            #         3, 0, 1, 2).repeat(1, 3, 1, 1)
            #     grid_image = make_grid(image, 5, normalize=False)
            #     writer.add_image('train/Model2_Predicted_label',
            #                      grid_image, iter_num)

            #     image = label_batch[0, :, :, 20:61:10].unsqueeze(
            #         0).permute(3, 0, 1, 2).repeat(1, 3, 1, 1)
            #     grid_image = make_grid(image, 5, normalize=False)
            #     writer.add_image('train/Groundtruth_label',
            #                      grid_image, iter_num)

            # if iter_num % 20 == 0:
            if iter_num > 800 and iter_num % 200 == 0:
                # 测试并保存模型一
                model1.eval()
                if args.dataset_name == "LA":
                    dice_sample1 = test_patch.var_all_case(model1, num_classes=num_classes, patch_size=args.patch_size,
                                                          stride_xy=18, stride_z=4, dataset_name='LA')
                elif args.dataset_name == "Pancreas":
                    dice_sample1 = test_patch.var_all_case(model1, num_classes=num_classes, patch_size=args.patch_size,
                                                          stride_xy=16, stride_z=16, dataset_name='Pancreas')
                elif args.dataset_name == "BraTS":
                    dice_sample1 = test_patch.var_all_case(model1, num_classes=num_classes, patch_size=args.patch_size,
                                                          stride_xy=64, stride_z=64, dataset_name='BraTS')
                elif args.dataset_name == "Prostate":
                    dice_sample1 = test_patch.var_all_case(model1, num_classes=num_classes, patch_size=args.patch_size,
                                                          stride_xy=18, stride_z=4, dataset_name='Prostate')

                if dice_sample1 > best_performance1:
                    best_performance1 = dice_sample1
                    save_mode_path = os.path.join(snapshot_path, 'model1_iter_{}_dice_{}.pth'.format(iter_num, best_performance1))
                    save_best_path = os.path.join(snapshot_path, '{}_best_model1.pth'.format(args.model_name))
                    torch.save(model1.state_dict(), save_mode_path)
                    torch.save(model1.state_dict(), save_best_path)
                    logging.info("save best model1 to {}".format(save_mode_path))
                writer.add_scalar('Var_dice/Dice', dice_sample1, iter_num)
                writer.add_scalar('Var_dice/Best_dice', best_performance1, iter_num)
                model1.train()

                # 测试并保存模型二
                model2.eval()
                if args.dataset_name == "LA":
                    dice_sample2 = test_patch.var_all_case(model2, num_classes=num_classes, patch_size=args.patch_size,
                                                          stride_xy=18, stride_z=4, dataset_name='LA')
                elif args.dataset_name == "Pancreas":
                    dice_sample2 = test_patch.var_all_case(model2, num_classes=num_classes, patch_size=args.patch_size,
                                                          stride_xy=16, stride_z=16, dataset_name='Pancreas')
                elif args.dataset_name == "BraTS":
                    dice_sample2 = test_patch.var_all_case(model2, num_classes=num_classes, patch_size=args.patch_size,
                                                          stride_xy=64, stride_z=64, dataset_name='BraTS')
                elif args.dataset_name == "Prostate":
                    dice_sample2 = test_patch.var_all_case(model2, num_classes=num_classes, patch_size=args.patch_size,
                                                          stride_xy=18, stride_z=4, dataset_name='Prostate')
                if dice_sample2 > best_performance2:
                    best_performance2 = dice_sample2
                    save_mode_path = os.path.join(snapshot_path, 'model2_iter_{}_dice_{}.pth'.format(iter_num, best_performance2))
                    save_best_path = os.path.join(snapshot_path, '{}_best_model2.pth'.format(args.model_name))
                    torch.save(model2.state_dict(), save_mode_path)
                    torch.save(model2.state_dict(), save_best_path)
                    logging.info("save best model2 to {}".format(save_mode_path))
                writer.add_scalar('Var_dice/Dice', dice_sample2, iter_num)
                writer.add_scalar('Var_dice/Best_dice', best_performance2, iter_num)
                model2.train()

            if iter_num % 3000 == 0:
                save_mode_path = os.path.join(
                    snapshot_path, 'model1_iter_' + str(iter_num) + '.pth')
                torch.save(model1.state_dict(), save_mode_path)
                logging.info("save model1 to {}".format(save_mode_path))

                save_mode_path = os.path.join(
                    snapshot_path, 'model2_iter_' + str(iter_num) + '.pth')
                torch.save(model2.state_dict(), save_mode_path)
                logging.info("save model2 to {}".format(save_mode_path))

            if iter_num >= max_iterations:
                break
            time1 = time.time()
        if iter_num >= max_iterations:
            iterator.close()
            break
    writer.close()


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

    snapshot_path = "model/{}_{}/{}".format(args.dataset_name, args.labeled_num, args.model_name)
    if not os.path.exists(snapshot_path):
        os.makedirs(snapshot_path)

    logging.basicConfig(filename=snapshot_path+"/log.txt", level=logging.INFO,
                        format='[%(asctime)s.%(msecs)03d] %(message)s', datefmt='%H:%M:%S')
    logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))
    logging.info(str(args))
    train(args, snapshot_path)