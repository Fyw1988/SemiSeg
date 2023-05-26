import os
import sys
from tqdm import tqdm
from tensorboardX import SummaryWriter
import argparse
import logging
import time
import random
import numpy as np

import torch
import torch.optim as optim
from torchvision import transforms
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader
from torchvision.utils import make_grid
from dataloaders import utils

from networks.net_factory import net_factory
from utils import ramps, losses, test_patch
from dataloaders.la_heart import *


parser = argparse.ArgumentParser()
parser.add_argument('--dataset_name', type=str,  default='Prostate', help='dataset_name')
parser.add_argument('--root_path', type=str, default='./', help='Name of Dataset')
parser.add_argument('--model_name', type=str,  default='UAMT', help='model_name')
parser.add_argument('--model_type', type=str, default='vnet', help='model_type')
parser.add_argument('--max_iterations', type=int,  default=12000, help='maximum epoch number to train')
parser.add_argument('--batch_size', type=int, default=4, help='batch_size per gpu')
parser.add_argument('--labeled_bs', type=int, default=2, help='labeled_batch_size per gpu')
parser.add_argument('--labelnum', type=int, default=32, help='trained samples')
parser.add_argument('--base_lr', type=float,  default=0.01, help='maximum epoch number to train')
parser.add_argument('--deterministic', type=int,  default=1, help='whether use deterministic training')
parser.add_argument('--seed', type=int,  default=1337, help='random seed')
parser.add_argument('--gpu', type=str,  default='0', help='GPU to use')
### costs
parser.add_argument('--ema_decay', type=float,  default=0.99, help='ema_decay')
parser.add_argument('--consistency_type', type=str,  default="mse", help='consistency_type')
parser.add_argument('--consistency', type=float,  default=0.1, help='consistency')
parser.add_argument('--consistency_rampup', type=float,  default=40.0, help='consistency_rampup')
args = parser.parse_args()

snapshot_path = "model/{}_{}/{}".format(args.dataset_name, args.labelnum, args.model_name)

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

os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
batch_size = args.batch_size * len(args.gpu.split(','))
max_iterations = args.max_iterations
base_lr = args.base_lr
labeled_bs = args.labeled_bs

if args.deterministic:
    cudnn.benchmark = False
    cudnn.deterministic = True
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)


def get_current_consistency_weight(epoch):
    # Consistency ramp-up from https://arxiv.org/abs/1610.02242
    return args.consistency * ramps.sigmoid_rampup(epoch, args.consistency_rampup)


def update_ema_variables(model, ema_model, alpha, global_step):
    # Use the true average until the exponential average is more correct
    alpha = min(1 - 1 / (global_step + 1), alpha)
    for ema_param, param in zip(ema_model.parameters(), model.parameters()):
        ema_param.data.mul_(alpha).add_(1 - alpha, param.data)

if __name__ == "__main__":
    # make logger file
    if not os.path.exists(snapshot_path):
        os.makedirs(snapshot_path)

    logging.basicConfig(filename=snapshot_path + "/log.txt", level=logging.INFO,
                        format='[%(asctime)s.%(msecs)03d] %(message)s', datefmt='%H:%M:%S')
    logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))
    logging.info(str(args))

    def create_model(ema=False):
        # Network definition
        net = net_factory(net_type=args.model_type, in_chns=1, class_num=num_classes)
        model = net.cuda()
        if ema:
            for param in model.parameters():
                param.detach_()
        return model

    model = create_model()
    ema_model = create_model(ema=True)
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

    labeled_idxs = list(range(args.labelnum))
    unlabeled_idxs = list(range( args.labelnum, args.max_samples))
    batch_sampler = TwoStreamBatchSampler(labeled_idxs, unlabeled_idxs, batch_size, batch_size-labeled_bs)
    def worker_init_fn(worker_id):
        random.seed(args.seed+worker_id)
    trainloader = DataLoader(db_train, batch_sampler=batch_sampler, num_workers=4, pin_memory=True, worker_init_fn=worker_init_fn)

    model.train()
    ema_model.train()
    optimizer = optim.SGD(model.parameters(), lr=base_lr, momentum=0.9, weight_decay=0.0001)

    if args.consistency_type == 'mse':
        consistency_criterion = losses.softmax_mse_loss
    elif args.consistency_type == 'kl':
        consistency_criterion = losses.softmax_kl_loss
    else:
        assert False, args.consistency_type

    writer = SummaryWriter(snapshot_path+'/log')
    logging.info("{} itertations per epoch".format(len(trainloader)))

    iter_num = 0
    best_dice = 0
    max_epoch = max_iterations//len(trainloader)+1
    lr_ = base_lr
    model.train()
    for epoch_num in tqdm(range(max_epoch), ncols=70):
        time1 = time.time()
        for i_batch, sampled_batch in enumerate(trainloader):
            time2 = time.time()
            # print('fetch data cost {}'.format(time2-time1))
            volume_batch, label_batch = sampled_batch['image'], sampled_batch['label']
            volume_batch, label_batch = volume_batch.cuda(), label_batch.cuda()
            unlabeled_volume_batch = volume_batch[labeled_bs:]    # (2, 1, H, W, D)

            noise = torch.clamp(torch.randn_like(unlabeled_volume_batch) * 0.1, -0.2, 0.2)
            ema_inputs = unlabeled_volume_batch + noise
            outputs = model(volume_batch)
            with torch.no_grad():
                ema_output = ema_model(ema_inputs)
            T = 8
            volume_batch_r = unlabeled_volume_batch.repeat(2, 1, 1, 1, 1)
            stride = volume_batch_r.shape[0] // 2
            preds = torch.zeros([stride * T, 2] + list(args.patch_size)).cuda()   # (16, 2, 112, 112, 80)
            for i in range(T//2):
                ema_inputs = volume_batch_r + torch.clamp(torch.randn_like(volume_batch_r) * 0.1, -0.2, 0.2)
                with torch.no_grad():
                    preds[2 * stride * i:2 * stride * (i + 1)] = ema_model(ema_inputs)
            preds = F.softmax(preds, dim=1)
            preds = preds.reshape(T, stride, 2, args.patch_size[0], args.patch_size[1], args.patch_size[2])
            preds = torch.mean(preds, dim=0)  #(batch, 2, 112,112,80)
            uncertainty = -1.0*torch.sum(preds*torch.log(preds + 1e-6), dim=1, keepdim=True) #(batch, 1, 112,112,80)


            ## calculate the loss
            loss_seg = F.cross_entropy(outputs[:labeled_bs], label_batch[:labeled_bs])
            outputs_soft = F.softmax(outputs, dim=1)
            loss_seg_dice = losses.dice_loss(outputs_soft[:labeled_bs, 1, :, :, :], label_batch[:labeled_bs] == 1)
            supervised_loss = 0.5*(loss_seg+loss_seg_dice)

            consistency_weight = get_current_consistency_weight(iter_num//150)
            consistency_dist = consistency_criterion(outputs[labeled_bs:], ema_output) #(batch, 2, 112,112,80)
            threshold = (0.75+0.25*ramps.sigmoid_rampup(iter_num, max_iterations))*np.log(2)
            mask = (uncertainty<threshold).float()
            consistency_dist = torch.sum(mask*consistency_dist)/(2*torch.sum(mask)+1e-16)
            consistency_loss = consistency_weight * consistency_dist
            loss = supervised_loss + consistency_loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            update_ema_variables(model, ema_model, args.ema_decay, iter_num)

            iter_num = iter_num + 1
            writer.add_scalar('uncertainty/mean', uncertainty[0,0].mean(), iter_num)
            writer.add_scalar('uncertainty/max', uncertainty[0,0].max(), iter_num)
            writer.add_scalar('uncertainty/min', uncertainty[0,0].min(), iter_num)
            writer.add_scalar('uncertainty/mask_per', torch.sum(mask)/mask.numel(), iter_num)
            writer.add_scalar('uncertainty/threshold', threshold, iter_num)
            writer.add_scalar('lr', lr_, iter_num)
            writer.add_scalar('loss/loss', loss, iter_num)
            writer.add_scalar('loss/loss_seg', loss_seg, iter_num)
            writer.add_scalar('loss/loss_seg_dice', loss_seg_dice, iter_num)
            writer.add_scalar('train/consistency_loss', consistency_loss, iter_num)
            writer.add_scalar('train/consistency_weight', consistency_weight, iter_num)
            writer.add_scalar('train/consistency_dist', consistency_dist, iter_num)

            logging.info('iteration %d : loss : %f cons_dist: %f, loss_weight: %f' %
                         (iter_num, loss.item(), consistency_dist.item(), consistency_weight))
            
            # if iter_num % 50 == 0:
            #     image = volume_batch[0, 0:1, :, :, 20:61:10].permute(3, 0, 1, 2).repeat(1, 3, 1, 1)
            #     grid_image = make_grid(image, 5, normalize=True)
            #     writer.add_image('train/Image', grid_image, iter_num)

            #     # image = outputs_soft[0, 3:4, :, :, 20:61:10].permute(3, 0, 1, 2).repeat(1, 3, 1, 1)
            #     image = torch.max(outputs_soft[0, :, :, :, 20:61:10], 0)[1].permute(2, 0, 1).data.cpu().numpy()
            #     image = utils.decode_seg_map_sequence(image)
            #     grid_image = make_grid(image, 5, normalize=False)
            #     writer.add_image('train/Predicted_label', grid_image, iter_num)

            #     image = label_batch[0, :, :, 20:61:10].permute(2, 0, 1)
            #     grid_image = make_grid(utils.decode_seg_map_sequence(image.data.cpu().numpy()), 5, normalize=False)
            #     writer.add_image('train/Groundtruth_label', grid_image, iter_num)

            #     image = uncertainty[0, 0:1, :, :, 20:61:10].permute(3, 0, 1, 2).repeat(1, 3, 1, 1)
            #     grid_image = make_grid(image, 5, normalize=True)
            #     writer.add_image('train/uncertainty', grid_image, iter_num)

            #     mask2 = (uncertainty > threshold).float()
            #     image = mask2[0, 0:1, :, :, 20:61:10].permute(3, 0, 1, 2).repeat(1, 3, 1, 1)
            #     grid_image = make_grid(image, 5, normalize=True)
            #     writer.add_image('train/mask', grid_image, iter_num)
            #     #####
            #     image = volume_batch[-1, 0:1, :, :, 20:61:10].permute(3, 0, 1, 2).repeat(1, 3, 1, 1)
            #     grid_image = make_grid(image, 5, normalize=True)
            #     writer.add_image('unlabel/Image', grid_image, iter_num)

            #     # image = outputs_soft[-1, 3:4, :, :, 20:61:10].permute(3, 0, 1, 2).repeat(1, 3, 1, 1)
            #     image = torch.max(outputs_soft[-1, :, :, :, 20:61:10], 0)[1].permute(2, 0, 1).data.cpu().numpy()
            #     image = utils.decode_seg_map_sequence(image)
            #     grid_image = make_grid(image, 5, normalize=False)
            #     writer.add_image('unlabel/Predicted_label', grid_image, iter_num)

            #     image = label_batch[-1, :, :, 20:61:10].permute(2, 0, 1)
            #     grid_image = make_grid(utils.decode_seg_map_sequence(image.data.cpu().numpy()), 5, normalize=False)
            #     writer.add_image('unlabel/Groundtruth_label', grid_image, iter_num)

            ## change lr
            # if iter_num % 2500 == 0:
            #     lr_ = base_lr * 0.1 ** (iter_num // 2500)
            #     for param_group in optimizer.param_groups:
            #         param_group['lr'] = lr_

            if iter_num >= 800 and iter_num % 200 == 0:
            # if iter_num % 20 == 0:
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
                    
                if dice_sample > best_dice:
                    best_dice = dice_sample
                    save_mode_path = os.path.join(snapshot_path,  'iter_{}_dice_{}.pth'.format(iter_num, best_dice))
                    save_best_path = os.path.join(snapshot_path,'{}_best_model.pth'.format(args.model_name))
                    torch.save(model.state_dict(), save_mode_path)
                    torch.save(model.state_dict(), save_best_path)
                    logging.info("save best model to {}".format(save_mode_path))
                writer.add_scalar('Var_dice/Dice', dice_sample, iter_num)
                writer.add_scalar('Var_dice/Best_dice', best_dice, iter_num)
                model.train()

            if iter_num >= max_iterations:
                break
            time1 = time.time()
        if iter_num >= max_iterations:
            break
    save_mode_path = os.path.join(snapshot_path, 'iter_'+str(max_iterations)+'.pth')
    torch.save(model.state_dict(), save_mode_path)
    logging.info("save model to {}".format(save_mode_path))
    writer.close()
