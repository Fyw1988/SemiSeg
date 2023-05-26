import argparse
import logging
import os
import random
import sys

import torch.backends.cudnn as cudnn
import torch.optim as optim
from tensorboardX import SummaryWriter
from torch.nn.modules.loss import CrossEntropyLoss
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.utils import make_grid
from tqdm import tqdm
from dataloaders.la_heart import *

from networks.net_factory import net_factory
from utils import losses, ramps, test_patch

parser = argparse.ArgumentParser()
parser.add_argument('--dataset_name', type=str,
                    default='Prostate', help='dataset_name')
parser.add_argument('--root_path', type=str, 
                    default='./', help='Name of Dataset')
parser.add_argument('--model_name', type=str,
                    default='ICT', help='model_name')
parser.add_argument('--model_type', type=str,
                    default='vnet', help='model_type')
parser.add_argument('--max_iterations', type=int,
                    default=12000, help='maximum epoch number to train')
parser.add_argument('--batch_size', type=int, default=4,
                    help='batch_size per gpu')
parser.add_argument('--deterministic', type=int,  default=1,
                    help='whether use deterministic training')
parser.add_argument('--base_lr', type=float,  default=0.01,
                    help='segmentation network learning rate')
parser.add_argument('--seed', type=int,  default=1337, help='random seed')

# label and unlabel
parser.add_argument('--labeled_bs', type=int, default=2,
                    help='labeled_batch_size per gpu')
parser.add_argument('--labeled_num', type=int, default=32,
                    help='labeled data')
parser.add_argument('--ict_alpha', type=int, default=0.2,
                    help='ict_alpha')
# costs
parser.add_argument('--ema_decay', type=float,  default=0.99, help='ema_decay')
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
                param.detach_()
        return model

    model = create_model()
    ema_model = create_model(ema=True)
    # create Dataset
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
        labeled_idxs, unlabeled_idxs, batch_size, batch_size-args.labeled_bs)

    trainloader = DataLoader(db_train, batch_sampler=batch_sampler,
                             num_workers=4, pin_memory=True, worker_init_fn=worker_init_fn)

    model.train()
    ema_model.train()

    optimizer = optim.SGD(model.parameters(), lr=base_lr,
                          momentum=0.9, weight_decay=0.0001)
    ce_loss = CrossEntropyLoss()
    dice_loss = losses.DiceLoss(2)

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
            labeled_volume_batch = volume_batch[:args.labeled_bs]
            unlabeled_volume_batch = volume_batch[args.labeled_bs:]

            # ICT mix factors
            ict_mix_factors = np.random.beta(
                args.ict_alpha, args.ict_alpha, size=(args.labeled_bs//2, 1, 1, 1, 1))
            ict_mix_factors = torch.tensor(
                ict_mix_factors, dtype=torch.float).cuda()
            unlabeled_volume_batch_0 = unlabeled_volume_batch[0:1, ...]
            unlabeled_volume_batch_1 = unlabeled_volume_batch[1:2, ...]

            # Mix images
            batch_ux_mixed = unlabeled_volume_batch_0 * \
                (1.0 - ict_mix_factors) + \
                unlabeled_volume_batch_1 * ict_mix_factors
            input_volume_batch = torch.cat(
                [labeled_volume_batch, batch_ux_mixed], dim=0)
            outputs = model(input_volume_batch)
            outputs_soft = torch.softmax(outputs, dim=1)
            with torch.no_grad():
                ema_output_ux0 = torch.softmax(
                    ema_model(unlabeled_volume_batch_0), dim=1)
                ema_output_ux1 = torch.softmax(
                    ema_model(unlabeled_volume_batch_1), dim=1)
                batch_pred_mixed = ema_output_ux0 * \
                    (1.0 - ict_mix_factors) + ema_output_ux1 * ict_mix_factors

            loss_ce = ce_loss(outputs[:args.labeled_bs],
                              label_batch[:args.labeled_bs][:])
            loss_dice = dice_loss(
                outputs_soft[:args.labeled_bs], label_batch[:args.labeled_bs].unsqueeze(1))
            supervised_loss = 0.5 * (loss_dice + loss_ce)
            consistency_weight = get_current_consistency_weight(iter_num//150)
            consistency_loss = torch.mean(
                (outputs_soft[args.labeled_bs:] - batch_pred_mixed)**2)
            loss = supervised_loss + consistency_weight * consistency_loss
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            update_ema_variables(model, ema_model, args.ema_decay, iter_num)

            # 更新学习率
            # lr_ = base_lr * (1.0 - iter_num / max_iterations) ** 0.9
            # for param_group in optimizer.param_groups:
            #     param_group['lr'] = lr_

            iter_num = iter_num + 1
            # writer.add_scalar('info/lr', lr_, iter_num)
            writer.add_scalar('info/total_loss', loss, iter_num)
            writer.add_scalar('info/loss_ce', loss_ce, iter_num)
            writer.add_scalar('info/loss_dice', loss_dice, iter_num)
            writer.add_scalar('info/consistency_loss',
                              consistency_loss, iter_num)
            writer.add_scalar('info/consistency_weight',
                              consistency_weight, iter_num)

            logging.info(
                'iteration %d : loss : %f, loss_ce: %f, loss_dice: %f' %
                (iter_num, loss.item(), loss_ce.item(), loss_dice.item()))
            writer.add_scalar('loss/loss', loss, iter_num)

            # if iter_num % 20 == 0:
            #     image = volume_batch[0, 0:1, :, :, 20:61:10].permute(
            #         3, 0, 1, 2).repeat(1, 3, 1, 1)
            #     grid_image = make_grid(image, 5, normalize=True)
            #     writer.add_image('train/Image', grid_image, iter_num)

            #     image = outputs_soft[0, 1:2, :, :, 20:61:10].permute(
            #         3, 0, 1, 2).repeat(1, 3, 1, 1)
            #     grid_image = make_grid(image, 5, normalize=False)
            #     writer.add_image('train/Predicted_label',
            #                      grid_image, iter_num)

            #     image = label_batch[0, :, :, 20:61:10].unsqueeze(
            #         0).permute(3, 0, 1, 2).repeat(1, 3, 1, 1)
            #     grid_image = make_grid(image, 5, normalize=False)
            #     writer.add_image('train/Groundtruth_label',
            #                      grid_image, iter_num)

            if iter_num > 0 and iter_num % 200 == 0:
                model.eval()
                if args.dataset_name == "LA":
                    dice_sample = test_patch.var_all_case(model, num_classes=num_classes, patch_size=args.patch_size,
                                                          stride_xy=18, stride_z=4, dataset_name='LA')
                elif args.dataset_name == "Pancreas":
                    dice_sample = test_patch.var_all_case(model, num_classes=num_classes, patch_size=args.patch_size,
                                                          stride_xy=16, stride_z=16, dataset_name='Pancreas')
                elif args.dataset_name == "BraTS":
                    dice_sample = test_patch.var_all_case(model, num_classes=num_classes, patch_size=args.patch_size,
                                                          stride_xy=32, stride_z=32, dataset_name='BraTS')
                elif args.dataset_name == "Prostate":
                    dice_sample = test_patch.var_all_case(model, num_classes=num_classes, patch_size=args.patch_size,
                                                          stride_xy=18, stride_z=4, dataset_name='Prostate')

                if dice_sample > best_performance:
                    best_performance = dice_sample
                    save_mode_path = os.path.join(snapshot_path,
                                                  'iter_{}_dice_{}.pth'.format(
                                                      iter_num, round(best_performance, 4)))
                    save_best = os.path.join(snapshot_path,
                                             '{}_best_model.pth'.format(args.model_name))
                    torch.save(model.state_dict(), save_mode_path)
                    torch.save(model.state_dict(), save_best)

                writer.add_scalar('Var_dice/Dice', dice_sample, iter_num)
                writer.add_scalar('Var_dice/Best_dice', best_performance, iter_num)
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

    snapshot_path = "./model/{}_{}/{}".format(
        args.dataset_name, args.labeled_num, args.model_name)
    if not os.path.exists(snapshot_path):
        os.makedirs(snapshot_path)

    logging.basicConfig(filename=snapshot_path+"/log.txt", level=logging.INFO,
                        format='[%(asctime)s.%(msecs)03d] %(message)s', datefmt='%H:%M:%S')
    logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))
    logging.info(str(args))
    train(args, snapshot_path)