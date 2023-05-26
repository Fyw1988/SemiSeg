import os
import argparse
import torch

from networks.net_factory import net_factory
from utils.test_patch import test_all_case, var_all_case
from glob import glob

parser = argparse.ArgumentParser()
parser.add_argument('--dataset_name', type=str,  default='Prostate', help='dataset_name')
parser.add_argument('--root_path', type=str, default='./', help='Name of Experiment')
parser.add_argument('--model_name', type=str,  default='CCNET', help='model_name')
parser.add_argument('--model_type', type=str,  default='ccnet3d_v1', help='model_type')
parser.add_argument('--save_result', type=str,  default=True, help='save result or not')
parser.add_argument('--gpu', type=str,  default='0', help='GPU to use')
parser.add_argument('--detail', type=int,  default=1, help='print metrics for every samples?')
parser.add_argument('--labelnum', type=int, default=32, help='labeled data')
parser.add_argument('--nms', type=int, default=0, help='apply NMS post-procssing?')

FLAGS = parser.parse_args()
os.environ['CUDA_VISIBLE_DEVICES'] = FLAGS.gpu
# snapshot_path = FLAGS.root_path + "model/SUP/{}_{}_{}".format(FLAGS.dataset_name, FLAGS.model_type, FLAGS.labelnum)
snapshot_path = FLAGS.root_path + "model/{}_{}/{}".format(FLAGS.dataset_name, FLAGS.labelnum, FLAGS.model_name)

test_save_path = FLAGS.root_path + "model/{}_{}/{}_predictions/".format(FLAGS.dataset_name, FLAGS.labelnum, FLAGS.model_name)

num_classes = 2
if FLAGS.dataset_name == "LA":
    FLAGS.patch_size = (112, 112, 80)
    FLAGS.root_path = FLAGS.root_path + 'datasets/LA'
    image_list = glob(FLAGS.root_path + '/Testing Set/*/mri_norm2.h5')
elif FLAGS.dataset_name == "Pancreas":
    FLAGS.patch_size = (96, 96, 96)
    FLAGS.root_path = FLAGS.root_path + 'datasets/Pancreas-CT'
    with open(FLAGS.root_path + '/test.list', 'r') as f:
        image_list = f.readlines()
    image_list = [FLAGS.root_path + "/Pancreas_h5/" + item.replace('\n', '') + "_norm.h5" for item in image_list]
elif FLAGS.dataset_name == "BraTS":
    FLAGS.patch_size = (96, 96, 96)
    FLAGS.root_path = FLAGS.root_path + 'datasets/BraTS'
    with open(FLAGS.root_path + '/val.txt', 'r') as f:
        image_list = f.readlines()
    image_list = ["./datasets/BraTS/data/" + item.replace('\n', '') + ".h5" for item in image_list]
elif FLAGS.dataset_name == "Prostate":
    FLAGS.patch_size = (112, 112, 80)
    FLAGS.root_path = FLAGS.root_path + 'datasets/Prostate'
    image_list = glob(FLAGS.root_path + '/Testing Set/*/mri_norm2.h5')

if FLAGS.save_result:
    if not os.path.exists(test_save_path):
        os.makedirs(test_save_path)
# import pdb; pdb.set_trace()
def test_calculate_metric():
    
    net = net_factory(net_type=FLAGS.model_type, in_chns=1, class_num=num_classes, mode="test")
    save_mode_path = os.path.join(snapshot_path, '{}_best_model.pth'.format(FLAGS.model_name))
    # save_mode_path = os.path.join(snapshot_path, 'SUP_best_model.pth')
    # save_mode_path = '/root/data/fyw/Semi-seg./model/SUP/Prostate_vnet_trans_160/iter_3200_dice_0.8051590572014813.pth'
    net.load_state_dict(torch.load(save_mode_path), strict=False)
    print("init weight from {}".format(save_mode_path))
    net.eval()

    if FLAGS.dataset_name == "LA":
        avg_metric = test_all_case(FLAGS.model_name, 1, net, image_list, num_classes=num_classes,
                        patch_size=FLAGS.patch_size, stride_xy=18, stride_z=4, model_type=FLAGS.model_type,
                        save_result=FLAGS.save_result, test_save_path=test_save_path,
                        metric_detail=FLAGS.detail, nms=FLAGS.nms)
    elif FLAGS.dataset_name == "Pancreas":
        avg_metric = test_all_case(FLAGS.model_name, 1, net, image_list, num_classes=num_classes,
                        patch_size=FLAGS.patch_size, stride_xy=16, stride_z=16, model_type=FLAGS.model_type,
                        save_result=FLAGS.save_result, test_save_path=test_save_path,
                        metric_detail=FLAGS.detail, nms=FLAGS.nms)
    elif FLAGS.dataset_name == "BraTS":
        avg_metric = test_all_case(FLAGS.model_name, 1, net, image_list, num_classes=num_classes,
                        patch_size=FLAGS.patch_size, stride_xy=64, stride_z=64, model_type=FLAGS.model_type,
                        save_result=FLAGS.save_result, test_save_path=test_save_path,
                        metric_detail=FLAGS.detail, nms=FLAGS.nms)
    elif FLAGS.dataset_name == "Prostate":
        avg_metric = test_all_case(FLAGS.model_name, 1, net, image_list, num_classes=num_classes,
                        patch_size=FLAGS.patch_size, stride_xy=18, stride_z=4, model_type=FLAGS.model_type,
                        save_result=FLAGS.save_result, test_save_path=test_save_path,
                        metric_detail=FLAGS.detail, nms=FLAGS.nms)
    return avg_metric


if __name__ == '__main__':
    metric = test_calculate_metric()
    print(metric)