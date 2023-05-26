from networks.vnet import VNet
from networks.unet_urpc import unet_3D_dv_semi
from networks.vnet_sdf import VNet as vnet_sdf
from networks.vnet_trans import TransVNet

from networks.unet_MCNet import UNet, MCNet2d_v1, MCNet2d_v2, MCNet2d_v3
from networks.vnet_MCNet import MCNet3d_v1, MCNet3d_v2
from networks.vnet_CCNet import CCNet3d_V1


def net_factory(net_type="unet", in_chns=1, class_num=4, mode = "train"):
    # global net
    if net_type == "mcnet2d_v1":
        net = MCNet2d_v1(in_chns=in_chns, class_num=class_num).cuda()
    elif net_type == "mcnet2d_v2":
        net = MCNet2d_v2(in_chns=in_chns, class_num=class_num).cuda()
    elif net_type == "mcnet2d_v3":
        net = MCNet2d_v3(in_chns=in_chns, class_num=class_num).cuda()

    elif net_type == "vnet" and mode == "train":
        net = VNet(n_channels=in_chns, n_classes=class_num, normalization='batchnorm', has_dropout=True).cuda()
    elif net_type == "vnet" and mode == "test":
        net = VNet(n_channels=in_chns, n_classes=class_num, normalization='batchnorm', has_dropout=False).cuda()
    
    elif net_type == "mcnet3d_v1" and mode == "train":
        net = MCNet3d_v1(n_channels=in_chns, n_classes=class_num, normalization='batchnorm', has_dropout=True).cuda()
    elif net_type == "mcnet3d_v1" and mode == "test":
        net = MCNet3d_v1(n_channels=in_chns, n_classes=class_num, normalization='batchnorm', has_dropout=False).cuda()
    
    elif net_type == "mcnet3d_v2" and mode == "train":
        net = MCNet3d_v2(n_channels=in_chns, n_classes=class_num, normalization='batchnorm', has_dropout=True).cuda()
    elif net_type == "mcnet3d_v2" and mode == "test":
        net = MCNet3d_v2(n_channels=in_chns, n_classes=class_num, normalization='batchnorm', has_dropout=False).cuda()
    
    elif net_type == "ccnet3d_v1" and mode == "train":
        net = CCNet3d_V1(n_channels=in_chns, n_classes=class_num, normalization='batchnorm', has_dropout=True).cuda()
    elif net_type == "ccnet3d_v1" and mode == "test":
        net = CCNet3d_V1(n_channels=in_chns, n_classes=class_num, normalization='batchnorm', has_dropout=False).cuda()

    elif net_type == "unet_3D_dv_semi" and mode == "train":
        net = unet_3D_dv_semi(in_channels=in_chns, n_classes=class_num).cuda()
    elif net_type == "unet_3D_dv_semi" and mode == "test":
        net = unet_3D_dv_semi(in_channels=in_chns, n_classes=class_num).cuda()

    elif net_type == "vnet_sdf" and mode == "train":
        net = vnet_sdf(n_channels=in_chns, n_classes=class_num, normalization='batchnorm', has_dropout=True).cuda()
    elif net_type == "vnet_sdf" and mode == "test":
        net = vnet_sdf(n_channels=in_chns, n_classes=class_num, normalization='batchnorm', has_dropout=False).cuda()

    elif net_type == "vnet_trans" and mode == 'train':
        net = TransVNet(n_channels=in_chns, n_classes=class_num,
                    normalization='batchnorm', has_dropout=True).cuda()
    elif net_type == "vnet_trans" and mode == 'test':
        net = TransVNet(n_channels=in_chns, n_classes=class_num,
                   normalization='batchnorm', has_dropout=False).cuda()

    return net