import torch
from torch import nn
# from networks.guide_filter import GuidedFilter
import torch.nn.functional as F


class PST(nn.Module):
    """ The implement of PST block.

    Args:
        dim (int): Number of feature channels
        depth (int): Depths of this stage.
        num_heads (int): Number of attention head.
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim. Default: 4.
        qkv_bias (bool, optional): If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float | None, optional): Override default qk scale of head_dim ** -0.5 if set.
        drop (float, optional): Dropout rate. Default: 0.0
        attn_drop (float, optional): Attention dropout rate. Default: 0.0
        drop_path (float | tuple[float], optional): Stochastic depth rate. Default: 0.0
        norm_layer (nn.Module, optional): Normalization layer. Default: nn.LayerNorm
        downsample (nn.Module | None, optional): Downsample layer at the end of the layer. Default: None
    """

    def __init__(self,
                 dim,
                 num_heads=8,
                 depth=2,
                 mlp_ratio=4.,
                 qkv_bias=False,
                 qk_scale=None,
                 drop=0.,
                 attn_drop=0.,
                 drop_path=0.,
                 norm_layer=nn.LayerNorm,
                 use_checkpoint=False,
                 shift_type='psm'):
        super().__init__()
        self.depth = depth
        self.use_checkpoint = use_checkpoint
        self.shift_type = shift_type

        # build blocks
        self.blocks = nn.ModuleList([
            AttnBlock(
                dim=dim,
                num_heads=num_heads,
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                qk_scale=qk_scale,
                drop=drop,
                attn_drop=attn_drop,
                drop_path=drop_path[i] if isinstance(drop_path, list) else drop_path,
                norm_layer=norm_layer,
                shift=True,
                shift_type='tsm' if (i % 2 == 0 and self.shift_type == 'psm') or self.shift_type == 'tsm' else 'psm',
            )
            for i in range(depth)])

    def forward(self, x):
        """ Forward function.
        Args:
            x: Input feature, tensor size (B, C, W, H, D).
        """
        for blk in self.blocks:
            x = blk(x)   # (B, C, W, H, D)

        return x


class AttnBlock(nn.Module):
    """ Self-Attention Block.

    Args:
        dim (int): Number of input channels.
        num_heads (int): Number of attention heads.
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim.
        qkv_bias (bool, optional): If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float | None, optional): Override default qk scale of head_dim ** -0.5 if set.
        drop (float, optional): Dropout rate. Default: 0.0
        attn_drop (float, optional): Attention dropout rate. Default: 0.0
        drop_path (float, optional): Stochastic depth rate. Default: 0.0
        act_layer (nn.Module, optional): Activation layer. Default: nn.GELU
        norm_layer (nn.Module, optional): Normalization layer.  Default: nn.LayerNorm
    """

    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=True, qk_scale=None, drop=0., attn_drop=0., drop_path=0.,
                 act_layer=nn.GELU, norm_layer=nn.LayerNorm, shift=False, shift_type='psm'):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.mlp_ratio = mlp_ratio
        self.shift = shift
        self.shift_type = shift_type

        self.norm1 = norm_layer(dim)
        self.attn = SelfAttention3D(dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale,
                                attn_drop=attn_drop, proj_drop=drop, shift=self.shift, shift_type=self.shift_type)

        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

    def forward_part1(self, x):
        B, C, W, H, D = x.shape
        x = x.permute(0, 4, 3, 2, 1)  # B, D, H, W, C = x.shape
        x = self.norm1(x)
        x = x.flatten(1, 3)
        # multi-head attention 3D
        x_attn = self.attn(x, shape=(W, H, D), frame_len=D)  # B*nW, Wd*Wh*Ww, C
        # reshape-->(B, D, H, W, C)
        x_attn = x_attn.view(B, D, H, W, C)

        return x_attn

    def forward_part2(self, x):
        return self.drop_path(self.mlp(self.norm2(x)))

    def forward(self, x):
        shortcut = x.permute(0, 4, 3, 2, 1)  # (B, D, H, W, C)
        x = self.forward_part1(x)
        # import pdb; pdb.set_trace()
        x = shortcut + self.drop_path(x)
        x = x + self.forward_part2(x)
        # (B, D, H, W, C)-->(B, C, W, H, D)
        x = x.permute(0, 4, 3, 2, 1)

        return x


class SelfAttention3D(nn.Module):
    """ 3D Multi-head self attention (3D-MSA) module.
    Args:
        dim (int): Number of input channels.
        num_heads (int): Number of attention heads.
        qkv_bias (bool, optional):  If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float | None, optional): Override default qk scale of head_dim ** -0.5 if set
        attn_drop (float, optional): Dropout ratio of attention weight. Default: 0.0
        proj_drop (float, optional): Dropout ratio of output. Default: 0.0
    """

    def __init__(self, dim, num_heads, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0.,
                 shift=False, shift_type='psm'):

        super().__init__()
        self.dim = dim
        ## for bayershift
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5
        self.shift = shift
        self.shift_type = shift_type

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        self.softmax = nn.Softmax(dim=-1)

        if self.shift and self.shift_type == 'psm':
            self.shift_op = PatchShift(False, 1)
            self.shift_op_back = PatchShift(True, 1)
        elif self.shift and self.shift_type == 'tsm':
            self.shift_op = TemporalShift(8)

    def forward(self, x, shape, frame_len=10):
        """ Forward function.
        Args:
            x: input features with shape of (num_windows*B, N, C)
        """
        B, N, C = x.shape  # c = C // n_h
        if self.shift:
            x = x.flatten(2).view(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
            # (B, n_h, D*H*W, C//n_h)
            x = self.shift_op(x, frame_len, shape)
            x = x.permute(0, 2, 1, 3).reshape(B, N, C)
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)  # (3,B,nH,N,C // self.num_heads)
        q, k, v = qkv[0], qkv[1], qkv[2]  # B, nH, N, c

        q = q * self.scale
        attn = q @ k.transpose(-2, -1)  # B, nH, N, N

        attn = self.softmax(attn)
        attn = self.attn_drop(attn)

        if self.shift and self.shift_type == 'psm':
            x = self.shift_op_back(attn @ v, frame_len, shape).transpose(1, 2).reshape(B, N, C)
        else:
            x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)  # (B, N, C)
        return x


class CoAttention3D(nn.Module):
    """ 3D Multi-head self attention (3D-MSA) module.
    Args:
        dim (int): Number of input channels.
        num_heads (int): Number of attention heads.
        qkv_bias (bool, optional):  If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float | None, optional): Override default qk scale of head_dim ** -0.5 if set
        attn_drop (float, optional): Dropout ratio of attention weight. Default: 0.0
        proj_drop (float, optional): Dropout ratio of output. Default: 0.0
    """

    def __init__(self, dim, num_heads, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0.,
                 shift=False, shift_type='psm'):

        super().__init__()
        self.dim = dim
        ## for bayershift
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5
        self.shift = shift
        self.shift_type = shift_type

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
        self.linear_e = nn.Linear(dim, dim, bias = False)

        self.softmax = nn.Softmax(dim=-1)

        if self.shift and self.shift_type == 'psm':
            self.shift_op = PatchShift(False, 1)
            self.shift_op_back = PatchShift(True, 1)
        elif self.shift and self.shift_type == 'tsm':
            self.shift_op = TemporalShift(8)

    def forward(self, x, shape, frame_len=10):
        """ Forward function.
        Args:
            x: input features with shape of (num_windows*B, N, C)
        """
        # x.shape = (B, D, H, W, C)
        B, D, H, W, C = x.shape
        N = D*W*H
        if self.shift:
            x = x.permute(0, 4, 3, 2, 1)  # (B, C, W, H, D)
            x = x.flatten(2).view(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)  # (B, n_h, N, c)
            x = self.shift_op(x, frame_len, shape)
            # x = x.permute(0, 2, 1, 3).reshape(B, N, C)   # (B, N, C)
            x = x.permute(0, 2, 1, 3).reshape(B, N, C).view(B, D, H, W, C).permute(0, 4, 3, 2, 1)  # (B, C, W, H, D)

        # 分块 (112, 112, 80)-->nx(14, 14, 10)
        x_patchs = [x[:, :, 14*i:14*(i+1), 14*i:14*(i+1), 10*i:10*(i+1)] for i in range(8)]
        for k in x_patchs:
            for q in x_patchs:
                output, _ = self.generate_attention(k, q)

        # qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)  # (3,B,nH,N,C // self.num_heads)
        # q, k, v = qkv[0], qkv[1], qkv[2]  # B, nH, N, c
        #
        # q = q * self.scale
        # attn = q @ k.transpose(-2, -1)  # B, nH, N, N
        #
        # attn = self.softmax(attn)
        # attn = self.attn_drop(attn)
        #
        # if self.shift and self.shift_type == 'psm':
        #     x = self.shift_op_back(attn @ v, frame_len, shape).transpose(1, 2).reshape(B, N, C)
        # else:
        #     x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        # x = self.proj(x)
        # x = self.proj_drop(x)  # (B, N, C)
        return x

    def generate_attention(self, exemplar, query):
        # input_shape (B, C, W, H, D)
        fea_size = query.size()[2:]
        exemplar_flat = exemplar.view(-1, self.dim, fea_size[0] * fea_size[1] * fea_size[2])  # (B,C,H*W*D)
        query_flat = query.view(-1, self.dim, fea_size[0] * fea_size[1] * fea_size[2])  # (B,C,H*W*D)
        exemplar_t = torch.transpose(exemplar_flat, 1, 2).contiguous()  # (B, H*W*D, C)
        exemplar_corr = self.linear_e(exemplar_t)
        A = torch.bmm(exemplar_corr, query_flat)  # (B, H*W*D, H*W*D)
        B = F.softmax(torch.transpose(A, 1, 2), dim=1)
        query_att = torch.bmm(exemplar_flat, A).contiguous()  # 注意我们这个地方要不要用交互以及Residual的结构
        exemplar_att = torch.bmm(query_flat, B).contiguous()

        input1_att = exemplar_att.view(-1, self.dim, fea_size[0], fea_size[1], fea_size[2])
        input2_att = query_att.view(-1, self.dim, fea_size[0], fea_size[1], fea_size[2])
        # input1_mask = self.gate(input1_att)
        # input2_mask = self.gate(input2_att)
        # input1_mask = self.gate_s(input1_mask)
        # input2_mask = self.gate_s(input2_mask)
        # input1_att = input1_att * input1_mask
        # input2_att = input2_att * input2_mask

        return input1_att, input2_att


class Mlp(nn.Module):
    """ Multilayer perceptron."""

    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


def drop_path(x, drop_prob: float = 0., training: bool = False, scale_by_keep: bool = True):
    """Drop paths (Stochastic Depth) per sample (when applied in main path of residual blocks).

    This is the same as the DropConnect impl I created for EfficientNet, etc networks, however,
    the original name is misleading as 'Drop Connect' is a different form of dropout in a separate paper...
    See discussion: https://github.com/tensorflow/tpu/issues/494#issuecomment-532968956 ... I've opted for
    changing the layer and argument names to 'drop path' rather than mix DropConnect as a layer name and use
    'survival rate' as the argument.

    """
    if drop_prob == 0. or not training:
        return x
    keep_prob = 1 - drop_prob
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)  # work with diff dim tensors, not just 2D ConvNets
    random_tensor = x.new_empty(shape).bernoulli_(keep_prob)
    if keep_prob > 0.0 and scale_by_keep:
        random_tensor.div_(keep_prob)
    return x * random_tensor


class DropPath(nn.Module):
    """Drop paths (Stochastic Depth) per sample  (when applied in main path of residual blocks).
    """
    def __init__(self, drop_prob: float = 0., scale_by_keep: bool = True):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob
        self.scale_by_keep = scale_by_keep

    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training, self.scale_by_keep)

    def extra_repr(self):
        return f'drop_prob={round(self.drop_prob,3):0.3f}'


class PatchShift(nn.Module):
    def __init__(self, inv=False, ratio=1):
        super(PatchShift, self).__init__()
        self.inv = inv
        self.ratio = ratio
        if inv:
            print('=> Using inverse PatchShift, head_num: {}, ratio {}, tps'.format(8, ratio))
        else:
            print('=> Using bayershift, head_num: {}, ratio {}, tps'.format(8, ratio))

    def forward(self, x, frame_len, shape):
        x = self.shift(x, inv=self.inv, ratio=self.ratio,
                       frame_len=frame_len, shape=shape)
        return x  # self.net(x)

    @staticmethod
    def shift(x, shape, inv=False, ratio=0.5, frame_len=10):
        B, num_heads, N, c = x.size()
        W, H, D = shape
        fold = int(num_heads * ratio)
        feat = x.contiguous()
        feat = feat.view(B, frame_len, -1, num_heads, H, W, c)
        out = feat.clone()
        multiplier = 1
        stride = 1
        if inv:
            multiplier = -1

        ## Pattern C
        out[:, :, :, :fold, 0::3, 0::3, :] = torch.roll(feat[:, :, :, :fold, 0::3, 0::3, :],
                                                        shifts=-4 * multiplier * stride, dims=1)
        out[:, :, :, :fold, 0::3, 1::3, :] = torch.roll(feat[:, :, :, :fold, 0::3, 1::3, :],
                                                        shifts=multiplier * stride, dims=1)
        out[:, :, :, :fold, 1::3, 0::3, :] = torch.roll(feat[:, :, :, :fold, 1::3, 0::3, :],
                                                        shifts=-multiplier * stride, dims=1)
        out[:, :, :, :fold, 0::3, 2::3, :] = torch.roll(feat[:, :, :, :fold, 0::3, 2::3, :],
                                                        shifts=2 * multiplier * stride, dims=1)
        out[:, :, :, :fold, 2::3, 0::3, :] = torch.roll(feat[:, :, :, :fold, 2::3, 0::3, :],
                                                        shifts=-2 * multiplier * stride, dims=1)
        out[:, :, :, :fold, 1::3, 2::3, :] = torch.roll(feat[:, :, :, :fold, 1::3, 2::3, :],
                                                        shifts=3 * multiplier * stride, dims=1)
        out[:, :, :, :fold, 2::3, 1::3, :] = torch.roll(feat[:, :, :, :fold, 2::3, 1::3, :],
                                                        shifts=-3 * multiplier * stride, dims=1)
        out[:, :, :, :fold, 2::3, 2::3, :] = torch.roll(feat[:, :, :, :fold, 2::3, 2::3, :],
                                                        shifts=4 * multiplier * stride, dims=1)

        out = out.view(B, num_heads, N, c)
        return out


class TemporalShift(nn.Module):
    def __init__(self, n_div=8):
        super(TemporalShift, self).__init__()
        self.fold_div = n_div
        print('=> Using channel shift, fold_div: {}'.format(self.fold_div))

    def forward(self, x, frame_len, shape):
        x = self.shift(x, fold_div=self.fold_div, shape=shape)
        return x

    @staticmethod
    def shift(x, shape, fold_div=8):
        W, H, D = shape
        B, num_heads, N, c = x.size()
        fold = c // fold_div
        feat = x.contiguous()
        feat = feat.view(B, D, num_heads, H*W, c)
        out = feat.clone()
        # 1/4的通道使用了shift操作
        out[:, 1:, :, :, :fold] = feat[:, :-1, :, :, :fold]  # shift left
        out[:, :-1, :, :, fold:2 * fold] = feat[:, 1:, :, :, fold:2 * fold]  # shift right

        out = out.view(B, num_heads, N, c)

        return out


class ConvBlock(nn.Module):
    def __init__(self, n_stages, n_filters_in, n_filters_out, normalization='none'):
        super(ConvBlock, self).__init__()

        ops = []
        for i in range(n_stages):
            if i==0:
                input_channel = n_filters_in
            else:
                input_channel = n_filters_out

            ops.append(nn.Conv3d(input_channel, n_filters_out, 3, padding=1))
            if normalization == 'batchnorm':
                ops.append(nn.BatchNorm3d(n_filters_out))
            elif normalization == 'groupnorm':
                ops.append(nn.GroupNorm(num_groups=16, num_channels=n_filters_out))
            elif normalization == 'instancenorm':
                ops.append(nn.InstanceNorm3d(n_filters_out))
            elif normalization != 'none':
                assert False
            ops.append(nn.ReLU(inplace=True))

        self.conv = nn.Sequential(*ops)

    def forward(self, x):
        x = self.conv(x)
        return x


class ResidualConvBlock(nn.Module):
    def __init__(self, n_stages, n_filters_in, n_filters_out, normalization='none'):
        super(ResidualConvBlock, self).__init__()

        ops = []
        for i in range(n_stages):
            if i == 0:
                input_channel = n_filters_in
            else:
                input_channel = n_filters_out

            ops.append(nn.Conv3d(input_channel, n_filters_out, 3, padding=1))
            if normalization == 'batchnorm':
                ops.append(nn.BatchNorm3d(n_filters_out))
            elif normalization == 'groupnorm':
                ops.append(nn.GroupNorm(num_groups=16, num_channels=n_filters_out))
            elif normalization == 'instancenorm':
                ops.append(nn.InstanceNorm3d(n_filters_out))
            elif normalization != 'none':
                assert False

            if i != n_stages-1:
                ops.append(nn.ReLU(inplace=True))

        self.conv = nn.Sequential(*ops)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = (self.conv(x) + x)
        x = self.relu(x)
        return x


class DownsamplingConvBlock(nn.Module):
    def __init__(self, n_filters_in, n_filters_out, stride=2, normalization='none'):
        super(DownsamplingConvBlock, self).__init__()

        ops = []
        if normalization != 'none':
            ops.append(nn.Conv3d(n_filters_in, n_filters_out, stride, padding=0, stride=stride))
            if normalization == 'batchnorm':
                ops.append(nn.BatchNorm3d(n_filters_out))
            elif normalization == 'groupnorm':
                ops.append(nn.GroupNorm(num_groups=16, num_channels=n_filters_out))
            elif normalization == 'instancenorm':
                ops.append(nn.InstanceNorm3d(n_filters_out))
            else:
                assert False
        else:
            ops.append(nn.Conv3d(n_filters_in, n_filters_out, stride, padding=0, stride=stride))

        ops.append(nn.ReLU(inplace=True))

        self.conv = nn.Sequential(*ops)

    def forward(self, x):
        x = self.conv(x)
        return x


class UpsamplingDeconvBlock(nn.Module):
    def __init__(self, n_filters_in, n_filters_out, stride=2, normalization='none'):
        super(UpsamplingDeconvBlock, self).__init__()

        ops = []
        if normalization != 'none':
            ops.append(nn.ConvTranspose3d(n_filters_in, n_filters_out, stride, padding=0, stride=stride))
            if normalization == 'batchnorm':
                ops.append(nn.BatchNorm3d(n_filters_out))
            elif normalization == 'groupnorm':
                ops.append(nn.GroupNorm(num_groups=16, num_channels=n_filters_out))
            elif normalization == 'instancenorm':
                ops.append(nn.InstanceNorm3d(n_filters_out))
            else:
                assert False
        else:
            ops.append(nn.ConvTranspose3d(n_filters_in, n_filters_out, stride, padding=0, stride=stride))

        ops.append(nn.ReLU(inplace=True))

        self.conv = nn.Sequential(*ops)

    def forward(self, x):
        x = self.conv(x)
        return x


class Upsampling(nn.Module):
    # upsample + conv3d + norm + relu
    def __init__(self, n_filters_in, n_filters_out, stride=2, normalization='none'):
        super(Upsampling, self).__init__()

        ops = []
        ops.append(nn.Upsample(scale_factor=stride, mode='trilinear',align_corners=False))
        ops.append(nn.Conv3d(n_filters_in, n_filters_out, kernel_size=3, padding=1))
        if normalization == 'batchnorm':
            ops.append(nn.BatchNorm3d(n_filters_out))
        elif normalization == 'groupnorm':
            ops.append(nn.GroupNorm(num_groups=16, num_channels=n_filters_out))
        elif normalization == 'instancenorm':
            ops.append(nn.InstanceNorm3d(n_filters_out))
        elif normalization != 'none':
            assert False
        ops.append(nn.ReLU(inplace=True))

        self.conv = nn.Sequential(*ops)

    def forward(self, x):
        x = self.conv(x)
        return x


class TransVNet(nn.Module):
    def __init__(self, n_channels=3, n_classes=2, n_filters=16, normalization='none', has_dropout=False,
                 dgf=False, dgf_r=4, dgf_eps=1e-2):
        super(TransVNet, self).__init__()
        self.has_dropout = has_dropout

        self.block_one = ConvBlock(1, n_channels, n_filters, normalization=normalization)
        self.block_one_dw = DownsamplingConvBlock(n_filters, 2 * n_filters, normalization=normalization)
        # self.trans_one = PST(n_filters)

        self.block_two = ConvBlock(2, n_filters * 2, n_filters * 2, normalization=normalization)
        self.block_two_dw = DownsamplingConvBlock(n_filters * 2, n_filters * 4, normalization=normalization)
        # self.trans_two = PST(2 * n_filters)

        self.block_three = ConvBlock(3, n_filters * 4, n_filters * 4, normalization=normalization)
        self.block_three_dw = DownsamplingConvBlock(n_filters * 4, n_filters * 8, normalization=normalization)
        # self.trans_three = PST(4 * n_filters)

        self.block_four = ConvBlock(3, n_filters * 8, n_filters * 8, normalization=normalization)
        self.block_four_dw = DownsamplingConvBlock(n_filters * 8, n_filters * 16, normalization=normalization)
        self.trans_four = PST(8 * n_filters)
        self.trans_five = PST(8 * n_filters)
        self.trans_six = PST(8 * n_filters)
        # self.trans_seven = PST(8 * n_filters)

        self.block_five = ConvBlock(3, n_filters * 16, n_filters * 16, normalization=normalization)
        self.block_five_up = UpsamplingDeconvBlock(n_filters * 16, n_filters * 8, normalization=normalization)

        self.block_six = ConvBlock(3, n_filters * 8, n_filters * 8, normalization=normalization)
        self.block_six_up = UpsamplingDeconvBlock(n_filters * 8, n_filters * 4, normalization=normalization)

        self.block_seven = ConvBlock(3, n_filters * 4, n_filters * 4, normalization=normalization)
        self.block_seven_up = UpsamplingDeconvBlock(n_filters * 4, n_filters * 2, normalization=normalization)

        self.block_eight = ConvBlock(2, n_filters * 2, n_filters * 2, normalization=normalization)
        self.block_eight_up = UpsamplingDeconvBlock(n_filters * 2, n_filters, normalization=normalization)

        self.block_nine = ConvBlock(1, n_filters, n_filters, normalization=normalization)
        self.out_conv = nn.Conv3d(n_filters, n_classes, 1, padding=0)

        self.dropout = nn.Dropout3d(p=0.5, inplace=False)

        # DGF
        # self.dgf = dgf
        # if self.dgf:
        #     self.guided_map_conv1 = nn.Conv3d(1, 64, 1)
        #     self.guided_map_relu1 = nn.ReLU(inplace=True)
        #     self.guided_map_conv2 = nn.Conv3d(64, 2, 1)
        #     self.guided_filter = GuidedFilter(dgf_r, dgf_eps)
        # self.__init_weight()

    def encoder(self, input):
        x1 = self.block_one(input)
        # x1_dw_trans = self.trans_one(x1)
        x1_dw = self.block_one_dw(x1)

        x2 = self.block_two(x1_dw)
        # x2_dw_trans = self.trans_two(x2)
        x2_dw = self.block_two_dw(x2)

        x3 = self.block_three(x2_dw)
        # x3_dw_trans = self.trans_three(x3)
        x3_dw = self.block_three_dw(x3)

        x4 = self.block_four(x3_dw)
        # x4_dw_trans = self.trans_seven(self.trans_six(self.trans_five(self.trans_four(x4))))
        x4_dw_trans = self.trans_six(self.trans_five(self.trans_four(x4)))
        # x4_dw_trans = self.trans_five(self.trans_four(x4))
        # x4_dw_trans = self.trans_four(x4)
        x4_dw = self.block_four_dw(x4)

        x5 = self.block_five(x4_dw)
        # x5 = F.dropout3d(x5, p=0.5, training=True)
        if self.has_dropout:
            x5 = self.dropout(x5)

        # res = [x1_dw_trans, x2_dw_trans, x3_dw_trans, x4_dw_trans, x5]
        res = [x1, x2, x3, x4_dw_trans, x5]
        # res = [x1, x2, x3, x4, x5]

        return res

    def decoder(self, features):
        x1 = features[0]
        x2 = features[1]
        x3 = features[2]
        x4 = features[3]
        x5 = features[4]

        x5_up = self.block_five_up(x5)
        x5_up = x5_up + x4

        x6 = self.block_six(x5_up)
        x6_up = self.block_six_up(x6)
        x6_up = x6_up + x3

        x7 = self.block_seven(x6_up)
        x7_up = self.block_seven_up(x7)
        x7_up = x7_up + x2

        x8 = self.block_eight(x7_up)
        x8_up = self.block_eight_up(x8)
        x8_up = x8_up + x1
        x9 = self.block_nine(x8_up)
        # x9 = F.dropout3d(x9, p=0.5, training=True)
        if self.has_dropout:
            x9 = self.dropout(x9)
        out = self.out_conv(x9)
        return out

    def forward(self, input, turnoff_drop=False):
        if turnoff_drop:
            has_dropout = self.has_dropout
            self.has_dropout = False
        features = self.encoder(input)
        out = self.decoder(features)
        if turnoff_drop:
            self.has_dropout = has_dropout

        # GFN
        # if self.dgf:
        #     g = self.guided_map_relu1(self.guided_map_conv1(input))
        #     g = self.guided_map_conv2(g)

        #     out = self.guided_filter(g, out)

        return out


if __name__ == '__main__':
    # compute FLOPS & PARAMETERS
    from thop import profile
    from thop import clever_format
    model = TransVNet(n_channels=1, n_classes=2)
    input = torch.randn(1, 1, 112, 112, 80)
    flops, params = profile(model, inputs=(input,))
    print(flops, params)
    macs, params = clever_format([flops, params], "%.3f")
    print(macs, params)
    print("VNet have {} paramerters in total".format(sum(x.numel() for x in model.parameters())))