from functools import reduce
from operator import add
import torch.nn as nn
import torch.nn.functional as F
import torch

# for cvnet
_IN_STAGE_DS = {50: (3, 4, 6, 3), 101: (3, 4, 23, 3), 152: (3, 8, 36, 3)}
TRANS_FUN = "bottleneck_transform"
NUM_GROUPS = 1
WIDTH_PER_GROUP = 64
STRIDE_1X1 = False
BN_EPS = 1e-5
BN_MOM = 0.1
RELU_INPLACE = True
def get_trans_fun(name):
    """Retrieves the transformation function by name."""
    trans_funs = {
        "basic_transform": BasicTransform,
        "bottleneck_transform": BottleneckTransform,
    }
    err_str = "Transformation function '{}' not supported"
    assert name in trans_funs.keys(), err_str.format(name)
    return trans_funs[name]


class GlobalHead(nn.Module):
    def __init__(self, w_in, nc, pp=3):
        super(GlobalHead, self).__init__()
        self.pool = GeneralizedMeanPoolingP(norm=pp)
        self.fc = nn.Linear(w_in, nc, bias=True)

    def forward(self, x):
        x = self.pool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x

class GeneralizedMeanPooling(nn.Module):
    r"""Applies a 2D power-average adaptive pooling over an input signal composed of several input planes.
    The function computed is: :math:`f(X) = pow(sum(pow(X, p)), 1/p)`
        - At p = infinity, one gets Max Pooling
        - At p = 1, one gets Average Pooling
    The output is of size H x W, for any input size.
    The number of output features is equal to the number of input planes.
    Args:
        output_size: the target output size of the image of the form H x W.
                     Can be a tuple (H, W) or a single H for a square image H x H
                     H and W can be either a ``int``, or ``None`` which means the size will
                     be the same as that of the input.
    """

    def __init__(self, norm, output_size=1, eps=1e-6):
        super(GeneralizedMeanPooling, self).__init__()
        assert norm > 0
        self.p = float(norm)
        self.output_size = output_size
        self.eps = eps

    def forward(self, x):
        x = x.clamp(min=self.eps).pow(self.p)
        return torch.nn.functional.adaptive_avg_pool2d(x, self.output_size).pow(1. / self.p)

    def __repr__(self):
        return self.__class__.__name__ + '(' \
               + str(self.p) + ', ' \
               + 'output_size=' + str(self.output_size) + ')'


class GeneralizedMeanPoolingP(GeneralizedMeanPooling):
    """ Same, but norm is trainable
    """

    def __init__(self, norm=3, output_size=1, eps=1e-6):
        super(GeneralizedMeanPoolingP, self).__init__(norm, output_size, eps)
        self.p = nn.Parameter(torch.ones(1) * norm)


class BasicTransform(nn.Module):
    """Basic transformation: 3x3, BN, ReLU, 3x3, BN."""

    def __init__(self, w_in, w_out, stride, w_b=None, num_gs=1):
        err_str = "Basic transform does not support w_b and num_gs options"
        assert w_b is None and num_gs == 1, err_str
        super(BasicTransform, self).__init__()
        self.a = nn.Conv2d(w_in, w_out, 3, stride=stride, padding=1, bias=False)
        self.a_bn = nn.BatchNorm2d(w_out, eps=BN_EPS, momentum=BN_MOM)
        self.a_relu = nn.ReLU(inplace=RELU_INPLACE)
        self.b = nn.Conv2d(w_out, w_out, 3, stride=1, padding=1, bias=False)
        self.b_bn = nn.BatchNorm2d(w_out, eps=BN_EPS, momentum=BN_MOM)
        self.b_bn.final_bn = True

    def forward(self, x):
        for layer in self.children():
            x = layer(x)
        return x


class BottleneckTransform(nn.Module):
    """Bottleneck transformation: 1x1, BN, ReLU, 3x3, BN, ReLU, 1x1, BN."""

    def __init__(self, w_in, w_out, stride, w_b, num_gs):
        super(BottleneckTransform, self).__init__()
        # MSRA -> stride=2 is on 1x1; TH/C2 -> stride=2 is on 3x3
        (s1, s3) = (stride, 1) if STRIDE_1X1 else (1, stride)
        self.a = nn.Conv2d(w_in, w_b, 1, stride=s1, padding=0, bias=False)
        self.a_bn = nn.BatchNorm2d(w_b, eps=BN_EPS, momentum=BN_MOM)
        self.a_relu = nn.ReLU(inplace=RELU_INPLACE)
        self.b = nn.Conv2d(w_b, w_b, 3, stride=s3, padding=1, groups=num_gs, bias=False)
        self.b_bn = nn.BatchNorm2d(w_b, eps=BN_EPS, momentum=BN_MOM)
        self.b_relu = nn.ReLU(inplace=RELU_INPLACE)
        self.c = nn.Conv2d(w_b, w_out, 1, stride=1, padding=0, bias=False)
        self.c_bn = nn.BatchNorm2d(w_out, eps=BN_EPS, momentum=BN_MOM)
        self.c_bn.final_bn = True

    def forward(self, x):
        for layer in self.children():
            x = layer(x)
        return x

class ResBlock(nn.Module):
    """Residual block: x + F(x)."""

    def __init__(self, w_in, w_out, stride, trans_fun, w_b=None, num_gs=1):
        super(ResBlock, self).__init__()
        # Use skip connection with projection if shape changes
        self.proj_block = (w_in != w_out) or (stride != 1)
        if self.proj_block:
            self.proj = nn.Conv2d(w_in, w_out, 1, stride=stride, padding=0, bias=False)
            self.bn = nn.BatchNorm2d(w_out, eps=BN_EPS, momentum=BN_MOM)
        self.f = trans_fun(w_in, w_out, stride, w_b, num_gs)
        self.relu = nn.ReLU(RELU_INPLACE)

    def forward(self, x):
        if self.proj_block:
            x = self.bn(self.proj(x)) + self.f(x)
        else:
            x = x + self.f(x)
        x = self.relu(x)
        return x

class ResStage(nn.Module):
    """Stage of ResNet."""

    def __init__(self, w_in, w_out, stride, d, w_b=None, num_gs=1):
        super(ResStage, self).__init__()
        for i in range(d):
            b_stride = stride if i == 0 else 1
            b_w_in = w_in if i == 0 else w_out
            trans_fun = get_trans_fun(TRANS_FUN)
            res_block = ResBlock(b_w_in, w_out, b_stride, trans_fun, w_b, num_gs)
            self.add_module("b{}".format(i + 1), res_block)

    def forward(self, x):
        for block in self.children():
            x = block(x)
        return x

class ResStage_basetransform(nn.Module):
    """Stage of ResNet."""

    def __init__(self, w_in, w_out, stride, d, w_b=None, num_gs=1):
        super(ResStage, self).__init__()
        for i in range(d):
            b_stride = stride if i == 0 else 1
            b_w_in = w_in if i == 0 else w_out
            trans_fun = "basic_transform"
            res_block = ResBlock(b_w_in, w_out, b_stride, trans_fun, w_b, num_gs)
            self.add_module("b{}".format(i + 1), res_block)

    def forward(self, x):
        for block in self.children():
            x = block(x)
        return x

class ResStemIN(nn.Module):
    """ResNet stem for ImageNet: 7x7, BN, ReLU, MaxPool."""

    def __init__(self, w_in, w_out):
        super(ResStemIN, self).__init__()
        self.conv = nn.Conv2d(w_in, w_out, 7, stride=2, padding=3, bias=False)
        self.bn = nn.BatchNorm2d(w_out, eps=BN_EPS, momentum=BN_MOM)
        self.relu = nn.ReLU(RELU_INPLACE)
        self.pool = nn.MaxPool2d(3, stride=2, padding=1)

    def forward(self, x):
        for layer in self.children():
            x = layer(x)
        return x

class ResNet(nn.Module):
    """ResNet model."""

    def __init__(self, RESNET_DEPTH, REDUCTION_DIM):
        super(ResNet, self).__init__()
        self.RESNET_DEPTH = RESNET_DEPTH
        self.REDUCTION_DIM = REDUCTION_DIM
        self._construct()

    def _construct(self):
        g, gw = NUM_GROUPS, WIDTH_PER_GROUP
        (d1, d2, d3, d4) = _IN_STAGE_DS[self.RESNET_DEPTH]
        w_b = gw * g
        self.stem = ResStemIN(3, 64)
        self.s1 = ResStage(64, 256, stride=1, d=d1, w_b=w_b, num_gs=g)
        self.s2 = ResStage(256, 512, stride=2, d=d2, w_b=w_b * 2, num_gs=g)
        self.s3 = ResStage(512, 1024, stride=2, d=d3, w_b=w_b * 4, num_gs=g)
        self.s4 = ResStage(1024, 2048, stride=2, d=d4, w_b=w_b * 8, num_gs=g)
        self.head = GlobalHead(2048, nc=self.REDUCTION_DIM)
        
    def forward(self, x):
        x = self.stem(x)
        x1 = self.s1(x)
        x2 = self.s2(x1)
        x3 = self.s3(x2)
        x4 = self.s4(x3)
        x4_p = self.head.pool(x4)
        x4_p = x4_p.view(x4_p.size(0), -1)
        x = self.head.fc(x4_p)
        return x, x3

def extract_feat_res_pycls(img, backbone, feat_ids, bottleneck_ids, lids):
    r""" Extract intermediate features from ResNet"""
    feats = []

    # Layer 0
    feat = backbone.stem(img)

    # Layer 1-4
    for hid, (bid, lid) in enumerate(zip(bottleneck_ids, lids)):
        res = feat
        feat = backbone.__getattr__('s%d' % lid).__getattr__('b%d' % (bid+1)).f.forward(feat)

        if bid == 0:
            res = backbone.__getattr__('s%d' % lid).__getattr__('b%d' % (bid+1)).proj.forward(res)
            res = backbone.__getattr__('s%d' % lid).__getattr__('b%d' % (bid+1)).bn.forward(res)
        feat += res

        if hid + 1 in feat_ids:
            feats.append(feat.clone())

        feat = backbone.__getattr__('s%d' % lid).__getattr__('b%d' % (bid+1)).relu.forward(feat)

    return feats

class CVNet_Rerank(nn.Module):

    def __init__(self, RESNET_DEPTH, REDUCTION_DIM):
        super(CVNet_Rerank, self).__init__()

        self.encoder_q = ResNet(RESNET_DEPTH, REDUCTION_DIM)
        self.encoder_q.eval()

        self.scales = [0.25, 0.5, 1.0]
        self.num_scales = len(self.scales)

        feat_dim_l3 = 1024
        self.channel_compressed = 256

        self.softmax = nn.Softmax(dim=1)
        self.extract_feats = extract_feat_res_pycls

        if RESNET_DEPTH == 50:
            nbottlenecks = [3, 4, 6, 3]
            self.feat_ids = [13]
        elif RESNET_DEPTH == 101:
            nbottlenecks = [3, 4, 23, 3]
            self.feat_ids = [30]
        else:
            raise Exception('Unavailable RESNET_DEPTH %s' % RESNET_DEPTH)

        self.bottleneck_ids = reduce(add, list(map(lambda x: list(range(x)), nbottlenecks)))
        self.lids = reduce(add, [[i + 1] * x for i, x in enumerate(nbottlenecks)])

        self.conv2ds = nn.ModuleList([nn.Conv2d(feat_dim_l3, 256, kernel_size=3, padding=1, bias=False) for _ in self.scales])

        self.cv_learner = CVLearner([self.num_scales*self.num_scales, self.num_scales*self.num_scales, self.num_scales*self.num_scales])

    def forward(self, query_img, key_img):
        with torch.no_grad():
            query_feats = self.extract_feats(query_img, self.encoder_q, self.feat_ids, self.bottleneck_ids, self.lids)
            key_feats = self.extract_feats(key_img, self.encoder_q, self.feat_ids, self.bottleneck_ids, self.lids)
            corr_qk = Correlation.build_crossscale_correlation(query_feats[0], key_feats[0], self.scales, self.conv2ds)
            logits_qk = self.cv_learner(corr_qk)
            score = self.softmax(logits_qk)[:,1]
        return score

    def extract_global_descriptor(self, im_q):
        # compute query features
        q = self.encoder_q(im_q)[0]
        q = nn.functional.normalize(q, dim=1)
        return q

    def extract_featuremap(self, img):
        with torch.no_grad():
            feats = self.extract_feats(img, self.encoder_q, self.feat_ids, self.bottleneck_ids, self.lids)
        return feats

    def extract_score_with_featuremap(self, query_feats, key_feats):
        with torch.no_grad():
            corr_qk = Correlation.build_crossscale_correlation(query_feats[0], key_feats[0], self.scales, self.conv2ds)
            logits_qk = self.cv_learner(corr_qk)
            score = self.softmax(logits_qk)[0][1]
        return score


class Conv4d(nn.Module):
    r""" CenterPivot 4D conv"""
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, bias=True):
        super(Conv4d, self).__init__()

        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size[:2], stride=stride[:2],
                               bias=bias, padding=padding[:2])
        self.conv2 = nn.Conv2d(in_channels, out_channels, kernel_size[2:], stride=stride[2:],
                               bias=bias, padding=padding[2:])

        self.stride34 = stride[2:]
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.idx_initialized = False
        self.idx_initialized_2 = False

    def prune(self, ct):
        bsz, ch, ha, wa, hb, wb = ct.size()
        idxh = torch.arange(start=0, end=hb, step=self.stride[2:][0], device=ct.device)
        idxw = torch.arange(start=0, end=wb, step=self.stride[2:][1], device=ct.device)
        self.len_h = len(idxh)
        self.len_w = len(idxw)
        self.idx = (idxw.repeat(self.len_h, 1) + idxh.repeat(self.len_w, 1).t() * wb).view(-1)
        self.idx_initialized = True
        ct_pruned = ct.view(bsz, ch, ha, wa, -1).index_select(4, self.idx).view(bsz, ch, ha, wa, self.len_h, self.len_w)

        return ct_pruned

    def prune_out2(self, ct):
        bsz, ch, ha, wa, hb, wb = ct.size()
        idxh = torch.arange(start=0, end=ha, step=self.stride[:2][0], device=ct.device)
        idxw = torch.arange(start=0, end=wa, step=self.stride[:2][1], device=ct.device)
        self.len_h = len(idxh)
        self.len_w = len(idxw)
        self.idx = (idxw.repeat(self.len_h, 1) + idxh.repeat(self.len_w, 1).t() * wa).view(-1)
        self.idx_initialized_2 = True
        ct_pruned = ct.view(bsz, ch, -1, hb, wb).permute(0,1,3,4,2).index_select(4, self.idx).permute(0,1,4,2,3).view(bsz, ch, self.len_h, self.len_w, hb, wb)

        return ct_pruned

    def forward(self, x):
        if self.stride[2:][-1] > 1:
            out1 = self.prune(x)
        else:
            out1 = x
        bsz, inch, ha, wa, hb, wb = out1.size()
        out1 = out1.permute(0, 4, 5, 1, 2, 3).contiguous().view(-1, inch, ha, wa)
        out1 = self.conv1(out1)
        outch, o_ha, o_wa = out1.size(-3), out1.size(-2), out1.size(-1)
        out1 = out1.view(bsz, hb, wb, outch, o_ha, o_wa).permute(0, 3, 4, 5, 1, 2).contiguous()


        if self.stride[:2][-1] > 1:
            out2 = self.prune_out2(x)
        else:
            out2 = x
        bsz, inch, ha, wa, hb, wb = out2.size()
        out2 = out2.permute(0, 2, 3, 1, 4, 5).contiguous().view(-1, inch, hb, wb)
        out2 = self.conv2(out2)
        outch, o_hb, o_wb = out2.size(-3), out2.size(-2), out2.size(-1)
        out2 = out2.view(bsz, ha, wa, outch, o_hb, o_wb).permute(0, 3, 1, 2, 4, 5).contiguous()
        if out1.size()[-2:] != out2.size()[-2:] and self.padding[-2:] == (0, 0):
            out1 = out1.view(bsz, outch, o_ha, o_wa, -1).sum(dim=-1)
            out2 = out2.squeeze()
        y = out1 + out2
        return y


class CVLearner(nn.Module):
    def __init__(self, inch):
        super(CVLearner, self).__init__()

        def make_building_block(in_channel, out_channels, kernel_sizes, query_strides, key_strides, group=4):
            assert len(out_channels) == len(kernel_sizes) == len(key_strides)

            building_block_layers = []
            for idx, (outch, ksz, query_stride, key_stride) in enumerate(zip(out_channels, kernel_sizes, query_strides, key_strides)):
                inch = in_channel if idx == 0 else out_channels[idx - 1]
                ksz4d = (ksz,) * 4
                str4d = (query_stride,) * 2 + (key_stride,) * 2
                pad4d = (ksz // 2,) * 4

                building_block_layers.append(Conv4d(inch, outch, ksz4d, str4d, pad4d))
                building_block_layers.append(nn.GroupNorm(group, outch))
                building_block_layers.append(nn.ReLU(inplace=True))

            return nn.Sequential(*building_block_layers)

        outch1, outch2, outch3, outch4 = 16, 32, 64, 128

        self.block1 = make_building_block(inch[1], [outch1], [5], [2], [2])
        self.block2 = make_building_block(outch1, [outch1, outch2], [3, 3], [1, 2], [1, 2])
        self.block3 = make_building_block(outch2, [outch2, outch2, outch3], [3, 3, 3], [1, 1, 2], [1, 1, 2])
        self.block4 = make_building_block(outch3, [outch3, outch3, outch4], [3, 3, 3], [1, 1, 1], [1, 1, 1])

        self.mlp = nn.Sequential(nn.Linear(outch4, outch4), nn.ReLU(), nn.Linear(outch4, 2))
        
    def interpolate_support_dims(self, hypercorr, spatial_size=None):
        bsz, ch, ha, wa, hb, wb = hypercorr.size()
        hypercorr = hypercorr.permute(0, 4, 5, 1, 2, 3).contiguous().view(bsz * hb * wb, ch, ha, wa)
        hypercorr = F.interpolate(hypercorr, spatial_size, mode='bilinear', align_corners=True)
        o_hb, o_wb = spatial_size
        hypercorr = hypercorr.view(bsz, hb, wb, ch, o_hb, o_wb).permute(0, 3, 4, 5, 1, 2).contiguous()
        return hypercorr

    def interpolate_query_dims(self, hypercorr, spatial_size=None):
        bsz, ch, ha, wa, hb, wb = hypercorr.size()
        hypercorr = hypercorr.permute(0, 2, 3, 1, 4, 5).contiguous().view(bsz * ha * wa, ch, hb, wb)
        hypercorr = F.interpolate(hypercorr, spatial_size, mode='bilinear', align_corners=True)
        o_ha, o_wa = spatial_size
        hypercorr = hypercorr.view(bsz, ha, wa, ch, o_ha, o_wa).permute(0, 3, 1, 2, 4, 5).contiguous()
        return hypercorr

    def forward(self, corr):
        # Encode correlation from each layer (Squeezing building blocks)
        out_block1 = self.block1(corr)
        out_block2 = self.block2(out_block1)
        out_block3 = self.block3(out_block2)
        out_block4 = self.block4(out_block3)
        
        # Predict logits with the encoded 4D-tensor
        bsz, ch, _, _, _, _= out_block4.size()
        out_block4_pooled = out_block4.view(bsz, ch, -1).mean(-1)
        logits = self.mlp(out_block4_pooled).squeeze(-1).squeeze(-1)
        
        return logits

