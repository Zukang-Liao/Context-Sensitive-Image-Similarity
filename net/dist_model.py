import os
import argparse
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn import linear_model
from pandas.plotting import parallel_coordinates
import torch.nn as nn
import torch.utils.model_zoo as model_zoo
import torch.nn.functional as F
import torch
from torchvision import models
import timm

import warnings
warnings.filterwarnings("ignore")

seed = 2
reg_thres = 0.5

dims = (224, 224, 3) # specified by the generator (fixed for this script)
CLASSES_OPTIONS = ['Almost the same', 'Similar / Close idea', 'Related', 'Weakly connected', 'Irrelevant']
CLASSES_DICT = {'Almost the same': 1, 'Similar / Close idea': 2, 'Related': 3, 'Weakly connected': 4, 'Irrelevant': 5}
nb_display = 3 # check dist_annotate.py -- the number of candidates when labelling
nb_ol = 4 # number of ontology used -- fixed
nbcol4labels = 6 # order, cls1, cls2, refid, id1, id2 -- when order=0, cls1<=cls2
nb_coef = 10 # 10 coefficients for the linear model
training_mode = 'feat' # 'easy' or 'feat' # keyword+fc
test_mulref = './dist/mulref.npy'
# test_paths are cross ref test paths
# if a 'path' is args.data_path, skip (do not test on the data that's used for training / non-cross ref testing)
test_paths = {'254' : './dist/dist_254.npy', 
              '1673': './dist/dist_1673.npy', 
              '2723': './dist/dist_2723.npy', 
              '667' : './dist/dist_667.npy', 
              '715' : './dist/dist_715.npy', 
              '389' : './dist/dist_389.npy', 
              '1006': './dist/dist_1006.npy',
              '2352': './dist/dist_2352.npy'}
# test_paths = {'254': test_254, '2723': test_2723, '667': test_667, 'mulref': test_mulref}

def argparser():
    parser = argparse.ArgumentParser()
    # You must specify:
    # parser.add_argument('--data_path', type=str, default='./dist/dist_1673.npy')
    parser.add_argument('--data_path', type=str, default='./dist/dist_2352.npy')
    parser.add_argument('--mulref', action='store_true', default=False)
    parser.add_argument('--ConfigPath', type=str, default="./config.yaml")
    args = parser.parse_args()
    return args


def evaluate(y_true, pred):
    pred = (pred >= reg_thres) * 1
    acc = np.mean(y_true == pred)
    return acc

# linear regression
def train(train_data, nb_train=-1, mode=training_mode, feat_name='place365'):
    print(f"Training Linear Regression + {feat_name} Keyword")
    if nb_train > 0:
        # np.random.seed(seed)
        chosen_ids = np.random.choice(range(len(train_data)), nb_train, replace=False)
        train_data = train_data[chosen_ids]
    if mode == 'easy':
        model = linear_model.LinearRegression().fit(train_data[:, :-nbcol4labels], train_data[:, -nbcol4labels])
    elif mode == 'feat':
        from dist_dataprep import get_extracted_feats
        train_feat = get_extracted_feats(train_data, feat_name=feat_name)
        model = linear_model.LinearRegression().fit(train_feat, train_data[:, -nbcol4labels])
    return model

# test linear regression
def test(model, test_data, mode=training_mode, feat_name='place365'):
    if mode == 'easy':
        test_pred = model.predict(test_data[:, :-nbcol4labels])
    elif mode == 'feat':
        from dist_dataprep import get_extracted_feats
        test_feat = get_extracted_feats(test_data, feat_name=feat_name)
        test_pred = model.predict(test_feat)
    test_acc  = evaluate(test_data[:, -nbcol4labels], test_pred)
    return test_acc

# predict linear regression
def predict(model, test_data, mode=training_mode, feat_name='place365'):
    if mode == 'easy':
        test_pred = model.predict(test_data[:, :-nbcol4labels])
    elif mode == 'feat':
        from dist_dataprep import get_extracted_feats
        test_data = get_extracted_feats(test_data, feat_name=feat_name)
        test_pred = model.predict(test_data)
    return (test_pred >= reg_thres) * 1


def wrong_pred_check(model, test_data):
    from dist_dataprep import get_dbs
    dbs = get_dbs(args)
    test_pred = model.predict(test_data[:, :-nbcol4labels])
    test_pred = (test_pred >= reg_thres) * 1
    test_gt = test_data[:, -nbcol4labels]
    wrong_ids = np.where(test_gt != test_pred)[0]
    labels = test_data[:, -3:] # refid, cand1, cand2
    np.random.shuffle(wrong_ids)
    for i in wrong_ids:
        ref, id1, id2 = labels[i]
        ref, id1, id2 = int(ref), int(id1), int(id2)
        gt = int(test_gt[i])
        pred = test_pred[i]
        fig, axes = plt.subplots(1, 3)
        axes[0].imshow(dbs[ref]['img_data'].resize((200, 200)))
        axes[1].imshow(dbs[id1]['img_data'].resize((200, 200)))
        axes[2].imshow(dbs[id2]['img_data'].resize((200, 200)))
        # gt/pred: 0 -- means id1 is closer than id2 given ref.
        axes[0].set_title(f"Ref: {ref}")
        axes[1].set_title(f"Cand1: {id1}")
        axes[2].set_title(f"Cand2: {id2}")
        for ax in axes:
            ax.set_xticks([])
            ax.set_yticks([])
        fig.suptitle(f"\n\n\n\n\nGt: {gt}, pred: {pred}\n Gt/Pred 0: Cand1 is closer to Cand2, 1: otherwise")
        plt.show()
        import ipdb; ipdb.set_trace()


def visualize_models(models, nb_coef):
    nb_models = len(models)
    modelids = range(nb_models)
    coefs = np.empty([nb_coef, nb_models+1], dtype=object)
    coefs[:, 0] = [f'Param #{x}' for x in range(nb_coef)]
    for mid in modelids:
        model = models[mid]
        coefs[:, mid+1] = model.coef_
    df = pd.DataFrame(coefs, columns=['Name']+[str(x+3) for x in modelids])
    parallel_coordinates(df, 'Name', colormap=plt.get_cmap('tab10'))
    plt.show()


class OrderResNet(nn.Module):
    def __init__(self, block, num_blocks, num_classes=2):
        super(OrderResNet, self).__init__()
        self.in_planes = 64
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d((1,1))
        self.fc = nn.Linear(512*block.expansion*3, num_classes)

    
    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)


    def forward(self, x1, x2, x3):
        pre_outs = []
        for x in [x1, x2, x3]:
            pre_outs.append(self.encode_image(x))
        final = self.fc(torch.concat(pre_outs, dim=1))
        return final, pre_outs

    def encode_image(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        return x.view(x.size(0), -1)

    def forward_head(self, ref_emb, cand1_emb, cand2_emb):
        return self.fc(torch.concat([ref_emb, cand1_emb, cand2_emb], dim=1))

    def load(self, path="resnet_cifar10.pth"):
        tm = torch.load(path, map_location="cpu")        
        self.load_state_dict(tm)
        

class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(in_planes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion*planes)
            )
    
    def forward(self, x):
        t = self.conv1(x)
        out = F.relu(self.bn1(t))
        t = self.conv2(out)
        out = self.bn2(self.conv2(out))
        t = self.shortcut(x)
        out += t
        out = F.relu(out)
        return out


def conv3x3(in_planes, out_planes, stride=1):
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False)


model_urls = {
    'resnet18': 'https://download.pytorch.org/models/resnet18-5c106cde.pth',
    'resnet34': 'https://download.pytorch.org/models/resnet34-333f7ec4.pth',
    'resnet50': 'https://download.pytorch.org/models/resnet50-19c8e357.pth',
    'resnet101': 'https://download.pytorch.org/models/resnet101-5d3b4d8f.pth',
    'resnet152': 'https://download.pytorch.org/models/resnet152-b121ed2d.pth',
}

def load_imgnet_resnet_weight(model, modelarch):
    checkpoint = model_zoo.load_url(model_urls[modelarch])
    model_dict = model.state_dict()

    in_sd = {}
    out_sd = {}
    for k in model_dict:
        if k in checkpoint:
            in_sd[k] = checkpoint[k]
        else:
            if 'shortcut' in k:
                k_model = k.replace('shortcut', 'downsample')
                if k_model in checkpoint:
                    in_sd[k] = checkpoint[k_model]
                    continue
            out_sd[k] = model_dict[k]

    del in_sd['fc.bias']
    del in_sd['fc.weight']
    model_dict.update(in_sd)
    model.load_state_dict(model_dict)
    return model


def resnet18(num_classes, pretrained=True):
    """Constructs a ResNet-18 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = OrderResNet(BasicBlock, [2, 2, 2, 2], num_classes=num_classes)
    if pretrained:
        model = load_imgnet_resnet_weight(model, 'resnet18')
    return model


class RGBBranch(nn.Module):
    """
    Generate Model Architecture
    """
    def __init__(self, arch='ResNet-18', num_classes=2, att=False, triplet=False):
        super(RGBBranch, self).__init__()

        # --------------------------------#
        #          Base Network           #
        # ------------------------------- #
        if arch == 'ResNet-18':
            base = models.resnet.resnet18(pretrained=True)
        elif arch == 'ResNet-50':
            base = models.resnet.resnet50(pretrained=True)
        # --------------------------------#
        #           RGB Branch            #
        # ------------------------------- #
        # First initial block
        self.in_block = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1, return_indices=True)
        )
        # Encoder
        self.encoder1 = base.layer1
        self.encoder2 = base.layer2
        self.encoder3 = base.layer3
        self.encoder4 = base.layer4
        # -------------------------------------#
        #            RGB Classifier            #
        # ------------------------------------ #
        self.att     = att
        self.triplet = triplet
        self.dropout = nn.Dropout(0.3)
        self.avgpool = nn.AvgPool2d(7, stride=1)
        self.attconc = torch.nn.MultiheadAttention(512*3, 4) # 4 heads
        self.classifier  = nn.Sequential(nn.Linear(512*3, num_classes)) # num_classes = 2
        self.lnorm       = torch.nn.LayerNorm((512,)) # not including batch size


    def forward(self, x1, x2, x3):
        pre_outs, feats = [], [] # the difference is for triplet loss only
        for x in [x1, x2, x3]:
            act = self.encode_image(x)
            pre_outs.append(act)
            if self.triplet:
                act = self.lnorm(act)
                # feats.append(nn.functional.normalize(act, dim=1))
            feats.append(act)
        # diff1 = torch.pow(torch.sub(pre_outs[0], pre_outs[1]), 2)
        # diff2 = torch.pow(torch.sub(pre_outs[0], pre_outs[2]), 2)
        # feat  = torch.concat([diff1, diff2], dim=1)
        # final = self.classifier(feat)
        if self.att:
            feat  = torch.concat(feats, dim=1)
            feat  = feat + self.attconc(feat, feat, feat)[0]
            final = self.classifier(feat)
        else:
            final = self.classifier(torch.concat(pre_outs, dim=1))
        # final = self.classifier(torch.concat(pre_outs, dim=1))
        return final, feats


    def encode_image(self, x):
        """
        Netowrk forward
        :param x: RGB Image
        :return: Scene recognition predictions
        """
        batch_size = x.size(0)
        x, pool_indices = self.in_block(x)
        e1  = self.encoder1(x)
        e2  = self.encoder2(e1)
        e3  = self.encoder3(e2)
        e4  = self.encoder4(e3)
        act = self.avgpool(e4)
        act = act.view(batch_size, -1)
        return act


    def forward_head(self, ref_emb, cand1_emb, cand2_emb):
        return self.classifier(torch.concat([ref_emb, cand1_emb, cand2_emb], dim=1))




class ResNetOrder(nn.Module):
    def __init__(self, num_classes, arch, feat_dim, pretrained=True):
        super(ResNetOrder, self).__init__()
        # self.model = models.vgg13_bn(pretrained=True, progress=True)
        self.arch = arch
        self.num_classes = num_classes
        if '50' in arch:
            self.model = models.resnet50()
        elif '34' in arch:
            self.model = models.resnet34()
        self.avgpool = nn.AdaptiveAvgPool2d((1,1))
        self.classifier = nn.Linear(feat_dim*3, num_classes)
    
    def forward(self, x1, x2, x3):
        pre_outs = []
        for x in [x1, x2, x3]:
            pre_outs.append(self.encode_image(x))
        final = self.classifier(torch.concat(pre_outs, dim=1))
        return final, pre_outs

    def encode_image(self, x):
        batch_size = x.size(0)
        for layer in self.model.named_children():
            if layer[0] == 'fc':
                break
            x = layer[1](x)
        return x.view(batch_size, -1)

    def forward_head(self, ref_emb, cand1_emb, cand2_emb):
        return self.classifier(torch.concat([ref_emb, cand1_emb, cand2_emb], dim=1))


class VGGnet(nn.Module):
    def __init__(self, num_classes, arch='13', pretrained=True):
        super(VGGnet, self).__init__()
        # self.model = models.vgg13_bn(pretrained=True, progress=True)
        self.arch = arch
        self.num_classes = num_classes
        if '13' in arch:
            if 'bn' in arch:
                self.model = models.vgg13_bn(pretrained=pretrained, progress=True)
            else:
                self.model = models.vgg13(pretrained=pretrained, progress=True)
        elif '11' in arch:
            if 'bn' in arch:
                self.model = models.vgg11_bn(pretrained=pretrained, progress=True)
            else:
                self.model = models.vgg11(pretrained=pretrained, progress=True)
        elif '16' in arch:
            if 'bn' in arch:
                self.model = models.vgg16_bn(pretrained=pretrained, progress=True)
            else:
                self.model = models.vgg16(pretrained=pretrained, progress=True)
        self.avgpool = nn.AdaptiveAvgPool2d((1,1))
        self.model.classifier = None
        self.classifier = nn.Sequential(nn.Linear(512*3, num_classes))
    
    def forward(self, x1, x2, x3):
        pre_outs = []
        for x in [x1, x2, x3]:
            pre_outs.append(self.encode_image(x))
        final = self.classifier(torch.concat(pre_outs, dim=1))
        return final, pre_outs

    def encode_image(self, x):
        batch_size = x.size(0)
        x = self.model.features(x)
        x = self.avgpool(x)
        return x.view(batch_size, -1)

    def forward_head(self, ref_emb, cand1_emb, cand2_emb):
        return self.classifier(torch.concat([ref_emb, cand1_emb, cand2_emb], dim=1))


class ViT(nn.Module):
    def __init__(self, arch, num_classes, img_size=224, pretrained=True, drop_rate=0.1):
        super(ViT, self).__init__()
        self.arch = arch
        self.num_classes = num_classes
        self.img_size = img_size
        self.drop_rate = drop_rate
        self.pretrained = pretrained
        self.model = timm.create_model(
                arch,
                pretrained=pretrained,
                num_classes=num_classes,
                drop_rate = drop_rate,
                img_size = img_size
            )
        self.model.reset_classifier(num_classes)
        self.classifier = nn.Sequential(nn.Linear(768*3, num_classes))


    def forward(self, x1, x2, x3):
        pre_outs = []
        for x in [x1, x2, x3]:
            pre_outs.append(self.encode_image(x))
        final = self.classifier(torch.concat(pre_outs, dim=1))
        return final, pre_outs

    def encode_image(self, x):
        feat = self.model.forward_features(x)
        return feat[:, 0, :]

    def forward_head(self, ref_emb, cand1_emb, cand2_emb):
        return self.classifier(torch.concat([ref_emb, cand1_emb, cand2_emb], dim=1))



def test_trainnumber(data_path, train_data, test_data, mulref=False, ifplot=True, ifviswrong=True):
    from dist_dataprep import get_data_from_path
    testrange = np.linspace(3, len(train_data), 20)
    if mulref:
        _, mulref_data = get_data_from_path(data_path, incflip=False, mode='mulref')
        mulrefacc = np.zeros([len(test_paths), len(testrange)])
    # import ipdb; ipdb.set_trace()
    # testrange = range(3, len(train_data), 1)
    trainacc, testacc = np.zeros(len(testrange)), np.zeros(len(testrange))
    models = []
    for i, num in enumerate(testrange):
        model = train(train_data, nb_train=int(num))
        trainacc[i] = test(model, train_data)
        testacc[i] = test(model, test_data)
        if mulref:
            for j, ref in enumerate(mulref_data):
                mulrefacc[j][i] = test(model, mulref_data[ref])
        models.append(model)

    if ifplot:
        plt.plot(testrange, trainacc, label='train_acc')
        plt.plot(testrange, testacc, label='test_acc')
        if mulref:
            for j, ref in enumerate(mulref_data):
                plt.plot(testrange, mulrefacc[j], label=f'{ref}_acc')
        plt.legend()
        plt.show()

    bestid = np.argmax(testacc)
    print(f"Best testing acc: {testacc[bestid]}")
    if ifviswrong:
        # check best model
        wrong_pred_check(models[bestid], test_data)

    return models, bestid



if __name__ == "__main__":
    args = argparser()
    train_data, test_data = get_data_from_path(args.data_path, incflip=False)
    models, _ = test_trainnumber(args.data_path, train_data, test_data, mulref=args.mulref, ifviswrong=False)
    # visualize_models(models, nb_coef=nb_coef) # 10 coefficients