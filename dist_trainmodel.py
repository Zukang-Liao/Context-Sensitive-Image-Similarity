import os
import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import torch.optim as optim
import torch.nn as nn
import numpy as np
from dist_dataprep import get_dbs, get_mixing_data, get_mixingdata_byref
from triplet_loss import OrderTripletLoss, compute_squared_distance
from dist_loadmodel import get_dlmodel


modelarch       = 'resnet18'
label_dir       = './metadata/npy_labels'
save_model_dir  = './saved_models' # save CS models to
resize          = (224, 224)
batch_size      = 8
lr              = 0.00001 # 0.000001 # 0.0001 for lora vit, 0.00001 for resnet18
device          = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
refids          = (2723,) # trained on which datase # refids = (1673, 667, 254)
nb_epochs       = 25
seed            = 2
combinedloss    = True
combine_alpha   = 0.1
softmax         = torch.nn.Softmax()

def cosine_distances(a, b):
    return 1-np.dot(a, b)/(np.linalg.norm(a)*np.linalg.norm(b))

def get_datapaths():
    data_paths, pthname = [], ''
    for i in refids:
        data_paths.append(os.path.join(label_dir, f'dist_{i}.npy'))
        pthname += f'{i}_'
    return data_paths, pthname[:-1]+'.pth'

def get_acc(preds, labels):
    return np.mean(preds == labels)

class orderDb(Dataset):
    def __init__(self, datapaths, resize, split='', transform=None, splitbyref=False, incflip=True, bgdb_split='val'):
        self.datapaths = datapaths
        if splitbyref:
            assert len(datapaths) == 1, 'When split by ref, only one data path is used'
            self.metadata = get_mixingdata_byref(datapaths[0], split, incflip) # not shuffled
        else:
            train, test = get_mixing_data(datapaths, split, incflip, shuffle=False) # not shuffled
            if split != 'test':
                self.metadata = train
                if split != 'all':
                    np.random.shuffle(self.metadata) # shuffled
            else:
                self.metadata = test # not shuffled
        self.resize    = resize
        self.dbs       = get_dbs(split=bgdb_split) # 'val' or 'train'
        self.incflip   = incflip
        if transform is not None:
            self.transform = transform
        else:
            if split == 'train':
                self.transform = transforms.Compose([
                                 transforms.ToTensor(),
                                 transforms.RandomResizedCrop(size=(224, 224)),
                                 transforms.RandomHorizontalFlip(p=0.5),
                                 transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
                                 ])
            else:
                self.transform = transforms.Compose([
                                 transforms.ToTensor(),
                                 transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
                                 ])

    def __len__(self):
        return len(self.metadata)

    def __getitem__(self, idx):
        refid, candid1, candid2 = self.metadata[idx][-3:]
        cls1, cls2 = self.metadata[idx][-5:-3]
        ref   = self.transform(self.dbs[int(refid)]['img_data'].resize(self.resize))
        cand1 = self.transform(self.dbs[int(candid1)]['img_data'].resize(self.resize))
        cand2 = self.transform(self.dbs[int(candid2)]['img_data'].resize(self.resize))
        ytrue = self.metadata[idx, 0] # order, cls1, cls2, refid, id1, id2
        return {'ref': ref, 'cand1': cand1, 'cand2': cand2, 'label': ytrue, 'refid': refid, 'candid1': candid1, 'candid2': candid2,
                'refpath': self.dbs[int(refid)]['img_path'], 'c1path': self.dbs[int(candid1)]['img_path'], 'c2path': self.dbs[int(candid2)]['img_path'],
                'cls1': cls1, 'cls2': cls2}


def dataGen(paths, batch_size, resize, split, incflip, shuffle=False):
    data    = orderDb(paths, resize, split, incflip=incflip)
    dataGen = DataLoader(data, batch_size, shuffle=shuffle)
    return dataGen


def get_loss(criterion, outs, feats, labels, cls1, cls2):
    if combinedloss:
        return criterion[0](outs, labels) + combine_alpha * criterion[1](feats, labels, cls1, cls2)
    else:
        return criterion(outs, labels)


def get_preds(outs, feats, use_embs=False, dist_type='cosine'):
    if use_embs:
        # cand1 <= cand2 when label is 0
        if dist_type == 'l2':
            dref1 = compute_squared_distance(feats[0], feats[1])
            dref2 = compute_squared_distance(feats[0], feats[2])
            preds = torch.mul(1, dref1 > dref2)
            confs = torch.abs(dref1 - dref2)
        elif dist_type == 'cosine':
            dref1 = cosine_distances(feats[0], feats[1])
            dref2 = cosine_distances(feats[0], feats[2])
            preds = torch.mul(1, dref1 > dref2)
            confs = np.abs(dref1 - dref2)
    else:
        confs, preds = torch.max(outs, axis=1)
    return confs, preds


def train():
    data_paths, pthname = get_datapaths()
    print(f"Training model on D#: {refids}\nTraining arch: {modelarch}")
    trainGen  = dataGen(data_paths, batch_size, resize, 'train', incflip=True, shuffle=True)
    testGen   = dataGen(data_paths, batch_size, resize, 'test', incflip=False)
    model     = get_dlmodel(modelarch=modelarch, resize=resize, triplet=combinedloss)
    optimiser = optim.Adam(model.parameters(), lr=lr)
    if not combinedloss:
        criterion = nn.CrossEntropyLoss()
    else:
        criterion = [nn.CrossEntropyLoss(), OrderTripletLoss(device)]
    if combinedloss:
        print("Using Triplet Loss to assist training")
    if not os.path.exists(os.path.join(save_model_dir, modelarch)):
        os.makedirs(os.path.join(save_model_dir, modelarch))
    for e in range(nb_epochs):
        model.train()
        running_loss  = 0.
        nb_trainbatch = len(trainGen)
        for i, data in enumerate(trainGen):
            optimiser.zero_grad()
            ref      = data['ref']
            cand1    = data['cand1']
            cand2    = data['cand2']
            labels   = data['label'].long()
            cls1     = data['cls1']
            cls2     = data['cls2']
            outs, feats = model(ref, cand1, cand2)
            loss     = get_loss(criterion, outs, feats, labels, cls1, cls2) # criterion(outs, labels)
            loss.backward()
            optimiser.step()
            running_loss += loss.item()
            print(f'Epoch: {e}, batch: {i} / {nb_trainbatch}, training loss: {running_loss/(i+1)}')
        running_loss /= len(trainGen)
        with torch.no_grad():
            test_loss, _correct, _embcorr, _total = 0, 0, 0, 0
            model.eval()
            for i, data in enumerate(testGen):
                ref      = data['ref']
                cand1    = data['cand1']
                cand2    = data['cand2']
                labels   = data['label'].long()
                cls1     = data['cls1']
                cls2     = data['cls2']
                outs, feats = model(ref, cand1, cand2)
                loss     = get_loss(criterion, outs, feats, labels, cls1, cls2) # criterion(outs, labels)
                _, emb_preds = get_preds(outs, feats, use_embs=True, dist_type='l2')
                _, preds     = get_preds(outs, feats) # torch.max(outs, axis=1)
                _correct    += sum(preds==labels).item()
                _embcorr    += sum(emb_preds==labels).item()
                _total      += labels.size(0)
                test_loss   += loss.item()
            embtest_acc = _embcorr / _total
            test_acc = _correct / _total # binary block
            test_loss /= len(testGen)
            print("Epoch: %d, test_loss:%.3f, test_acc:%.3f, embtest_acc:%.3f" % (e+1, test_loss, test_acc, embtest_acc))
    if 'lora' in modelarch:
        model.save_lora_parameters(os.path.join(save_model_dir, modelarch, f'{pthname[:-4]}.safetensors'))
    else:
        torch.save(model.state_dict(), os.path.join(save_model_dir, modelarch, pthname))
        

if __name__ == "__main__":
    train()
