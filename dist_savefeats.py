import os
import argparse
import torch
import numpy as np
from dist_loadmodel import get_dlmodel, recover_weights
from dist_dataprep import get_dbs
import torchvision.transforms as transforms
import warnings
warnings.filterwarnings("ignore")

resize = (224, 224)

def argparser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--cs_refid', type=str, default='254')
    parser.add_argument('--cs_arch', type=str, default='resnet18') # or bgdb
    parser.add_argument('--bgdb_split', type=str, default='val') # 'val', or 'train' -- only for 10k testing triplets
    parser.add_argument('--model_path', type=str, default='./saved_models') # trained cs models
    parser.add_argument('--feat_path', type=str, default='./saved_feats') # save cs models' embeddings to
    args = parser.parse_args()
    args.model_path = os.path.join(args.model_path, args.cs_arch, args.cs_refid)
    return args

# extract and save features/embeddings from proxy models we trained
def save_proxy_feats(args):
    if 'lora' in args.cs_arch:
        mpath = f'{args.model_path}.safetensors'
        lora  = True
        f_dim = 768
    else:
        mpath = f'{args.model_path}.pth'
        lora  = False
        f_dim = 512
        if args.cs_arch == 'resnet50':
            f_dim = 2048
    print(f"Saving features/embeddings for {args.cs_arch}: {args.cs_refid}")
    transform = transforms.Compose([
                                 transforms.ToTensor(),
                                 transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
                                 ])
    model = get_dlmodel(args.cs_arch)
    model = recover_weights(model, mpath, lora=lora)
    model.eval()
    dbs = get_dbs(split=args.bgdb_split)
    allfeats = np.zeros([len(dbs), f_dim])
    with torch.no_grad():
        for i in range(len(dbs)):
            allfeats[i] = model.encode_image(torch.unsqueeze(transform(dbs[i]['img_data'].resize(resize)), 0))
            print(f"Finished {i}/{len(dbs)}")
    npydir = os.path.join(args.feat_path, args.bgdb_split, args.cs_arch)
    if not os.path.exists(npydir):
        os.makedirs(npydir)
    np.save(f'{npydir}/feat_{args.cs_refid}.npy', allfeats)


if __name__ == "__main__":
    args = argparser()
    save_proxy_feats(args)


