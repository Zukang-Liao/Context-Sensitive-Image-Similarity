import torch
from net.dist_model import resnet18, VGGnet, RGBBranch, ResNetOrder
from net.lora import LoRA_ViT
from net.cvnet import CVNet_Rerank

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
vitbase_path = './net/pretrained/dino_vitbase16_pretrain.pth'
resnet18_place365_path = './net/pretrained/RGB_ResNet18_Places.pth.tar'


def get_dlmodel(modelarch, resize=(224,224), useAtt=False, triplet=False):
    if modelarch == 'resnet18':
        model  = resnet18(num_classes=2, pretrained=True)
    elif modelarch == 'resnet50' or modelarch == 'resnet34':
        feat_dim = 2048 if modelarch == 'resnet50' else 512
        model  = ResNetOrder(num_classes=2, arch=modelarch, feat_dim=feat_dim, pretrained=True)
    elif modelarch == 'resnet18-place365' or modelarch == 'place365':
        model  = RGBBranch(num_classes=2, att=useAtt, triplet=triplet)
        ckpt   = torch.load(resnet18_place365_path, map_location=torch.device(device))
        model_dict = model.state_dict()
        in_sd = {}
        for k in model_dict:
            if k in ckpt['state_dict']:
                in_sd[k] = ckpt['state_dict'][k]
        model_dict.update(in_sd)
        model.load_state_dict(model_dict)
        # model.load_state_dict(ckpt['state_dict'])
    elif 'vgg' in modelarch:
        model = VGGnet(num_classes=2, arch=modelarch, pretrained=True)
    elif modelarch == 'lora_vit':
        from base_vit import ViT
        net = ViT('B_16_imagenet1k')
        net = load_dino_to_lora_vit(net, model_path=vitbase_path)
        model = LoRA_ViT(net, r=4)
        num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"trainable parameters: {num_params}")
    elif 'vit' in modelarch.lower():
        from dist_model import ViT
        if '14' in modelarch:
            model = ViT('vit_base_patch14_dinov2', num_classes=2, img_size=resize[0], pretrained=True, drop_rate=0.1)
        else:
            model = ViT('vit_base_patch16_224_in21k', num_classes=2, img_size=resize[0], pretrained=True, drop_rate=0.1)
    elif modelarch.lower() == 'cvnet':
        # model = CVNet_Rerank(cfg.MODEL.DEPTH, cfg.MODEL.HEADS.REDUCTION_DIM)
        model = CVNet_Rerank(50, 2048) # resnet 50
    if useAtt:
        print("With Attention Heads")
    return model


def load_dino_to_lora_vit(model, model_path):
    ckpt = torch.load(model_path, map_location=torch.device(device))
    if 'state_dict' in ckpt:
        ckpt = ckpt['state_dict']
    model_dict = model.state_dict()
    in_sd, out_sd = {}, set()
    for k in model_dict:
        kw = k.replace('transformer.', '')
        if 'pwff' in kw:
            kw = kw.replace('pwff', 'mlp')
        elif 'proj_q' in kw:
            kw = kw.replace('proj_q', 'qkv')
        elif 'proj_k' in kw:
            kw = kw.replace('proj_k', 'qkv')
        elif 'proj_v' in kw:
            kw = kw.replace('proj_v', 'qkv')
        elif 'proj.' in kw:
            kw = kw.replace('proj.', 'attn.proj.')
        elif kw == 'positional_embedding.pos_embedding':
            kw = 'pos_embed'
        elif 'embedding' in kw:
            kw = kw.replace('embedding', 'embed.proj')
        elif kw == 'class_token':
            kw = 'cls_token'
        if k in ckpt:
            in_sd[k] = ckpt[k]
        elif kw in ckpt:
            if 'proj_q' in k:
                if 'bias' in kw:
                    in_sd[k] = ckpt[kw][:768]
                else:
                    in_sd[k] = ckpt[kw][:768, :]
            elif 'proj_k' in k:
                if 'bias' in kw:
                    in_sd[k] = ckpt[kw][768:768*2]
                else:
                    in_sd[k] = ckpt[kw][768:768*2, :]
            elif 'proj_v' in k:
                if 'bias' in kw:
                    in_sd[k] = ckpt[kw][768*2:]
                else:
                    in_sd[k] = ckpt[kw][768*2:, :]
            else:
                in_sd[k] = ckpt[kw]
        else:
            out_sd.add(k)
    if len(out_sd) > 0:
        print(f"Loaded {len(in_sd)} / {len(model_dict)}. Failed to recover {len(out_sd)} weights")
        # import ipdb; ipdb.set_trace()
    model_dict.update(in_sd)
    model.load_state_dict(model_dict)
    return model


def recover_weights(model, model_path, lora=False):
    if lora:
        model.load_lora_parameters(model_path)
        return model
    try:
        model.load_state_dict(torch.load(model_path, map_location=torch.device(device)))
    except:
        ckpt = torch.load(model_path, map_location=torch.device(device))
        if 'state_dict' in ckpt:
            ckpt = ckpt['state_dict']
        elif 'model_state' in ckpt:
            ckpt = ckpt['model_state']
        model_dict = model.state_dict()
        in_sd, out_sd = {}, set()
        for k in model_dict:
            kw = k.replace('model.', '')
            if k in ckpt:
                in_sd[k] = ckpt[k]
            elif kw in ckpt:
                in_sd[k] = ckpt[kw]
            else:
                # gl18
                replace_dict = {'model.layer1': 'features.4', 'model.layer2': 'features.5', 'model.layer3': 'features.6', 'model.layer4': 'features.7', 'model.bn1': 'features.1', 'model.conv1': 'features.0'}
                for rk in replace_dict:
                    if rk in k:
                        kw = k.replace(rk, replace_dict[rk])
                        if kw in ckpt:
                            in_sd[k] = ckpt[kw]
                        break
                else:
                    out_sd.add(k)
        if len(out_sd) > 0:
            print(f"Loaded {len(in_sd)} / {len(model_dict)}. Failed to recover {len(out_sd)} weights")
        model_dict.update(in_sd)
        model.load_state_dict(model_dict)
    return model