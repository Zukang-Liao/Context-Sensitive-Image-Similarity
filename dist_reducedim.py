
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.manifold import TSNE
from dist_dataprep import get_dbs
import numpy as np

PCA_FEAT_DIR  = f'../dist/saved_feat'
PCA_FEAT_NAME = 'vit_feats.npy'

def get_data_from_db(dataids, dbs, rd_feat, img_resize, bgdb_split):
    if rd_feat == 'CNNFeat':
        PCA_FEAT_PATH = f'{PCA_FEAT_DIR}/{bgdb_split}/{PCA_FEAT_NAME}'
        cand_items = np.load(PCA_FEAT_PATH)
        data = np.zeros([len(dataids), cand_items.shape[1]])
        for i, refid in enumerate(dataids):
            try:
                data[i] = cand_items[int(refid)]
            except:
                import ipdb; ipdb.set_trace()
    elif rd_feat == 'raw':
        data = np.zeros((len(dataids), img_resize[0]*img_resize[1]*3))
        for i, refid in enumerate(dataids):
            img = np.array(dbs[int(refid)]['img_data'].resize(img_resize)) / 255.
            data[i] = img.reshape(-1)
    return data

def preprocess(imgdata, rd_feat):
    if rd_feat == 'raw':
        imgdata = StandardScaler().fit_transform(imgdata)
    return imgdata


def train_reduce_dimension(imgdata, nb_reduced_dim):
    rd = PCA(nb_reduced_dim)
    rd.fit(imgdata)
    print(rd.explained_variance_ratio_[:5])
    return rd


def tsne_reduce_dimension(imgdata, nb_reduced_dim):
    rd = TSNE(nb_reduced_dim)
    return rd.fit_transform(imgdata)


def get_rdparams(img_resize, nb_reduced_dim):
    params = {}
    params['feat']     = 'CNNFeat'
    params['img_size'] = img_resize
    params['dim']      = nb_reduced_dim
    return params

def all_data_reduced_dim(valrefs, testrefs, anchorids, rd_params, bgdb_split='val'):
    rd_feat  = rd_params['feat']
    resize   = rd_params['img_size']
    dim      = rd_params['dim']
    dbs      = get_dbs(bgdb_split)
    valdata  = get_data_from_db(valrefs, dbs, rd_feat, resize, bgdb_split)
    testdata = get_data_from_db(testrefs, dbs, rd_feat, resize, bgdb_split)
    anchors  = get_data_from_db(anchorids, dbs, rd_feat, resize, bgdb_split)
    valdata  = preprocess(valdata, rd_feat)
    testdata = preprocess(testdata, rd_feat)
    anchors  = preprocess(anchors, rd_feat)
    if dim < 1:
        rd = None # no dimensional reduction
    else:
        rd       = train_reduce_dimension(np.concatenate([valdata, anchors]), dim)
        valdata  = rd.transform(valdata)
        testdata = rd.transform(testdata)
        anchors  = rd.transform(anchors)
    return valdata, testdata, anchors, rd

def reduce_dim_by_refids(refids, rd, rd_params, bgdb_split):
    rd_feat = rd_params['feat']
    resize  = rd_params['img_size']
    dbs  = get_dbs(bgdb_split)
    data = get_data_from_db(refids, dbs, rd_feat, resize, bgdb_split)
    if rd is not None:
        return rd.transform(data)
    else:
        return data

