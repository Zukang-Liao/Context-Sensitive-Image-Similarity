import os
import numpy as np
import pandas as pd
import itertools
from math import factorial
from itertools import chain, combinations
from datasets import BG20K4LABEL
import warnings
warnings.filterwarnings("ignore")


nb_valref = 320 # number of reference images used for validation set --> to obtain ensemble strategies
nbcol4labels = 6 # order, cls1, cls2, refid, id1, id2 -- when order=0, cls1<=cls2
LABEL_PATH = './metadata/txt_labels/mulref_val.txt'

def powerset(iterable):
    "powerset([1,2,3]) --> () (1,) (2,) (3,) (1,2) (1,3) (2,3) (1,2,3)"
    s = list(iterable)
    # does not include empty set
    return chain.from_iterable(combinations(s, r) for r in range(1, len(s)+1))
    
def concatenate_flipped(data):
    flipped_data = np.zeros(data.shape)
    flipped_data[:, 0] = 1 - data[:, 0]
    flipped_data[:, -1] = data[:, -2]
    flipped_data[:, -2] = data[:, -1]
    flipped_data[:, -3] = data[:, -3]
    flipped_data[:, -4] = data[:, -5]
    flipped_data[:, -5] = data[:, -4]
    return np.concatenate([data, flipped_data], axis=0)

def get_dbs(split):
    dbs = BG20K4LABEL(split=split)
    return dbs

# return entries in indepenet_data that have any of the refid in refids
def get_data_from_refids(refids, independent_data):
    indices = np.empty(len(refids), dtype=object)
    for i, refid in enumerate(refids):
        indices[i] = np.where(independent_data[:, -3] == refid)[0]
    return independent_data[np.concatenate(indices)]

def split_refs(independent_data, shuffle=True, seed=2):
    allrefs = np.unique(independent_data[:, -3])
    if shuffle:
        np.random.seed(seed)
        np.random.shuffle(allrefs)
    # valrefs, testrefs = allrefs[:len(allrefs)//2], allrefs[len(allrefs)//2:]
    valrefs, testrefs = allrefs[:nb_valref], allrefs[nb_valref:]
    return valrefs, testrefs

def load_ind_embeddings_from_proxies(saved_embs_dir, anchorids):
    embs = {}
    for i, aid in enumerate(anchorids):
        embs[i] = np.load(os.path.join(saved_embs_dir, f'feat_{aid}.npy'))
    return embs


def concatenate_scores(cand_cls, ref_id, cand_ids, nb_comb):
    idx = 0
    processed_labels = np.zeros([nb_comb, nbcol4labels])
    for pair in itertools.combinations(range(nb_comb), 2):
        if np.random.random() >= 0.5:
            # gt/pred: 0 -- means id1 is closer than id2 given ref.
            processed_labels[idx] = [0, cand_cls[0], cand_cls[1], ref_id, cand_ids[pair[0]], cand_ids[pair[1]]]
        else:
            processed_labels[idx] = [1, cand_cls[1], cand_cls[0], ref_id, cand_ids[pair[1]], cand_ids[pair[0]]]
        idx += 1
    return processed_labels


def prep_data(label_path):
    nb_display = 3 # the number of candidates when labelling
    annotations = pd.read_csv(label_path, ' ')
    nb_comb = factorial(nb_display) // factorial(2) // factorial(nb_display - 2)
    labels = np.zeros([annotations.shape[0]*nb_comb, nbcol4labels]) # last three colomns for labels, order, cls1, cls2 --> cls1 <= cls2 when order is 0 
    for data_id in range(annotations.shape[0]):
        ref_id = annotations.values[data_id][0]
        cand_ids = annotations.values[data_id][[1, 3, 5]]
        cand_cls = annotations.values[data_id][[2, 4, 6]]
        cur_idx  = data_id * nb_comb
        label_b = concatenate_scores(cand_cls, ref_id, cand_ids, nb_comb)
        labels[cur_idx:cur_idx+nb_comb] = label_b
        print(f"Finished data_id: {data_id+1} / {annotations.shape[0]}")
    npy_path = label_path.replace('txt', 'npy')
    if not os.path.exists(os.path.dirname(npy_path)):
        os.makedirs(os.path.dirname(npy_path))
    np.save(npy_path, labels)


def get_mixing_data(data_paths, split='', incflip=False, shuffle=False, seed=2):
    train_data, test_data = [], []
    for data_path in data_paths:
        data = np.load(data_path)
        if split == 'all':
            testfrom = len(data)
        else:
            testfrom = len(data) * 2 // 3
        train_data.append(data[:testfrom])
        test_data.append(data[testfrom:])
    train_data = np.concatenate(train_data, axis=0)
    test_data  = np.concatenate(test_data, axis=0)
    if incflip:
        train_data = concatenate_flipped(train_data)
        if split  != 'all':
            test_data = concatenate_flipped(test_data)
    if shuffle:
        np.random.seed(seed)
        np.random.shuffle(train_data)
    return train_data, test_data


def get_mixingdata_byref(data_path, split, incflip=False):
    independent_data     = np.load(data_path)
    valrefs, testrefs    = split_refs(independent_data)
    if split == 'test':
        independent_data = get_data_from_refids(testrefs, independent_data)
    elif split == 'val':
        independent_data = get_data_from_refids(valrefs, independent_data)
    if incflip:
        independent_data = concatenate_flipped(independent_data)
    return independent_data


def find_val_test_from_ind(inddata, valrefs, testrefs):
    valrefs, testrefs = set(valrefs), set(testrefs)
    nb_val, nb_test = 0, 0
    for i in range(len(inddata)):
        ref = inddata[i, -3]
        if ref in valrefs:
            nb_val += 1
        if ref in testrefs:
            nb_test += 1
    val_indices   = np.zeros(nb_val).astype(np.int)
    test_indices  = np.zeros(nb_test).astype(np.int)
    val_i, test_i = 0, 0
    for i in range(len(inddata)):
        ref = inddata[i, -3]
        if ref in valrefs:
            val_indices[val_i] = i
            val_i  += 1
        if ref in testrefs:
            test_indices[test_i] = i
            test_i += 1
    return val_indices, test_indices


if __name__ == "__main__":
    prep_data(LABEL_PATH)


