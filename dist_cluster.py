import numpy as np
import os
from sklearn.neural_network import MLPRegressor
from dist_sliding import get_bins, get_refbin_dict
from dist_sliding import sliding_window2d, sliding_window1d
from dist_reducedim import all_data_reduced_dim, get_rdparams
from dist_reducedim import reduce_dim_by_refids
from dist_dataprep import find_val_test_from_ind, load_ind_embeddings_from_proxies
from dist_dataprep import split_refs, get_mixing_data, powerset
from collections import Counter
import collections
import torch

# Specify:
metadata_dir     = './metadata/npy_labels'
feat_dir         = '../dist/saved_feat' # saved vit embeddings
modelarch        = 'selected'

# Fixed:
nb_testintervals = 25
anchorids        = [715, 2723, 389, 1673, 1006, 254, 2352, 667]
nb_grids         = 200 # sliding window
window_size      = 10
resize           = (224, 224)
rd_params        = get_rdparams(img_resize=resize, nb_reduced_dim=2)
bgval_embs_dir   = f'{feat_dir}/val/{modelarch}'
bgtrain_embs_dir = f'{feat_dir}/train/{modelarch}'
mulref_val_path  = os.path.join(metadata_dir, 'mulref_val.npy')
mulref_test_path = os.path.join(metadata_dir, 'mulref_test.npy')
embs_dict        = {'bgval': load_ind_embeddings_from_proxies(saved_embs_dir=bgval_embs_dir, anchorids=anchorids), 
                    'bgtrain': load_ind_embeddings_from_proxies(saved_embs_dir=bgtrain_embs_dir, anchorids=anchorids)}

def cosine_distances(a, b):
    return 1-np.dot(a, b)/(np.linalg.norm(a)*np.linalg.norm(b))


def get_weighted_distance(weights, embs, refid, cand1, cand2):
    wd1, wd2 = 0, 0 # weighted distance
    for i in range(len(weights)):
        wd1 += weights[i] * cosine_distances(embs[i][refid], embs[i][cand1])
        wd2 += weights[i] * cosine_distances(embs[i][refid], embs[i][cand2])
    return wd1, wd2


def get_modelpreds(combo, independent_data, saved_embs_dir):
    comboname = '_'.join([str(x) for x in combo])
    predfile = f'feat_{comboname}.npy'
    predpath = f'{saved_embs_dir}/{predfile}'
    allembds = np.load(predpath)
    allpreds = np.zeros(len(independent_data))
    for i in range(len(independent_data)):
        refid, cand1, cand2 = independent_data[i, -3:]
        refid, cand1, cand2 = int(refid), int(cand1), int(cand2)
        wd1 = cosine_distances(allembds[refid], allembds[cand1])
        wd2 = cosine_distances(allembds[refid], allembds[cand2])
        allpreds[i] = (wd1 > wd2) * 1
    return allpreds


# a combo is combination of model_{refid}s, e.g., (715,), (715, 389)
def get_combos(anchorids=anchorids):
    mids = []
    for i in anchorids:
        mids.append((i, ))
    return mids


# used to plotting cluters (which points are red / blue) and compute accmaps
def get_refacc(allpreds, independent_data):
    refacc_dict = {}
    refcorr_dict, refcount_dict = Counter(), Counter()
    for i in range(len(independent_data)):
        refcount_dict[int(independent_data[i, -3])] += 1
        if allpreds[i] == independent_data[i, 0]:
            refcorr_dict[int(independent_data[i, -3])] += 1
    for refid in refcount_dict:
        refacc_dict[refid] = refcorr_dict[refid] / refcount_dict[refid]
    # refacc_dict[ref_img_id] stores accuracy (wrt the ref image) of the cid model (where allpreds are obtained from)
    return refacc_dict


def get_refpoint_counts(allrefs, independent_data):
    sizes = np.zeros(allrefs.shape)
    for i, refid in enumerate(allrefs):
        sizes[i]  = np.sum(independent_data[:, -3]==refid)
    return sizes


def get_allpreds(combos, inddata, bgdb_split):
    # refacc_dict[cid][ref_id] is the accuracy of the ref image
    if bgdb_split == 'val':
        embs_dir = bgval_embs_dir
    elif bgdb_split == 'train':
        embs_dir = bgtrain_embs_dir
    allpreds = np.zeros([len(combos), len(inddata)])
    for cid, combo in enumerate(combos):
        try:
            allpreds[cid] = get_modelpreds(combo, inddata, embs_dir)
        except:
            print(f"Failed to load {combo} preds")
    return allpreds


def get_refaccdict(combos, inddata, allpreds):
    # refacc_dict[cid][ref_id] is the accuracy of the ref image
    refacc_dict = collections.defaultdict(dict)
    for cid, combo in enumerate(combos):
        refacc_dict[cid] = get_refacc(allpreds[cid], inddata)
    return refacc_dict


def get_bgtrain_points(inddata, rd):
    valrefs, testrefs = split_refs(inddata)
    allrefs = np.concatenate([valrefs, testrefs])
    points = reduce_dim_by_refids(allrefs, rd, rd_params, bgdb_split='train')
    return points, allrefs


def get_show_points(inddata, pca_dim=2):
    if pca_dim != 2:
        rd_params['dim'] = pca_dim
    valrefs, testrefs = split_refs(inddata['bgval'])
    valpoints, testpoints, anchors, rd = all_data_reduced_dim(valrefs, testrefs, anchorids, rd_params, bgdb_split='val')
    # Handling all the bins/grids and start sliding window
    allrefs   = np.concatenate([valrefs, testrefs])
    allpoints = np.concatenate([valpoints, testpoints])
    show_refs = allrefs
    show_pts  = allpoints
    trainpoints, trainrefs = get_bgtrain_points(inddata['bgtrain'], rd) # used for testing
    show_refs = {'val': show_refs, 'test': testrefs, 'train': trainrefs}
    show_pts = {'val': show_pts, 'test': testpoints, 'train': trainpoints}
    return show_pts, show_refs, anchors, rd


def get_all_refbin_dict(show_pts, show_refs, allbins):
    refbin_dict, val_binref_dict = {}, {}
    for split in ['val', 'test', 'train']:
        refbin_dict[split], val_binref_dict[split] = get_refbin_dict(show_pts[split], show_refs[split], allbins)
    if 'anchor' in show_refs:
        refbin_dict['anchor'], val_binref_dict['anchor'] = get_refbin_dict(show_pts['anchor'], show_refs['anchor'], allbins)
    return refbin_dict, val_binref_dict


def get_pca_ensembleacc_from_strategy(strategy, inddata_dict, indpred_dict, spaceinfo):
    def get_weights_from_strategy(ensemble, refbin_dict, inddata):
        weights = {}
        for i in range(len(inddata)):
            refid, _, _ = inddata[i, -3:]
            if refid not in weights:
                refid = int(refid)
                x, y = refbin_dict[1][refid], refbin_dict[0][refid]
                weights[refid] = ensemble[:, x, y]
        return weights
    inddata_bgval, inddata_bgtrain = inddata_dict['bgval'], inddata_dict['bgtrain']
    val_weights = get_weights_from_strategy(strategy, spaceinfo['refbin_dict']['val'], inddata_bgval)
    wpreds_val = get_weighted_preds(val_weights, inddata_bgval, indpred_dict['bgval'], embs_dict['bgval'])
    ensemble_acc_val = np.mean(wpreds_val==inddata_bgval[:, 0])
    print(f"PCA-based ensemble mulref_val acc: {ensemble_acc_val:.5f}")
    train_weights = get_weights_from_strategy(strategy, spaceinfo['refbin_dict']['train'], inddata_bgtrain)
    wpreds_train = get_weighted_preds(train_weights, inddata_bgtrain, indpred_dict['bgtrain'], embs_dict['bgtrain'])
    ensemble_acc = np.mean(wpreds_train==inddata_bgtrain[:, 0])
    print(f"PCA-based ensemble mulref_test, all set acc: {ensemble_acc:.5f}")
    return ensemble_acc, ensemble_acc_val


def vis_assist_ensemble():
    # Very bad name -- BOooooo...
    # given a refid 'id' : (e.g., 'id = 2023')
    # --> find the index in valrefs, i.e., valrefs[val_id_dict[id]] corresponds to id - 2023
    print(f"Computing Ensemble Acc for {modelarch}")
    inddata = {'bgval': np.load(mulref_val_path), 'bgtrain': np.load(mulref_test_path)}
    ref_combos  = get_combos()
    bgval_preds = get_allpreds(ref_combos, inddata['bgval'], bgdb_split='val')
    refacc_dict = get_refaccdict(ref_combos, inddata['bgval'], bgval_preds) # refacc_dict[cid][ref_id] is the accuracy of the ref image
    # PCA
    show_pts, show_refs, anchors, _ = get_show_points(inddata)
    # refbin_dict: which bin does the ref id belong
    # binref_dict: which bin contains which ref id(s)
    allbins = get_bins(show_pts['val'], nb_grids=nb_grids)
    refbin_dict, binref_dict = get_all_refbin_dict(show_pts, show_refs, allbins)
    ref_counts    = get_refpoint_counts(show_refs['val'], inddata['bgval'])
    window_accs   = sliding_window1d(refacc_dict, allbins, binref_dict['val'], window_size) # accs computed/controlled by binref_dict
    window_accs2d = sliding_window2d(refacc_dict, allbins, binref_dict['val'], window_size) # accs computed/controlled by binref_dict
    
    strategy = get_ensemble_strategy2d(window_accs2d)
    spaceinfo = {'refs': show_refs, 'refbin_dict': refbin_dict}
    indpred_dict = {'bgval': bgval_preds, 'bgtrain': get_allpreds(ref_combos, inddata['bgtrain'], bgdb_split='train')}
    get_pca_ensembleacc_from_strategy(strategy, inddata, indpred_dict, spaceinfo)



def get_ensemble_strategy2d(window_accs2d, mode='thres'):
    counts        = window_accs2d[:, :, :, 1]
    window_accs2d = window_accs2d[:, :, :, 0]
    if mode == 'avg':
        window_sum = np.sum(np.nan_to_num(window_accs2d), axis=0)
        strategy   = window_accs2d / window_sum
        strategy[np.isnan(strategy)] = 1 / strategy.shape[0]
    elif mode == 'max':
        strategy   = np.zeros(window_accs2d.shape)
        bestmodels = np.argmax(np.nan_to_num(window_accs2d), axis=0)
        nanidx     = np.isnan(window_accs2d)[0]
        for x in range(strategy.shape[1]):
            for y in range(strategy.shape[2]):
                if nanidx[x, y]:
                    strategy[bestmodels[x, y], x, y] = 1 / strategy.shape[0]
                else:
                    strategy[bestmodels[x, y], x, y] = 1.0
    elif mode == 'thres':
        thres      = 0.75
        strategy   = np.zeros(window_accs2d.shape)
        passed_win = np.where(np.nan_to_num(window_accs2d)>thres)
        for cid, x, y in zip(passed_win[0], passed_win[1], passed_win[2]):
            strategy[cid, x, y] = 1
        strategy   = np.nan_to_num(np.multiply(strategy, window_accs2d))
        for x in range(strategy.shape[1]):
            for y in range(strategy.shape[2]):
                if np.sum(strategy[:, x, y]) == 0 and np.sum(counts[:, x, y]) > 0:
                    bestmodel = np.argmax(window_accs2d[:, x, y])
                    strategy[bestmodel, x, y] = 1 # always choose the best one
                    # strategy[:, x, y] = window_accs2d[:, x, y] / np.sum(window_accs2d[:, x, y]) # 1 / strategy.shape[0] # average
        strategy   = strategy/np.sum(strategy, axis=0)
        strategy[np.isnan(strategy)] = 1 / strategy.shape[0]
    return strategy


def ml_ensemble(pca_dim=8):
    print(f"Computing ML-based Ensemble Acc for {modelarch}")
    # labels of the all proxy models
    def get_accs(refids, combos, inddata, preds):
        refacc_dict = get_refaccdict(combos, inddata, preds)
        accs = np.zeros((len(combos), len(refids)))
        for i, refid in enumerate(refids):
            for cid, combo in enumerate(combos):
                accs[cid, i] = refacc_dict[cid][refid]
        return accs

    inddata = {'bgval': np.load(mulref_val_path), 'bgtrain': np.load(mulref_test_path)}
    ref_combos  = get_combos()
    bgval_preds = get_allpreds(ref_combos, inddata['bgval'], bgdb_split='val')
    bgtrain_preds = get_allpreds(ref_combos, inddata['bgtrain'], bgdb_split='train')
    # reduce dimension to pca_dim by PCA
    show_pts, show_refs, anchors, rd = get_show_points(inddata, pca_dim)
    valaccs = get_accs(show_refs['val'], ref_combos, inddata['bgval'], bgval_preds)
    testaccs = get_accs(show_refs['test'], ref_combos, inddata['bgval'], bgval_preds)
    trainaccs = get_accs(show_refs['train'], ref_combos, inddata['bgtrain'], bgtrain_preds)
    show_wpreds = np.zeros((len(show_refs['val']), len(ref_combos)))
    test_wpreds = np.zeros((len(show_refs['test']), len(ref_combos)))
    train_wpreds = np.zeros((len(show_refs['train']), len(ref_combos)))
    weights = {'train': {}, 'val': {}}
    regrs = {}
    for cid, combo in enumerate(ref_combos):
        regr = MLPRegressor(learning_rate_init=0.001, hidden_layer_sizes=16, activation='logistic').fit(show_pts['val'], valaccs[cid])
        show_wpreds[:, cid] = regr.predict(show_pts['val'])
        print(f"For {combo}, regr train-loss on mulref_val set: {regr.score(show_pts['val'], valaccs[cid])}")
        train_wpreds[:, cid] = regr.predict(show_pts['train'])
        print(f"For {combo}, regr test-loss on mulref_test set: {regr.score(show_pts['train'], trainaccs[cid])}")
        regrs[combo] = regr
    
    for i, refid in enumerate(show_refs['val']):
        weights['val'][refid] = show_wpreds[i] / np.sum(show_wpreds[i])
    for i, refid in enumerate(show_refs['train']):
        weights['train'][refid] = train_wpreds[i] / np.sum(train_wpreds[i])

    wpreds_val = get_weighted_preds(weights['val'], inddata['bgval'], bgval_preds, embs_dict['bgval'])
    print(f"ML-based ensemble ({modelarch})  train acc on mulref_val: {np.mean(wpreds_val==inddata['bgval'][:, 0]):.5f}")
    wpreds_train = get_weighted_preds(weights['train'], inddata['bgtrain'], bgtrain_preds, embs_dict['bgtrain'])
    print(f"ML-based ensemble ({modelarch})  test acc on mulref_test: {np.mean(wpreds_train==inddata['bgtrain'][:, 0]):.5f}")


def get_weighted_preds(weights, inddata, preds, embs):
    weighted_preds = np.zeros(len(inddata))
    for i in range(len(inddata)):
        refid, cand1, cand2 = inddata[i, -3:]
        refid, cand1, cand2 = int(refid), int(cand1), int(cand2)
        wd1, wd2 = get_weighted_distance(weights[refid], embs, refid, cand1, cand2)
        weighted_pred = (wd1 > wd2) * 1
        weighted_preds[i] = weighted_pred
    return weighted_preds


if __name__ == "__main__":
    # regrs = ml_ensemble(pca_dim=64) # MLP-based
    vis_assist_ensemble() # PCA-based