import numpy as np
import collections

def get_bins(imgpoints, nb_grids):
    allbins = np.zeros([imgpoints.shape[1], nb_grids+1])
    for d in range(imgpoints.shape[1]):
        allbins[d] = np.linspace(min(imgpoints[:, d]), max(imgpoints[:, d]), nb_grids+1)
    return allbins


def get_refbin_dict(imgpoints, refs, allbins):
    refbin_dict = collections.defaultdict(dict)
    binref_dict = collections.defaultdict(dict)
    for d in range(imgpoints.shape[1]):
        data1d = imgpoints[:, d]
        for i in range(len(allbins[d])-1):
            points = np.where(np.multiply(data1d < allbins[d][i+1], data1d >= allbins[d][i]))[0]
            if i == 0:
                morepoints = np.where(data1d < allbins[d][i])[0]
                points = np.concatenate([points, morepoints])
            elif i == len(allbins[d]) - 2:
                morepoints = np.where(data1d >= allbins[d][i+1])[0]
                points = np.concatenate([points, morepoints])
            if len(points) > 0:
                try:
                    points = refs[points]
                except:
                    points = np.array(refs)[points]
                for p in points:
                    refbin_dict[d][p] = i
            binref_dict[d][i] = points
    return refbin_dict, binref_dict


# return accuracy on each grid from allbins
def get_grid_accs1d(refacc_dict, allbins, binref_dict):    
    accs = np.zeros([len(refacc_dict), allbins.shape[0], allbins.shape[1]-1])
    for cid in refacc_dict:
        for d in range(allbins.shape[0]):
            for i in range(allbins.shape[1]-1):
                points = binref_dict[d][i]
                racc = 0
                for p in points:
                    racc += refacc_dict[cid][p] / len(points)
                accs[cid, d, i] = racc
    return accs


def get_grid_accs2d(refacc_dict, allbins, binref_dict):
    accmaps = np.zeros([len(refacc_dict), allbins.shape[1]-1, allbins.shape[1]-1])
    nummaps = np.zeros([allbins.shape[1]-1, allbins.shape[1]-1])
    for i in range(allbins.shape[1]-1):
        for j in range(allbins.shape[1]-1):
            points = np.intersect1d(binref_dict[1][i], binref_dict[0][j]) # number of ref images
            nummaps[i, j] = len(points)
            for cid in refacc_dict:
                racc = 0
                for p in points:
                    racc += refacc_dict[cid][p] / len(points)
                accmaps[cid, i, j] = racc
    return accmaps, nummaps


# really no need for accmaps.. can use refacc_dict and merge get_grid_accs1d and this function
def sliding_window1d(refacc_dict, allbins, binref_dict, window_size):
    # window_accs/accmaps: [number of combos, number of dimensions, number of grids] -- one combo represents one model
    accmaps = get_grid_accs1d(refacc_dict, allbins, binref_dict)
    window_accs = np.zeros(accmaps.shape)
    no_point_grids = collections.defaultdict(dict)
    for d in range(accmaps.shape[1]):
        for anchor_bid in range(allbins.shape[1]-1):
            l_window      = max(0, anchor_bid-window_size//2)
            r_window      = min(allbins.shape[1]-2, anchor_bid+window_size//2)
            combo_weights = np.zeros(accmaps.shape[0])
            window_count  = 0
            for pointer_bid in range(l_window, r_window+1):
                nb_points = len(binref_dict[d][pointer_bid])
                combo_weights += accmaps[:, d, pointer_bid] * nb_points
                window_count  += nb_points
            if window_count == 0:
                print(f"No point in the window centered at dimension {d}: {anchor_bid}")
                no_point_grids[d][anchor_bid] = True
            else:
                combo_weights /= window_count # Raw accuracy in the windoww
            window_accs[:, d, anchor_bid] = combo_weights

    # deal with grids with no points: use the nearest strategy
    for d in range(accmaps.shape[1]):
        for anchor_bid in no_point_grids[d]:
            l_bid, r_bid = anchor_bid, anchor_bid
            while l_bid in no_point_grids[d] and l_bid > 0:
                l_bid = max(0, l_bid-1)
            while r_bid in no_point_grids[d] and r_bid < allbins.shape[1]-2:
                r_bid = min(allbins.shape[1]-2, r_bid+1)
            if anchor_bid - l_bid <= r_bid - anchor_bid:
                combo_weights = window_accs[:, d, l_bid]
            else:
                combo_weights = window_accs[:, d, r_bid]
            window_accs[:, d, anchor_bid] = combo_weights
    return window_accs


def sliding_window2d(refacc_dict, allbins, binref_dict, window_size):
    if type(window_size) is int:
        window_size  = (window_size, window_size)
    v_window         = window_size[0] // 2
    h_window         = window_size[1] // 2
    accmaps, nummaps = get_grid_accs2d(refacc_dict, allbins, binref_dict)
    window_accs      = np.zeros(accmaps.shape+(2,))
    for i in range(accmaps.shape[1]):
        top    = max(0, i-v_window)
        bottom = min(accmaps.shape[1], i+v_window)
        for j in range(accmaps.shape[2]):
            left  = max(0, j-h_window)
            right = min(accmaps.shape[2], j+h_window)
            window_count = np.sum(nummaps[top:bottom, left:right])            
            if window_count > 0:
                window = np.multiply(accmaps[:, top:bottom, left:right], nummaps[top:bottom, left:right])
                window = np.sum(np.sum(window, axis=-1), axis=-1)
                window_accs[:, i, j, 0] = window / window_count
            else:
                window_accs[:, i, j, 0] = np.nan
            window_accs[:, i, j, 1] = window_count
    # # deal with grids with no points: use the nearest strategy
    # for i, j in no_point_grids:
    return window_accs



