import numpy as np
from scipy.io import loadmat

import os

def load_pos_mat(fdir, trialTypes=['hComb', 'openF'], extractAttributes=None):
    if not os.path.exists(fdir):
        raise ValueError
    mat = loadmat(fdir)
    pos = mat['pos'][0, 0]
    num_trialType = len(pos)
    assert len(trialTypes) == num_trialType
    d = {}
    d['goalPosition'], d['goalID'] = mat['goalPosition'], mat['goalID']
    d['frameSize'] = mat['frameSize'][0]
    for i in range(num_trialType):
        num_trials = pos[i].shape[1]
        d[trialTypes[i]] = {}
        allAttributes = np.array(list(np.dtype(pos[i].dtype).names))
        attributes = allAttributes if extractAttributes is None else np.array(extractAttributes)
        for j in range(len(attributes)):
            d[trialTypes[i]][attributes[j]] = []
            ind = np.where(allAttributes == attributes[j])[0][0]
            for k in range(num_trials):
                d[trialTypes[i]][attributes[j]].append(pos[i][0, k][ind])
    return d


def load_platformLoc_mat(fdir, trialTypes=['hComb', 'openF'], extractAttributes=['body']):
    if not os.path.exists(fdir):
        raise ValueError
    mat = loadmat(fdir)['platformLocations']
    d = {}
    assert len(mat[0, 0]) == len(trialTypes)
    for tt in range(len(trialTypes)):
        d[trialTypes[tt]] = {}
        num_trials = len(mat[0, 0][tt])
        for t in range(num_trials):
            d[trialTypes[tt]][t] = {}
        allAttributes = np.array(list(np.dtype(mat[0, 0][tt][t, 0].dtype).names))
        attributes = allAttributes if extractAttributes is None else np.array(extractAttributes)
        for i in range(len(attributes)):
            ind = np.where(allAttributes == attributes[i])[0][0]
            for t in range(num_trials):
                d[trialTypes[tt]][t][attributes[i]] = []
                num_ev = mat[0, 0][tt][t, 0].shape[1]
                for j in range(num_ev):
                    d[trialTypes[tt]][t][attributes[i]].append(mat[0, 0][tt][t, 0][0, j][ind][0, 0])
    return d


def load_rateMaps_mat(fdir, extractAttributes=None):
    if not os.path.exists(fdir):
        raise ValueError
    mat = loadmat(fdir)
    d = {}
    d['frameSize'] = mat['frameSize']
    mat = mat['rateMaps']
    num_cells = mat.shape[1]
    for i in range(num_cells):
        unit, hComb, openF, neuronType = mat[0, i]
        unit = unit[0]
        neuronType = neuronType[0]
        d[unit] = {}
        d[unit]['neuronType'] = neuronType
        hComb = hComb[0, 0]
        openF = openF[0, 0]
        allAttributes = np.array(list(np.dtype(hComb.dtype).names))
        attributes = allAttributes if extractAttributes is None else np.array(extractAttributes)
        d[unit]['hComb'] = {}
        for j in range(len(attributes)):
            ind = np.where(allAttributes == attributes[j])[0][0]
            if attributes[j] not in ['ultraLow', 'lowRes', 'highRes', 'ultraHigh']:
                d[unit]['hComb'][attributes[j]] = hComb[ind]
            else:
                d[unit]['hComb'][attributes[j]] = {}
                rateMap_attributes = np.array(list(np.dtype(hComb[ind].dtype).names))
                for k in range(len(rateMap_attributes)):
                    d[unit]['hComb'][attributes[j]][rateMap_attributes[k]] = hComb[ind][0, 0][k]
        d[unit]['openF'] = {}
        for j in range(len(attributes)):
            ind = np.where(allAttributes == attributes[j])[0][0]
            if attributes[j] not in ['ultraLow', 'lowRes', 'highRes', 'ultraHigh']:
                d[unit]['openF'][attributes[j]] = openF[ind]
            else:
                d[unit]['openF'][attributes[j]] = {}
                rateMap_attributes = np.array(list(np.dtype(openF[ind].dtype).names))
                for k in range(len(rateMap_attributes)):
                    d[unit]['openF'][attributes[j]][rateMap_attributes[k]] = openF[ind][0, 0][k]
    return d


def load_mrlFocus_ctrlDist_mat(fdir, trialTypes=['hComb', 'openF']):
    if not os.path.exists(fdir):
        raise ValueError
    mat = loadmat(fdir)
    d = {}
    d['angleEdges'] = mat['angleEdges'][0]
    d['purDirDists'] = {}
    d['relDirDists'] = {}
    num_angles = len(d['angleEdges']) - 1
    
    purDirDists = mat['purDirDists'][0, 0]
    assert len(purDirDists) == len(trialTypes)
    for i in range(len(trialTypes)):
        num_platforms = purDirDists[i].shape[1]
        d['purDirDists'][trialTypes[i]] = np.zeros((num_platforms, num_angles))
        for j in range(num_platforms):
            if len(purDirDists[i][0, j][0]) == 0:
                continue
            d['purDirDists'][trialTypes[i]][j] = purDirDists[i][0, j][0]
    
    relDirDists = mat['relDirDists'][0, 0]
    assert len(relDirDists) == len(trialTypes)
    conSink_loc_shape = relDirDists[0][0, 0].shape
    assert conSink_loc_shape[-1] == num_angles
    for i in range(len(trialTypes)):
        num_platforms = relDirDists[i].shape[1]
        d['relDirDists'][trialTypes[i]] = np.zeros((num_platforms, ) + conSink_loc_shape)
        for j in range(num_platforms):
            if len(relDirDists[i][0, j][0]) == 0:
                continue
            d['relDirDists'][trialTypes[i]][j] = relDirDists[i][0, j]
    return d


def circ_r(alpha, w=None, d=None, dim=None):
    if dim is None:
        dim = 0
    
    if w is None:
        w = np.ones_like(alpha)
    else:
        assert w.shape[0] == alpha.shape[0] # and w.shape[1] == alpha.shape[1]
    
    if d is None:
        d = 0
    
    r = np.sum(w * np.exp(1j * alpha), dim)
    r = np.abs(r) / np.sum(w, dim)
    
    if d != 0:
        c = d / 2 / np.sin(d/2)
        r = c * r
    
    return r

def circ_mean(alpha, w=None, dim=None):
    if dim is None:
        dim = 0
    
    if w is None:
        w = np.ones_like(alpha)
    else:
        assert w.shape[0] == alpha.shape[0] # and w.shape[1] == alpha.shape[1]
    
    r = np.sum(w * np.exp(1j * alpha), dim)
    mu = np.angle(r)
    
    return mu # TODO: could also return the upper and lower limits of the confidence interval

def circ_rtest(alpha, w=None, d=None):
    if len(alpha.shape) > 1:
        if alpha.shape[1] > alpha.shape[0]:
            alpha = alpha.T
    
    if w is None:
        r = circ_r(alpha)
        n = len(alpha)
    else:
        assert len(alpha) == len(w)
        if d is None:
            d = 0
        r = circ_r(alpha, w, d)
        n = np.sum(w)
    
    R = n * r # Rayleigh's R
    z = (R ** 2) / n # Rayleigh's z
    pval = np.exp(np.sqrt(1 + 4*n + 4*(n**2-R**2)) - (1+2*n))
    return z, pval