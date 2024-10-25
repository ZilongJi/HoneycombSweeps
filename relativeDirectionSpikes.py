import numpy as np
import pickle

import time

import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '../'))

from data_analysis_utils import (
    load_pos_mat, 
    load_platformLoc_mat, 
    load_rateMaps_mat, 
    load_mrlFocus_ctrlDist_mat, 
    circ_r, 
    circ_mean, 
    circ_rtest, 
)

def relativeDirectionSpikes(pos_fdir, platformLoc_fdir, 
                            rateMaps_fdir, mrlFocus_ctrlDist_fdir, savedir=None, verbose=False):
    if savedir is not None:
        if os.path.exists(savedir):
            with open(savedir, 'rb') as f:
                d_mrlFocus, angleEdges, histBinCentres, d_signifCellsNorm = pickle.load(f)
            return d_mrlFocus, angleEdges, histBinCentres, d_signifCellsNorm
    
    t0 = time.time()
    
    trialTypes = ['hComb', 'openF']
    d_pos = load_pos_mat(pos_fdir, trialTypes, extractAttributes=['sample'])
    d_platformLoc = load_platformLoc_mat(platformLoc_fdir)
    d_rateMaps = load_rateMaps_mat(rateMaps_fdir)
    d_mrlFocus_ctrlDist = load_mrlFocus_ctrlDist_mat(mrlFocus_ctrlDist_fdir)
    
    d_mrlFocus = {}
    
    samples = {}
    plat = {}
    for tt in trialTypes:
        samples[tt] = np.round(np.concatenate(d_pos[tt]['sample'], axis=0))[:, 0]
        plat[tt] = []
        
        num_trials = len(d_pos[tt]['sample'])
        for t in range(num_trials):
            plat[tt].append(np.array(d_platformLoc[tt][t]['body']).reshape(-1, 1))
        plat[tt] = np.concatenate(plat[tt], axis=0)
    
    num_cells = len(d_rateMaps) - 1
    cellCounter = 0
    angleEdges = d_mrlFocus_ctrlDist['angleEdges']
    histBinCentres = np.deg2rad((angleEdges[1]-angleEdges[0])/2 + angleEdges[:-1])
    
    binSize = 15
    frameSize = d_rateMaps['frameSize'][0]
    xAxis = np.arange(1, frameSize[0]+1, binSize)
    yAxis = np.arange(1, frameSize[1]+1, binSize)
    
    nShuffles = [50, 100, 200, 500, 1000]
    
    d_signifCellsNorm = {}
    for tt in range(len(trialTypes)):
        d_signifCellsNorm[trialTypes[tt]] = []
    
    for k in d_rateMaps.keys():
        if k == 'frameSize':
            continue
        if d_rateMaps[k]['neuronType'] not in ['pyramid', 'p']:
            continue
        
        cellFlag = 0
        
        for tt in range(len(trialTypes)):
            spikePos = d_rateMaps[k][trialTypes[tt]]['spikePos']
            spikeHD = d_rateMaps[k][trialTypes[tt]]['spikeHD']
            spikeSamples = d_rateMaps[k][trialTypes[tt]]['spikeSamples'][:, 0]
        
            if len(spikeHD) < 500:
                continue
            else:
                cellFlag += 1
            
            if cellFlag == 1:
                cellCounter += 1
                d_mrlFocus[k] = {}
            
            d_mrlFocus[k][trialTypes[tt]] = {}
            
            platInd = np.round(np.interp(spikeSamples, samples[trialTypes[tt]], np.arange(len(samples[trialTypes[tt]])))).astype(np.int32)
            del spikeSamples
            spikePlats = plat[trialTypes[tt]][platInd]

            platDist_Rel = []
            totalDist_Rel = []
            
            platDist_Pure = []
            totalDist_Pure = []
            
            for p in range(len(d_mrlFocus_ctrlDist['relDirDists'][trialTypes[tt]])):
                nSpikesPerPlatform = len(np.where(spikePlats-1 == p)[0])
                if nSpikesPerPlatform == 0:
                    continue
                
                platDist_Rel = d_mrlFocus_ctrlDist['relDirDists'][trialTypes[tt]][p] * nSpikesPerPlatform
                platDist_Pure = d_mrlFocus_ctrlDist['purDirDists'][trialTypes[tt]][p] * nSpikesPerPlatform
                
                if len(totalDist_Rel) == 0:
                    totalDist_Rel = platDist_Rel
                    totalDist_Pure = platDist_Pure
                else:
                    totalDist_Rel += platDist_Rel
                    totalDist_Pure += platDist_Pure
            
            totalDist_permute = np.transpose(totalDist_Rel, (2, 1, 0))
            
            dirRel2Goal_histCounts = dirHistCounts(spikePos, spikeHD, xAxis, yAxis, angleEdges, histBinCentres)
            normDist = dirRel2Goal_histCounts/totalDist_permute
            sumNormDist = np.sum(normDist, axis=0, keepdims=True)
            normDistFactor = len(spikeHD) / sumNormDist
            normDist = normDist * normDistFactor
            
            d_mrlNorm = mrlRelDir(normDist, xAxis, yAxis, histBinCentres)
            
            d_mrlFocus[k][trialTypes[tt]]['norm'] = d_mrlNorm
            
            hdCountsPure, _ = np.histogram(spikeHD, angleEdges)
            hdCountsNorm = hdCountsPure / totalDist_Pure
            hdCountsNorm = len(spikeHD) * hdCountsNorm / hdCountsNorm.sum()
            
            mrl = circ_r(histBinCentres, hdCountsNorm)
            direction = np.rad2deg(circ_mean(histBinCentres, hdCountsNorm))
            if direction < 0:
                direction += 360
            pval, z = circ_rtest(histBinCentres, hdCountsNorm)
            
            d_mrlFocus[k][trialTypes[tt]]['pureNorm'] = {}
            d_mrlFocus[k][trialTypes[tt]]['pureNorm']['mrl'] = mrl
            d_mrlFocus[k][trialTypes[tt]]['pureNorm']['dir'] = direction
            d_mrlFocus[k][trialTypes[tt]]['pureNorm']['distribution'] = hdCountsNorm
            d_mrlFocus[k][trialTypes[tt]]['pureNorm']['pval'] = pval
            d_mrlFocus[k][trialTypes[tt]]['pureNorm']['z'] = z
            
            print(time.time() - t0)
            
            # shuffles
            mrlVal_norm = []
            for sh in range(len(nShuffles)):
                nShufflesTemp = nShuffles[sh] - len(mrlVal_norm)
                mrlVal_temp = np.zeros((nShufflesTemp, ))
                
                if nShufflesTemp < 200:
                    sigLevel = int(np.floor(200/20))
                elif nShufflesTemp == 200:
                    sigLevel = 20
                else:
                    sigLevel = int(np.floor(1000/20))
                
                for s in range(nShufflesTemp):
                    spikeHDtemp = spikeHD
                    spikeHDtemp = np.random.permutation(spikeHDtemp)
                
                    dirRel2Goal_histCounts = dirHistCounts(spikePos, spikeHDtemp, xAxis, yAxis, angleEdges, histBinCentres)
                    normDist = dirRel2Goal_histCounts/totalDist_permute
                    sumNormDist = np.sum(normDist, axis=0, keepdims=True)
                    normDistFactor = len(spikeHD) / sumNormDist
                    normDist = normDist * normDistFactor
                    
                    mrlNorm_shuffle = mrlRelDir(normDist, xAxis, yAxis, histBinCentres)
                    mrlVal_temp[s] = mrlNorm_shuffle['mrl']
                
                mrlVal_norm.extend(mrlVal_temp)
                mrlVal_norm = sorted(mrlVal_norm, reverse=True)
                
                if d_mrlNorm['mrl'] < mrlVal_norm[sigLevel-1]:
                    sigLevel = int(np.floor(len(mrlVal_norm)/20))
                    d_mrlFocus[k][trialTypes[tt]]['norm']['CI95'] = mrlVal_norm[sigLevel-1]
                    break
                
                if nShuffles[sh] == 1000:
                    d_mrlFocus[k][trialTypes[tt]]['norm']['CI95'] = mrlVal_norm[49]
                    d_mrlFocus[k][trialTypes[tt]]['norm']['CI97.5'] = mrlVal_norm[24]
                    d_mrlFocus[k][trialTypes[tt]]['norm']['CI99.9'] = mrlVal_norm[0] # 1000 sorted values...
                
                    if d_mrlNorm['mrl'] > d_mrlFocus[k][trialTypes[tt]]['norm']['CI95']:
                        d_signifCellsNorm[trialTypes[tt]].append(k)
        if verbose:
            print(f'cell {k} done!')
    
    if savedir is not None:
        with open(savedir, 'wb') as f:
            pickle.dump((d_mrlFocus, angleEdges, histBinCentres, d_signifCellsNorm), f)
        f.close()
    
    return d_mrlFocus, angleEdges, histBinCentres, d_signifCellsNorm

def dirHistCounts(pos, hd, xAxis, yAxis, angleEdges, histBinCentres):
    xDistance = xAxis.reshape(-1, 1) - pos[:, [0]].T
    yDistance = yAxis.reshape(-1, 1) - pos[:, [1]].T
    
    if len(xDistance) > 1:
        xDistance = xDistance.reshape((len(xAxis), 1, len(hd)))
        xDistance = np.tile(xDistance, (1, len(yAxis), 1))
        
        yDistance = yDistance.reshape((1, len(yAxis), len(hd)))
        yDistance = np.tile(yDistance, (len(xAxis), 1, 1))
        
    dir2Goal = np.rad2deg(np.arctan2(xDistance, yDistance))
    dir2Goal[dir2Goal<0] += 360
    
    if len(xAxis) == 1:
        dirRel2Goal = hd - dir2Goal.reshape(1, -1)
        dirRel2Goal[dirRel2Goal<0] += 360
        histCounts, _ = np.histogram(dirRel2Goal, angleEdges)
    else:
        spikeHD_XP = hd.reshape((1, 1, len(hd)))
        spikeHD_XP = np.tile(spikeHD_XP, (len(xAxis), len(yAxis), 1))
        dirRel2Goal = spikeHD_XP - dir2Goal
        dirRel2Goal = np.transpose(dirRel2Goal, (2, 1, 0))
        dirRel2Goal[dirRel2Goal<0] += 360
        histCounts = np.zeros((len(histBinCentres), dirRel2Goal.shape[1], dirRel2Goal.shape[2]))
        for x in range(len(xAxis)):
            for y in range(len(yAxis)):
                histCounts_temp, _ = np.histogram(dirRel2Goal[:, y, x], angleEdges)
                histCounts[:, y, x] = histCounts_temp
    
    return histCounts

def mrlRelDir(histCounts, xAxis, yAxis, histBinCentres):
    d_mrl = {}
    histBinCenRep = np.tile(np.reshape(histBinCentres, (len(histBinCentres), 1, 1)), (1, len(yAxis), len(xAxis)))
    mrl = circ_r(histBinCenRep, histCounts)
    mrlMax = np.max(mrl)
    mrlInd = np.argmax(mrl)
    mrlMaxCoor = np.array([mrlInd // mrl.shape[1], mrlInd % mrl.shape[1]])
    d_mrl['mrl'] = mrlMax
    
    direction = circ_mean(histBinCenRep, histCounts)
    d_mrl['dir'] = np.rad2deg(direction[mrlMaxCoor[0], mrlMaxCoor[1]])
    
    z, pval = circ_rtest(histBinCentres, histCounts[:, mrlMaxCoor[0], mrlMaxCoor[1]])
    d_mrl['distribution'] = histCounts[:, mrlMaxCoor[0], mrlMaxCoor[1]]
    
    mrlMaxCoor = np.array([yAxis[mrlMaxCoor[0]], xAxis[mrlMaxCoor[1]]])
    d_mrl['coor'] = mrlMaxCoor[::-1]
    d_mrl['pval'] = pval
    d_mrl['z'] = z
    
    return d_mrl

if __name__=="__main__":
    pos_fdir = '/Users/changminyu/Desktop/research/data/Rat7/6-12-2019/positionalData/positionalDataByTrialType.mat'
    platformLoc_fdir = '/Users/changminyu/Desktop/research/data/Rat7/6-12-2019/positionalData/platformLocations.mat'
    rateMaps_fdir = '/Users/changminyu/Desktop/research/data/Rat7/6-12-2019/physiologyData/placeFieldData/rateMaps.mat'
    mrlFocus_ctrlDist_fdir = '/Users/changminyu/Desktop/research/data/Rat7/6-12-2019/physiologyData/direction/mrlFocus_ctrlDistribution_coarse.mat'
    
    savedir = 'data/Rat7/6-12-2019/mrlFocus_spikes_test.pkl'
    
    _ = relativeDirectionSpikes(pos_fdir, platformLoc_fdir, rateMaps_fdir, mrlFocus_ctrlDist_fdir, savedir, verbose=True)
    