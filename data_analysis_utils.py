from typing import List, Optional, Union

import pickle

import numpy as np
from scipy.io import loadmat

import os

from data_utils import (
    load_pos_mat, 
    load_running_gaps_mat, 
)


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


def active_scanning_behavioural(
    pos_mat_log_dir: str, 
    running_gaps_mat_log_dir: str, 
    trial_types: List[str] = ["hComb", "openF"], 
    angular_speed_threshold_quantile: float = 0.8, 
    angular_gaps: int = 0,
    cache_dir: Optional[str] = None, 
    rat_id: int = 7, 
    session_id: str = "6-12-2019",
    scanning_period_duration_threshold: float = 0.0, # in ms
):
    if cache_dir is not None:
        cache_dir = os.path.join(
            cache_dir, 
            "active_scanning_behavioural", 
            f"angular_gaps_{angular_gaps}",
            f"Rat{rat_id}", 
            session_id,
        )
        
        identifier = f"angular_speed_threshold_{angular_speed_threshold_quantile}_duration_threshold_{scanning_period_duration_threshold}"
        
    try:
        with open(os.path.join(cache_dir, f"{identifier}.pkl"), "rb") as f:
            d = pickle.load(f)
        f.close()
        
        print("Retrieved cached behavioural data during active scanning!")
    
    except Exception:
        d_pos = load_pos_mat(pos_mat_log_dir)
        d_running_gaps = load_running_gaps_mat(
            running_gaps_mat_log_dir, 
        )
        
        if "ripple_and_no_theta" not in d_running_gaps:
            for i in range(len(trial_types)):
                no_theta = d_running_gaps[trial_types[i]]["noTheta"]
                ripples = d_running_gaps[trial_types[i]]["ripples"]
                
                d_running_gaps[trial_types[i]]["ripple_and_no_theta"] = []
                
                num_trials = len(no_theta)
                
                for j in range(num_trials):
                    ripple_and_no_theta = merge_time_periods(no_theta[j], ripples[j])
                    d_running_gaps[trial_types[i]]["ripple_and_no_theta"].append(ripple_and_no_theta)
        
        goal_location = d_pos["goalPosition"][0][0][0]
        
        behavioural_keys = ["samples", "loc", "hd", "relative_direction_to_goal", "clockwise_flags", "start_end_samples"]
        
        sampling_frequency = 30000 if rat_id not in [3] else 20000
        
        d_active_scanning = {}
        d_non_active_scanning = {}
        for k in behavioural_keys:
            d_active_scanning[k] = {}
            d_non_active_scanning[k] = {}
            
        for i in range(len(trial_types)):
            for k in behavioural_keys:
                d_active_scanning[k][trial_types[i]] = {}
                d_non_active_scanning[k][trial_types[i]] = {}
                
            samples = d_pos["data"][trial_types[i]]["sample"]
            loc = d_pos["data"][trial_types[i]]["dlc_XYsmooth"]
            hd = d_pos["data"][trial_types[i]]["dlc_angle"]
            direction_to_goal = d_pos["data"][trial_types[i]]["dir2goal"]
            
            high_direction_change_clockwise, high_direction_change_anticlockwise = continuous_high_angular_change(
                hd, 
                angular_speed_threshold_quantile, 
                angular_gaps, 
            )
            
            num_trials = len(samples)
            
            for j in range(num_trials):
                goal_vec = goal_location - loc[j]
                direction_to_goal[j] = np.arctan2(
                    goal_vec[:, 1],
                    goal_vec[:, 0],
                ).reshape(-1, 1)
            
            for j in range(num_trials): 
                running_gap_samples = d_running_gaps[trial_types[i]]["ripple_and_no_theta"][j]
                
                relative_direction_to_goal = angular_difference(
                    hd[j][:, 0], 
                    np.rad2deg(direction_to_goal[j][:, 0]), 
                )
                
                if len(high_direction_change_clockwise[j]) > 0:
                    clockwise_diff_duration = (samples[j][high_direction_change_clockwise[j][:, 1], 0] - samples[j][high_direction_change_clockwise[j][:, 0], 0]) / sampling_frequency * 1000.0 # in ms
                    high_direction_change_clockwise[j] = high_direction_change_clockwise[j][
                        clockwise_diff_duration >= scanning_period_duration_threshold
                    ]
                else:
                    high_direction_change_clockwise[j] = np.empty((0, 2), dtype=np.int32)
                if len(high_direction_change_anticlockwise[j]) > 0:
                    anticlockwise_diff_duration = (samples[j][high_direction_change_anticlockwise[j][:, 1], 0] - samples[j][high_direction_change_anticlockwise[j][:, 0], 0]) / sampling_frequency * 1000.0 # in ms
                    
                    high_direction_change_anticlockwise[j] = high_direction_change_anticlockwise[j][
                        anticlockwise_diff_duration >= scanning_period_duration_threshold
                    ]
                else:
                    high_direction_change_anticlockwise[j] = np.empty((0, 2), dtype=np.int32)
                
                high_direction_change_all = np.concatenate([
                    high_direction_change_clockwise[j], 
                    high_direction_change_anticlockwise[j]
                ], axis=0).astype(np.int32)
                
                high_direction_change_all_samples = np.array([
                    samples[j][:, 0][high_direction_change_all[:, 0]], 
                    samples[j][:, 0][high_direction_change_all[:, 1]],
                    np.array([
                        ["clockwise"] * len(high_direction_change_clockwise[j]) +
                        ["anticlockwise"] * len(high_direction_change_anticlockwise[j])
                    ])[0]
                ]).T
                high_direction_change_all_samples = high_direction_change_all_samples[
                    np.argsort(high_direction_change_all_samples[:, 0])
                ]
                
                d_active_scanning["start_end_samples"][trial_types[i]][j] = high_direction_change_all_samples
                
                active_scanning_periods_all = [
                    np.arange(high_direction_change_clockwise[j][k, 0], high_direction_change_clockwise[j][k, 1] + 1)
                    for k in range(len(high_direction_change_clockwise[j]))
                ] + [
                    np.arange(high_direction_change_anticlockwise[j][k, 0], high_direction_change_anticlockwise[j][k, 1] + 1)
                    for k in range(len(high_direction_change_anticlockwise[j]))
                ]
                if len(active_scanning_periods_all) > 0:
                    active_scanning_periods_all = np.concatenate(active_scanning_periods_all, axis=0)
                
                if len(active_scanning_periods_all) > 0:
                    non_active_periods = np.setdiff1d(
                        np.arange(samples[j].shape[0]),
                        active_scanning_periods_all,
                    )
                else:
                    non_active_periods = np.arange(samples[j].shape[0])
                
                non_active_periods_start_end = retrieve_continuous_subseq(
                    np.sort(non_active_periods), 
                    gaps=0, 
                )
                non_active_periods_start_end_samples = np.array([
                    samples[j][:, 0][non_active_periods_start_end[:, 0]], 
                    samples[j][:, 0][non_active_periods_start_end[:, 1]],
                ]).T
                d_non_active_scanning["start_end_samples"][trial_types[i]][j] = non_active_periods_start_end_samples
                
                non_active_samples = samples[j][non_active_periods, 0]
                if len(running_gap_samples) > 0:
                    _, gap_inds = restrict_spikes(
                        non_active_samples, running_gap_samples[:, 0], running_gap_samples[:, 1]
                    )
                else:
                    gap_inds = []
                
                if len(gap_inds) > 0:
                    non_active_periods = np.delete(non_active_periods, gap_inds)
                
                clockwise_flags = np.array(
                    ["clockwise"] * len(high_direction_change_clockwise[j]) + 
                    ["anticlockwise"] * len(high_direction_change_anticlockwise[j])
                )
                
                d_non_active_scanning["samples"][trial_types[i]][j] = samples[j][non_active_periods, 0]
                d_non_active_scanning["loc"][trial_types[i]][j] = loc[j][non_active_periods]
                d_non_active_scanning["hd"][trial_types[i]][j] = hd[j][non_active_periods, 0]
                d_non_active_scanning["relative_direction_to_goal"][trial_types[i]][j] = relative_direction_to_goal[non_active_periods]

                if len(high_direction_change_all) == 0:
                    continue
                
                valid_periods_sample = extract_valid_periods_excluding_gaps(
                    samples[j][0, 0], 
                    samples[j][-1, 0], 
                    running_gap_samples
                )
                
                samples_temp = []
                loc_temp = []
                hd_temp = []
                relative_direction_to_goal_temp = []
                clockwise_flags_temp = []
                for l in range(len(high_direction_change_all)):
                    samples_temp_temp = samples[j][high_direction_change_all[l, 0]:(high_direction_change_all[l, 1]+1), 0]
                    if np.sum((valid_periods_sample[:, 0] <= samples_temp_temp[0]) * (valid_periods_sample[:, 1] >= samples_temp_temp[-1])) == 0:
                        continue
                    
                    samples_temp.append(
                        samples[j][high_direction_change_all[l, 0]:(high_direction_change_all[l, 1]+1), 0]
                    )
                    loc_temp.append(
                        loc[j][high_direction_change_all[l, 0]:(high_direction_change_all[l, 1]+1)]
                    )
                    hd_temp.append(
                        hd[j][high_direction_change_all[l, 0]:(high_direction_change_all[l, 1]+1), 0]
                    )
                    relative_direction_to_goal_temp.append(
                        relative_direction_to_goal[high_direction_change_all[l, 0]:(high_direction_change_all[l, 1]+1)]
                    )
                    clockwise_flags_temp.append([clockwise_flags[l]] * (high_direction_change_all[l, 1]+1-high_direction_change_all[l, 0]))
                if len(samples_temp) >0:
                    d_active_scanning["samples"][trial_types[i]][j] = np.concatenate(samples_temp, axis=0)
                    d_active_scanning["loc"][trial_types[i]][j] = np.concatenate(loc_temp, axis=0)
                    d_active_scanning["hd"][trial_types[i]][j] = np.concatenate(hd_temp, axis=0)
                    d_active_scanning["relative_direction_to_goal"][trial_types[i]][j] = np.concatenate(
                        relative_direction_to_goal_temp, axis=0, 
                    )
                    d_active_scanning["clockwise_flags"][trial_types[i]][j] = np.concatenate(clockwise_flags_temp, axis=0)
                else:
                    d_active_scanning["samples"][trial_types[i]][j] = np.array([])
                    d_active_scanning["loc"][trial_types[i]][j] = np.array([])
                    d_active_scanning["hd"][trial_types[i]][j] = np.array([])
                    d_active_scanning["relative_direction_to_goal"][trial_types[i]][j] = np.array([])
                    d_active_scanning["clockwise_flags"][trial_types[i]][j] = np.array([])

        d = {}
        d["active_scanning"] = d_active_scanning
        d["non_active_scanning"] = d_non_active_scanning
        if cache_dir is not None:
            if not os.path.exists(cache_dir):
                os.makedirs(cache_dir)
            
            with open(os.path.join(cache_dir, f"{identifier}.pkl"), "wb") as f:
                pickle.dump(d, f)
        
        print("Constructed behavioural data of active scanning periods from scratch!")
    
    return d


def merge_time_periods(array1: np.ndarray, array2: np.ndarray):
    # Combine the two arrays
    all_periods = np.vstack((array1, array2))
    
    if len(all_periods) == 0:
        return np.zeros((0, 2))
    
    # Sort the combined array by start times
    sorted_periods = all_periods[np.argsort(all_periods[:, 0])]
    
    # Initialize the merged array with the first period
    merged = [sorted_periods[0]]
    
    # Iterate through the sorted periods
    for current in sorted_periods[1:]:
        previous = merged[-1]
        
        # If the current period overlaps with the previous one, merge them
        if current[0] <= previous[1]:
            merged[-1][1] = max(previous[1], current[1])
        else:
            # If there's no overlap, add the current period to the merged list
            merged.append(current)
    
    return np.array(merged)


def continuous_high_angular_change(
    hd: List[np.ndarray], 
    threshold_quantile: float = 0.8, 
    gaps: int = 0, 
):
    """
    Find continuous subsequence with high angular speed
    
    Parameters
    ----------
    hd: np.ndarray
        head direction
    threshold_quantile: float
        thresholding quantile for angular speed
    gaps: int
        gaps
    """
    
    direction_change = [
        angular_difference(hd_trial[:-1, 0], hd_trial[1:, 0], wrap=False) 
        for hd_trial in hd
    ]
    
    threshold = np.quantile(np.abs(np.concatenate(direction_change, axis=0)), 
                            threshold_quantile)
    
    high_direction_change_period_anticlockwise = []
    high_direction_change_period_clockwise = []
    for i in range(len(direction_change)):
        inds_anticlockwise = np.where(direction_change[i] > threshold)[0]
        inds_clockwise = np.where(-direction_change[i] > threshold)[0]
        
        continuous_subseq_anticlockwise = retrieve_continuous_subseq(inds_anticlockwise, gaps)
        continuous_subseq_clockwise = retrieve_continuous_subseq(inds_clockwise, gaps)
        
        if len(continuous_subseq_anticlockwise) == 0:
            high_direction_change_period_anticlockwise.append(np.array([]).reshape(-1, 2))
        else:
            high_direction_change_period_anticlockwise.append(
                continuous_subseq_anticlockwise
            )
        if len(continuous_subseq_clockwise) == 0:
            high_direction_change_period_clockwise.append(np.array([]).reshape(-1, 2))
        else:
            high_direction_change_period_clockwise.append(
                continuous_subseq_clockwise
            )
        
    return high_direction_change_period_clockwise, high_direction_change_period_anticlockwise


def angular_difference(angles1, angles2, wrap=True):
    """
    Compute the difference between two arrays of angular variables (in degrees).
    
    Parameters:
    - angles1: array of angles (in degrees)
    - angles2: array of angles (in degrees)
    
    Returns:
    - diff: array of angular differences (in degrees), constrained to [0, 360]
    """
    
    # Convert angles from degrees to radians
    angles1_rad = np.radians(angles1)
    angles2_rad = np.radians(angles2)
    
    # Compute the difference
    diff_rad = angles1_rad - angles2_rad
    
    # Adjust the differences to be within the range [-pi, pi]
    diff = np.rad2deg((diff_rad + np.pi) % (2 * np.pi) - np.pi)
    
    if wrap:
        diff = wrap_angles(diff)
        
    return diff


def wrap_angles(angles):
    """
    Wrap angles from the range [-180, 180] to [0, 360].
    
    Parameters:
    - angles: array or single value of angles (in radians)
    
    Returns:
    - wrapped_angles: array or single value of angles (in degrees), wrapped to [0, 360]
    """
    wrapped_angles = (angles + 360) % 360
    return wrapped_angles


def retrieve_continuous_subseq(x: Union[List[int], np.ndarray], gaps: int = 0):
    """
    Find length of continuous subsequences
    Optionally include continuous subsequences that are separated by
    less than `gaps` away
    
    Parameters
    ----------
    x: Union[List[int], np.ndarray]
        list of indices in sub-sequences
    gaps: int
        gaps
    """
    if len(x) == 0:
        return np.array([])
    x = np.array(x)
    if gaps > 1:
        split_indices = np.where(np.diff(x) > gaps)[0]
    else:
        split_indices = np.where(np.diff(x) > 1)[0]
    
    subsequences = []
    start_num = x[0]
    for ind in split_indices:
        end_num = x[ind]
        subsequences.append([start_num, end_num])
        start_num = x[ind + 1]
    
    subsequences.append([start_num, x[-1]])

    return np.array(subsequences)


def extract_valid_periods_excluding_gaps(
    start_sample: float, 
    stop_sample: float, 
    running_gaps: np.ndarray, 
):
    if len(running_gaps) == 0:
        return np.array([[start_sample, stop_sample]])

    overlapping_gaps = []
    for gap_start, gap_end in running_gaps:
        if gap_start < stop_sample and gap_end > start_sample:
            overlapping_gaps.append([
                max(gap_start, start_sample), 
                min(gap_end, stop_sample), 
            ])
    if len(overlapping_gaps) == 0:
        return np.array([[start_sample, stop_sample]])
    
    overlapping_gaps = np.array(overlapping_gaps)
    valid_periods = []
    if overlapping_gaps[0, 0] > start_sample:
        valid_periods.append([start_sample, overlapping_gaps[0, 0]])
    
    for i in range(len(overlapping_gaps) - 1):
        gap_end = overlapping_gaps[i, 1]
        next_gap_start = overlapping_gaps[i + 1, 0]
        if gap_end < next_gap_start:
            valid_periods.append([gap_end, next_gap_start])
    
    if overlapping_gaps[-1, 1] < stop_sample:
        valid_periods.append([overlapping_gaps[-1, 1], stop_sample])
    
    return np.array(valid_periods)


def restrict_spikes(
    spikes: np.ndarray, 
    start_inds: np.ndarray, 
    stop_inds: np.ndarray, 
    flatten: bool = False, 
):
    """
    Restrict spikes within specified windows
    
    Parameters
    ----------
    start_inds: np.ndarray
        start index
    stop_inds: np.ndarray
        end index
    """
    if len(spikes.shape) > 1:
        if spikes.shape[0] < spikes.shape[1]:
            spikes = spikes.T
    
    assert len(start_inds) == len(stop_inds)
    
    rest_inds = []
    rest_spikes = []
    
    for i in range(len(start_inds)):
        rest_inds.append(np.where((spikes <= stop_inds[i]) * (spikes >= start_inds[i]))[0])
        rest_spikes.append(spikes[rest_inds[i]])

    if not flatten:
        rest_inds = np.concatenate(rest_inds, axis=0)
        rest_spikes = np.concatenate(rest_spikes, axis=0)
    
    return rest_spikes, rest_inds
