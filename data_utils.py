from typing import List

import os
from scipy.io import loadmat 
import numpy as np


def load_pos_mat(
    load_dir: str, 
    trial_types: List[str] = ['hComb', 'openF'],
):
    """
    Load .mat file with positional information
    
    Parameters
    ----------
    load_dir: str
        loading directory
    trial_types: List[str]
        trials to look at
    """
    if not os.path.exists(load_dir):
        raise ValueError
    mat = loadmat(load_dir)
    pos = mat['pos'][0, 0] # first level (num_trials, )
    num_trial_types = len(pos)
    assert len(trial_types) == num_trial_types
    
    # dictionary containing useful data
    d = {}
    
    d['goalPosition'] = mat['goalPosition']
    try:
        d["goalID"] = mat["goalID"]
    except KeyError:
        d["goalID"] = None
    d['frameSize'] = mat['frameSize'][0]
    d["data"] = {}
    
    for i in range(num_trial_types):
        num_trials = pos[i].shape[1]
        d["data"][trial_types[i]] = {}
        # all recorded attributed
        all_attributes = np.array(list(np.dtype(pos[i].dtype).names))
        for j in range(len(all_attributes)):
            d["data"][trial_types[i]][all_attributes[j]] = []
            for k in range(num_trials):
                d["data"][trial_types[i]][all_attributes[j]].append(pos[i][0, k][j])
    return d


def compute_smoothed_speed(
    pos: np.ndarray,
    samples: np.ndarray, 
    raw_freq: float, 
    smoothing_window: int = 20,
):
    """
    Compute instantaneous speed with noise reduction.
    
    Args:
        pos_x, pos_y: Position coordinates
        ts: Timestamps
        smoothing_window: Window size for moving average (odd number)
        speed_thresh: Optional speed threshold to remove outliers (in units/sec)
    """
    # Compute raw speed
    samples_ts = samples[:, 0] / raw_freq
    dt = np.median(np.diff(samples_ts))
    
    pos_x = pos[:, 0]
    pos_y = pos[:, 1]
    
    dx = np.diff(pos_x)
    dy = np.diff(pos_y)
    dt = np.diff(samples_ts)
    
    # Calculate raw speed
    speed = np.zeros_like(pos_x)
    speed[1:] = np.sqrt(dx**2 + dy**2) / dt
    speed[0] = speed[1]
    
    # Smooth with moving average
    kernel = np.ones(smoothing_window) / smoothing_window
    speed_smooth = np.convolve(speed, kernel, mode='same')
    
    return speed_smooth


def compute_smoothed_movement_direction(
    pos: np.ndarray, 
    samples: np.ndarray, 
    raw_freq: float, 
    smoothing_window: int = 5, 
):
    # return smoothed movement directions (in radians)
    pos_x, pos_y = pos[:, 0], pos[:, 1]
    ts = samples[:, 0] / raw_freq
    
    dx = np.diff(pos_x)
    dy = np.diff(pos_y)
    dt = np.diff(ts)

    vx = dx/dt
    vy = dy/dt

    # Smooth velocities
    kernel = np.ones(smoothing_window)/smoothing_window
    vx_smooth = np.convolve(vx, kernel, mode='same')
    vy_smooth = np.convolve(vy, kernel, mode='same')

    # Pad first element
    vx_smooth = np.concatenate(([vx_smooth[0]], vx_smooth))
    vy_smooth = np.concatenate(([vy_smooth[0]], vy_smooth))

    # Compute speed and heading

    movement_direction = np.arctan2(vy_smooth, vx_smooth)
    
    return movement_direction


def load_running_gaps_mat(
    load_dir: str, 
    trial_types: List[str] = ['hComb', 'openF'],
):
    """
    Load .mat file with running gap (low theta power) information
    
    Parameters
    ----------
    load_dir: str
        loading directory
    trial_types: List[str]
        trials to look at
    """
    if not os.path.exists(load_dir):
        raise ValueError
    mat = loadmat(load_dir)
    data = mat['runningGaps'][0, 0]
    
    num_trial_types = len(data)
    assert len(trial_types) == num_trial_types
    d = {}
    
    for i in range(len(trial_types)):
        d[trial_types[i]] = {}
        num_trials = len(data[i][0])
        all_attributes = np.array(list(np.dtype(data[i].dtype).names))
        for j in range(len(all_attributes)):
            if all_attributes[j] not in d[trial_types[i]]:
                d[trial_types[i]][all_attributes[j]] = []
            for k in range(num_trials):
                d[trial_types[i]][all_attributes[j]].append(data[i][0, k][j].reshape(-1, 2))
    return d
