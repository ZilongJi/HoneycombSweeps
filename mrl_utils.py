from typing import List, Optional, Dict, Any

import numpy as np

import pickle
import time

import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '../'))

from utils.data_loader import (  # noqa: E402
    load_pos_mat, 
    load_platform_location_mat, 
    load_mrl_distribution_mat, 
    load_ratemaps_mat, 
)
from utils.decoding_utils import construct_ratemaps_pos  # noqa: E402
from utils.circular_utils import (  # noqa: E402
    circular_mean_resultant_vector_length, 
    circular_mean, 
    circular_rayleigh_test, 
)


ALL_RESOLUTIONS = ["ultraLow", "lowRes", "highRes", "ultraHigh"]


def spike_mean_relative_direction(
    pos_mat_log_dir: Optional[str] = None, 
    platform_location_mat_log_dir: Optional[str] = None, 
    mrl_focus_dist_mat_log_dir: Optional[str] = None, 
    ratemaps_log_dir: Optional[str] = None, 
    trial_types: List[str] = ["hComb", "openF"], 
    cache_dir: Optional[str] = None, 
    rat_id: int = 7, 
    session_id: str = "6-12-2019", 
    frame_bin_size: int = 15, 
    platform_key: str = "body", 
    ratemaps_kwargs: Dict[str, Any] = {}, 
    verbose: bool = False, 
):
    """
    Compte mean relatie direction associated with spikes
    
    Parameters
    ----------
    pos_mat_log_dir: str
        log directory for position
    platform_location_mat_log_dir: str
        log directory for platform location
    mrl_focus_dist_mat_log_dir: str
        log directory for direction distribution on each platform
    ratemaps_log_dir: str
        log directory for ratemaps
    trial_types: List[str]
        trial types
    cache_dir: Optional[str]
        cache directory
    rat_id: int
        rat ID
    session_id: str
        session ID
    frame_bin_size: int
        frame bin size
    platform_key: str
        platform key
    ratemaps_kwargs: Dict[str, Any]
        parameters for constructing ratemaps
    verbose: bool
        verbose
    """
    
    try:
        with open(os.path.join(cache_dir, "spikes_mrl.pkl"), "rb") as f:
            d_spikes_mrl = pickle.load(f)
        f.close()
        
        print("Retrieved cached spikes mean relative directions!")
    
    except Exception:
        t0 = time.time()
        d_pos = load_pos_mat(pos_mat_log_dir, trial_types)
        d_platform_location = load_platform_location_mat(platform_location_mat_log_dir)
        if ".mat" in ratemaps_log_dir:
            d_ratemaps = load_ratemaps_mat(ratemaps_log_dir)
        else:
            d_ratemaps = construct_ratemaps_pos(
                log_dir=ratemaps_log_dir, 
                rat_id=rat_id, 
                session_id=session_id, 
                **ratemaps_kwargs, 
            )
        d_mrl_distributions = load_mrl_distribution_mat(mrl_focus_dist_mat_log_dir)
        
        d_spikes_mrl = {}
        
        angle_bin_edges = d_mrl_distributions["angle_edges"]
        angle_bin_centres = np.deg2rad((angle_bin_edges[1] - angle_bin_edges[0]) / 2 + 
                                       angle_bin_edges[:-1])
        d_spikes_mrl["angle_bin_edges"] = angle_bin_edges
        d_spikes_mrl["angle_bin_centres"] = angle_bin_centres
        
        frame_size = d_pos["frameSize"]
        x_bin_edges = np.arange(1, frame_size[0]+1, frame_bin_size)
        y_bin_edges = np.arange(1, frame_size[1]+1, frame_bin_size)
        
        n_shuffles = [50, 100, 200, 500, 1000]
        
        d_spikes_mrl["mrl"] = {}
        d_spikes_mrl["significant_cells_norm"] = {}
        
        for i in range(len(trial_types)):
            d_spikes_mrl["mrl"][trial_types[i]] = {}
            d_spikes_mrl["significant_cells_norm"][trial_types[i]] = []
            
            samples = np.concatenate(d_pos["data"][trial_types[i]]["sample"], axis=0)[:, 0]
            platform_locations = np.concatenate(
                d_platform_location[trial_types[i]][platform_key], axis=0
            )[:, 0]
            
            for c in d_ratemaps["ratemap"]:
                if d_ratemaps["ratemap"][c]["neuron_type"] not in ["pyramid", "p"]:
                    continue
                
                if ".mat" in ratemaps_log_dir:
                    # TODO: something is off with computing the spike head-direction with interpolation
                    # check! (it is mostly correct but sometimes it's off)
                    spike_pos = d_ratemaps["ratemap"][c][trial_types[i]]["spikePos"]
                    spike_hd = d_ratemaps["ratemap"][c][trial_types[i]]["spikeHD"][:, 0]
                    spike_samples = d_ratemaps["ratemap"][c][trial_types[i]]["spikeSamples"]
                else:
                    spike_pos = d_ratemaps["ratemap"][c][trial_types[i]]["spike_pos"]
                    spike_hd = d_ratemaps["ratemap"][c][trial_types[i]]["spike_hd"][:, 0]
                    spike_hd[spike_hd < 0] += 360
                    spike_samples = d_ratemaps["ratemap"][c][trial_types[i]]["spike_samples"]
                
                if len(spike_samples) < 500:
                    continue
                
                d_spikes_mrl["mrl"][trial_types[i]][c] = {}
                
                platform_ind = np.round(
                    np.interp(spike_samples, samples, np.arange(len(samples)))
                ).astype(np.int32)
                del spike_samples
                spike_platforms = platform_locations[platform_ind]
                
                relative_direction_distribution_aggregated = []
                pure_direction_distribution_aggregated = []
                
                n_platforms = len(d_mrl_distributions["pure_direction_distribution"][trial_types[i]])
                
                for p in range(n_platforms):
                    n_spikes_per_platform = len(np.where(spike_platforms == (p + 1))[0])

                    if n_spikes_per_platform == 0:
                        continue
                        
                    relative_direction_distribution_platform = d_mrl_distributions\
                        ["relative_direction_distribution"][trial_types[i]][p] * n_spikes_per_platform
                    pure_direction_distribution_platform = d_mrl_distributions\
                        ["pure_direction_distribution"][trial_types[i]][p] * n_spikes_per_platform
                    
                    if len(relative_direction_distribution_aggregated) == 0:
                        relative_direction_distribution_aggregated = relative_direction_distribution_platform
                        pure_direction_distribution_aggregated = pure_direction_distribution_platform
                    else:
                        relative_direction_distribution_aggregated += relative_direction_distribution_platform
                        pure_direction_distribution_aggregated += pure_direction_distribution_platform
                
                relative_direction_distribution_aggregated = np.transpose(
                    relative_direction_distribution_aggregated, (2, 1, 0), 
                )
                
                direction_relative_to_goals_histogram_counts = direction_histogram_counts(
                    spike_pos, spike_hd, x_bin_edges, y_bin_edges, angle_bin_edges, 
                )
                normalised_distribution = direction_relative_to_goals_histogram_counts / \
                    relative_direction_distribution_aggregated
                normalised_distribution *= len(spike_hd) / np.sum(normalised_distribution, axis=0, keepdims=True)
                
                d_mrl_normalised = mrl_relative_direction(
                    normalised_distribution, x_bin_edges, y_bin_edges, angle_bin_centres, 
                )
                
                d_spikes_mrl["mrl"][trial_types[i]][c]["normalised"] = d_mrl_normalised
                
                pure_direction_histogram_counts, _ = np.histogram(spike_hd, angle_bin_edges)
                normalised_pure_direction_counts = pure_direction_histogram_counts / \
                    pure_direction_distribution_aggregated
                normalised_pure_direction_counts *= len(spike_hd) / normalised_pure_direction_counts.sum()
                
                normalised_pure_mrl = circular_mean_resultant_vector_length(
                    angle_bin_centres, normalised_pure_direction_counts, 
                )
                normalised_pure_direction = np.rad2deg(circular_mean(
                    angle_bin_centres, normalised_pure_direction_counts
                ))
                if normalised_pure_direction < 0:
                    normalised_pure_direction += 360
                pval, z = circular_rayleigh_test(angle_bin_centres, normalised_pure_direction_counts)
                
                d_spikes_mrl["mrl"][trial_types[i]][c]["pure_normalised"] = {}
                
                d_spikes_mrl["mrl"][trial_types[i]][c]["pure_normalised"]["mrl"] = normalised_pure_mrl
                d_spikes_mrl["mrl"][trial_types[i]][c]["pure_normalised"]["direction"] = \
                    normalised_pure_direction
                d_spikes_mrl["mrl"][trial_types[i]][c]["pure_normalised"]["distribution"] = \
                    normalised_pure_direction_counts
                d_spikes_mrl["mrl"][trial_types[i]][c]["pure_normalised"]["pval"] = pval
                d_spikes_mrl["mrl"][trial_types[i]][c]["pure_normalised"]["z"] = z
                
                print(time.time() - t0)

                # random shuffles
                mrl_shuffles = []
                for n in range(len(n_shuffles)):
                    n_shuffles_temp = n_shuffles[n] - len(mrl_shuffles)
                    mrl_shuffles_temp = np.zeros((n_shuffles_temp, ))
                    
                    if n_shuffles_temp < 200:
                        significant_level = int(np.floor(200 / 20))
                    elif n_shuffles_temp == 200:
                        significant_level = 20
                    else:
                        significant_level = int(np.floor(1000 / 20))
                        
                    for s in range(n_shuffles_temp):
                        spike_hd_shuffled = np.random.permutation(spike_hd)

                        direction_relative_to_goals_histogram_counts_temp = direction_histogram_counts(
                            spike_pos, spike_hd_shuffled, x_bin_edges, y_bin_edges, angle_bin_edges, 
                        )
                        normalised_distribution_temp = direction_relative_to_goals_histogram_counts_temp / \
                            relative_direction_distribution_aggregated
                        normalised_distribution_temp *= len(spike_hd) / np.sum(normalised_distribution_temp, 
                                                                               axis=0, keepdims=True)
                        d_mrl_normalised_shuffled = mrl_relative_direction(
                            normalised_distribution_temp, x_bin_edges, y_bin_edges, angle_bin_centres, 
                        )
                        mrl_shuffles_temp[s] = d_mrl_normalised_shuffled["max_mrl"]
                    
                    mrl_shuffles.extend(mrl_shuffles_temp)
                    mrl_shuffles = sorted(mrl_shuffles, reverse=True)
                    
                    if d_mrl_normalised["max_mrl"] < mrl_shuffles[significant_level - 1]:
                        significant_level = int(np.floor(len(mrl_shuffles) / 20))
                        d_spikes_mrl["mrl"][trial_types[i]][c]["normalised"]["CI95"] = mrl_shuffles[
                            significant_level - 1
                        ]
                        break
                    
                    if n_shuffles[n] == 1000:
                        d_spikes_mrl["mrl"][trial_types[i]][c]["normalised"]["CI95"] = mrl_shuffles[49]
                        d_spikes_mrl["mrl"][trial_types[i]][c]["normalised"]["CI97.5"] = mrl_shuffles[24]
                        d_spikes_mrl["mrl"][trial_types[i]][c]["normalised"]["CI99.9"] = mrl_shuffles[0]
                        
                        if d_mrl_normalised["max_mrl"] > d_spikes_mrl["mrl"][trial_types[i]][c]\
                            ["normalised"]["CI95"]:
                            d_spikes_mrl["significant_cells_norm"][trial_types[i]].append(c)
                
                if verbose:
                    print(f"{trial_types[i]} {c} done!")
        
        print("Constructed MRL from scratch!")
        
        if cache_dir is not None:
            with open(os.path.join(cache_dir, "spikes_mrl.pkl"), "wb") as f:
                pickle.dump(d_spikes_mrl, f)
            f.close()
    return d_spikes_mrl
                

def direction_histogram_counts(
    spike_pos: np.ndarray, 
    spike_hd: np.ndarray, 
    x_bin_edges: np.ndarray, 
    y_bin_edges: np.ndarray, 
    angle_bin_edges: np.ndarray, 
):
    """
    Compute histogram counts for head-direction at spikes
    
    Parameters
    ----------
    spike_pos: np.ndarray
        spike position
    spike_hd: np.ndarray
        spike head direction
    x_bin_edges: np.ndarray
        x-axis bin edges
    y_bin_edges: np.ndarray
        y-axis bin edges
    angle_bin_edges: np.ndarray
        angle bin edges
    """
    
    n_pos_bins_x = len(x_bin_edges)
    n_pos_bins_y = len(y_bin_edges)
    n_samples = len(spike_pos)
    n_angle_bins = len(angle_bin_edges) - 1
    
    x_distance = x_bin_edges.reshape(-1, 1) - spike_pos[:, 0].reshape(1, -1)
    y_distance = y_bin_edges.reshape(-1, 1) - spike_pos[:, 1].reshape(1, -1)
    
    if len(x_distance) > 1:
        x_distance = x_distance.reshape((n_pos_bins_x, 1, n_samples))
        x_distance = np.tile(x_distance, (1, n_pos_bins_y, 1))

        y_distance = y_distance.reshape((1, n_pos_bins_y, n_samples))
        y_distance = np.tile(y_distance, (n_pos_bins_x, 1, 1))
        
    direction_to_goal = np.rad2deg(np.arctan2(x_distance, y_distance))
    direction_to_goal[direction_to_goal < 0] += 360
    
    if n_pos_bins_x == 1:
        direction_relative_to_goal = spike_hd - direction_to_goal
        direction_relative_to_goal[direction_relative_to_goal < 0] += 360
        histogram_counts, _ = np.histogram(direction_relative_to_goal, angle_bin_edges)
    else:
        spike_hd = spike_hd.reshape((1, 1, n_samples))
        spike_hd = np.tile(spike_hd, (n_pos_bins_x, n_pos_bins_y, 1))
        direction_relative_to_goal = spike_hd - direction_to_goal
        direction_relative_to_goal = np.transpose(direction_relative_to_goal, (2, 1, 0))
        direction_relative_to_goal[direction_relative_to_goal < 0] += 360
        
        histogram_counts = np.zeros(
            (n_angle_bins, n_pos_bins_y, n_pos_bins_x)
        )
        for i in range(n_pos_bins_x):
            for j in range(n_pos_bins_y):
                histogram_counts_temp, _ = np.histogram(direction_relative_to_goal[:, j, i], 
                                                        angle_bin_edges)
                histogram_counts[:, j, i] = histogram_counts_temp
    
    return histogram_counts


def mrl_relative_direction(
    histogram_counts: np.ndarray, 
    x_bin_edges: np.ndarray, 
    y_bin_edges: np.ndarray, 
    histogram_bin_centres: np.ndarray
):
    d = {}
    histogram_bin_centres_tiled = np.tile(
        histogram_bin_centres[:, None, None], (1, len(y_bin_edges), len(x_bin_edges)), 
    )
    mrl = circular_mean_resultant_vector_length(
        x=histogram_bin_centres_tiled, 
        w=histogram_counts, 
    )
    mrl_max = np.max(mrl)
    mrl_max_ind = np.where(mrl == mrl_max)
    
    d["max_mrl"] = mrl_max
    
    mean_direction = circular_mean(histogram_bin_centres_tiled, histogram_counts)
    d["max_mrl_direction"] = np.rad2deg(mean_direction[mrl_max_ind])
    
    pval, z = circular_rayleigh_test(
        histogram_bin_centres, histogram_counts[:, mrl_max_ind[0][0], mrl_max_ind[1][0]]
    )
    d["max_mrl_distribution"] = histogram_counts[:, mrl_max_ind[0][0], mrl_max_ind[1][0]]
    
    d["max_mrl_ind"] = np.array([x_bin_edges[mrl_max_ind[1][0]], y_bin_edges[mrl_max_ind[0][0]]])
    d["max_mrl_pval"] = pval
    d["max_mrl_z"] = z
    
    return d
    