from typing import Optional, Dict, Any, Tuple, List
import numpy as np
from scipy.interpolate import splprep, splev
from scipy.ndimage import gaussian_filter1d, gaussian_filter
from scipy import stats

import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib.patches import Wedge, Polygon
from matplotlib.colors import Normalize
from matplotlib.cm import get_cmap
import matplotlib.gridspec as gridspec



def plot_circular_histogram(
    angles: np.ndarray, 
    bins: int = 36, 
    density: bool = True, 
    offset: int = 0, 
    gaps: bool = True, 
    ax=None, 
    title: str = "", 
    fontsize: float = 12, 
    colors: Optional[Dict[str, Any]] = None, 
    smooth_window: int = 0, 
    alpha: float = 0.7, 
    show_xlabels: bool = True, 
    barcolor: Optional[str] = None, 
):
    """
    Plot a circular histogram of angles on a polar axis.

    Parameters:
    -----------
    angles : array
        Array of angles to plot, in radians.
    bins : int, optional
        Number of bins to use. Default is 36.
    density : bool, optional
        If True, normalize by bin width and total sample count. Default is True.
    offset : float, optional
        Sets the offset from the center for the bars. Default is 0.
    gaps : bool, optional
        Whether to include small gaps between bars. Default is True.
    """
    if ax is None:
        ax = plt.gca()
    
    if colors is None:
        colors = {
            'bars': '#8B3A62',          # Sophisticated burgundy - professional but engaging
            'edge': '#662B49',          # Slightly darker edge for definition  
            'grid': '#E5E5E5',          # Very light gray for subtle gridlines
            'text': 'gray',          # Nearly black for clear text
        }
    COLORS = colors
    if barcolor is not None:
        COLORS["bars"] = barcolor
    
    # Wrap angles to [-pi, pi]
    angles = (angles + np.pi) % (2*np.pi) - np.pi
    
    # Bin data and record counts
    count, bins = np.histogram(angles, bins=bins, range=(-np.pi, np.pi))
    
    if smooth_window > 0:
        padded = np.concatenate([count[-smooth_window:], count, count[:smooth_window]])
        smoothed_counts = np.convolve(padded, np.ones(smooth_window)/smooth_window, mode='valid')
        count = smoothed_counts[smooth_window-1:-smooth_window+1]
    
    # Compute width of each bin
    widths = np.diff(bins)
    
    # By default plot density (frequency potentially misleading)
    if density:
        # Area to assign each bin
        area = count / angles.size
        # Calculate corresponding bin radius
        radius = (area / widths) ** .5
    else:
        radius = count
    
    # Plot data on ax
    bars = ax.bar(bins[:-1], radius, zorder=1, align='edge', width=widths, color=COLORS["bars"],
                  edgecolor="k", fill=True, linewidth=0.5, alpha=alpha)
    
    # Set the direction of the zero angle
    ax.set_theta_offset(offset)
    ax.set_theta_zero_location('N')  # 0 degrees at North
    ax.set_theta_direction(-1)       # clockwise
    
    # Remove ylabels, they're meaningless for a circular histogram
    ax.grid(True, linestyle='-', color=COLORS['grid'], alpha=0.5)
    
    # Set r ticks and labels
    if not density:
        max_count = max(count)
        rticks = np.linspace(0, max_count, 5)
        ax.set_rticks(rticks)
    
        # Format the rtick labels
        rlabels = ["", "", "", "", int(max_count)]
        ax.set_yticklabels(rlabels, 
                        color=COLORS['text'],
                        fontsize=fontsize)
        ax.set_rlabel_position(22.5)
    else:
        # For density plots, we want to show only the max density value
        max_density = max(radius)
        rticks = np.linspace(0, max_density, 5)
        ax.set_rticks(rticks)

        # Format the rtick labels to show only max
        max_density = max(count) / np.sum(count)
        rlabels = ["", "", "", "", f"{max_density:.2f}"]
        ax.set_yticklabels(rlabels, 
                        color=COLORS['text'],
                        fontsize=fontsize)
        ax.set_rlabel_position(22.5)
        # max_count = max(count) / np.sum(count)
        # rticks = np.sqrt(np.linspace(0, max_count, 5))
        # ax.set_rticks(rticks)
    
        # # Format the rtick labels
        # rlabels = ["", "", "", "", int(max_count)]
        # ax.set_yticklabels(rlabels, 
        #                 color=COLORS['text'],
        #                 fontsize=fontsize)
        # ax.set_rlabel_position(22.5)
    
    ax.tick_params(axis='x', pad=10)
    ax.tick_params(axis='y', pad=-5)
    
    if gaps:
        # Add small gaps between bars
        for bar in bars:
            bar.set_edgecolor('white')
            bar.set_linewidth(0.5)

    # Use custom labels
    # ax.set_xticklabels(['0°', '45°', '90°', '135°', '180°', '225°', '270°', '315°'], fontsize=fontsize)
    if show_xlabels:
        ax.set_xticks(np.linspace(0, 2*np.pi, 8, endpoint=False))
        ax.set_xticklabels(['0°', '', '', '', '180°', '', '', ''], fontsize=fontsize)
    else:
        ax.set_xticks(np.linspace(0, 2*np.pi, 8, endpoint=False))
        ax.set_xticklabels(["", "", "", "", "", "", "", ""])
    
    ax.set_title(title, fontsize=fontsize, va="bottom")
    
    # for patch in ax.patches:
    #     patch.set_edgecolor('gray')
    #     patch.set_linewidth(1.5)
    
    # ax.set_thetagrids([])  # Remove angular grid lines
    ax.grid(True)  
    
    plt.tight_layout()
    
    return count


def plot_circular_histogram_with_values(
    fig,
    ax, 
    counts: np.ndarray, 
    bins: np.ndarray, 
    normalize: bool = False,
    gaps: bool = True,  
    offset: float = 0.0, 
    circle_size: float = 1.5, 
    fontsize: float = 20.0,
    colors: Optional[Dict[str, Any]] = None, 
    title: str = "", 
    mean_direction: Optional[float] = None, 
    show_dir_bins: bool = True, 
    alpha=0.8, 
    show_rlabels: bool = True, 
    **kwargs, 
):
    """
    Plot a circular histogram (rose plot) on a polar axis.
    
    Parameters:
    ax (matplotlib.axes.Axes): The polar axis to plot on.
    values (array-like): The height of each bin.
    bin_edges (array-like): The edges of the bins in radians.
    normalize (bool): If True, normalize the values to sum to 1.
    **kwargs: Additional keyword arguments to pass to ax.bar
    
    Returns:
    None
    """
    if colors is None:
        colors = {
            'bars': '#8B3A62',          # Sophisticated burgundy - professional but engaging
            'edge': '#662B49',          # Slightly darker edge for definition  
            'grid': '#E5E5E5',          # Very light gray for subtle gridlines
            'text': 'gray',          # Nearly black for clear text
        }
    COLORS = colors
    
    widths = np.diff(bins)
    
    if normalize:
        counts = counts / np.max(counts)
    
    bars = ax.bar(bins[:-1], counts, zorder=1, align='edge', width=widths, color=COLORS["bars"],
                  edgecolor=COLORS["edge"], fill=True, linewidth=0.5, alpha=0.7)
    
    # Set the direction of the zero angle
    ax.set_theta_offset(offset)
    ax.set_theta_zero_location('N')  # 0 degrees at North
    ax.set_theta_direction(-1)       # clockwise
    
    # Remove ylabels, they're meaningless for a circular histogram
    ax.grid(True, linestyle='-', color=COLORS['grid'])
    
    # Set r ticks and labels
    max_count = max(counts)
    rticks = np.linspace(0, max_count, 5)
    ax.set_rticks(rticks)
    
    # Format the rtick labels
    if show_rlabels:
        rlabels = ["", "", "", "", f"{max_count:.2f}"]
    else:
        rlabels = ["", "", "", "", ""]
    ax.set_yticklabels(rlabels, 
                       color=COLORS['text'],
                       fontsize=fontsize)
    ax.set_rlabel_position(10)
    
    ax.tick_params(axis='x', pad=15)
    ax.tick_params(axis='y', pad=-5)
    
    if gaps:
        # Add small gaps between bars
        for bar in bars:
            bar.set_edgecolor('white')
            bar.set_linewidth(0.5)
            bar.set_alpha(alpha)

    # Use custom labels
    # ax.set_xticklabels(['0°', '45°', '90°', '135°', '180°', '225°', '270°', '315°'], fontsize=fontsize)
    if show_dir_bins:
        ax.set_xticklabels(['0°', '', '', '', '180°', '', '', ''], fontsize=fontsize)
    else:
        ax.set_xticklabels(['', '', '', '', '', '', '', ''], fontsize=fontsize)
    
    ax.set_title(title, fontsize=fontsize)
    
    if mean_direction is not None:
        # Calculate arrow length (e.g., 80% of max radius)
        arrow_length = max(counts)
        
        # Plot arrow
        ax.annotate('',
            xy=(mean_direction, arrow_length),    # Arrow head
            xytext=(mean_direction, 0),           # Arrow base
            arrowprops=dict(
                arrowstyle='->',
                color='#4A4A4A',                    # Arrow color
                linewidth=4,                      # Arrow thickness
                mutation_scale=15                 # Arrow head size
            ),
            zorder=3  # Ensure arrow appears above bars
        )
        
        # Optionally add a small circle at the arrow base
        circle = plt.Circle((0, 0), max(counts)*0.02, 
                          fill=True, 
                          color='black', 
                          zorder=4)
        ax.add_artist(circle)
    
    plt.tight_layout()
    
    return bars


def plot_expected_theta_sweep(
    d_aggregated_theta_sweeps, 
    ax=None, 
    radius: float = 4.0, 
    alpha: float = 0.4, 
    linewidth: float = 4.0, 
    start_theta_sweep_ind: int = 0, 
    end_theta_sweep_ind: int = 12, 
    rays: bool = True, 
    plot_md: bool = False, 
    fontsize: int = 20, 
    use_smooth_trajectories: bool = True, 
    plot_scalar_bar: bool = True, 
    centre=np.array([0.0, 0.0]), 
):
    if ax is None:
        ax = plt.gca()
        
    sweep_colors = get_tab20_colors(len(d_aggregated_theta_sweeps))
    sweep_colors = sweep_colors
    # sweep_colors = sweep_colors[-3:] + sweep_colors[:-3]
    
    for i, direction in enumerate(d_aggregated_theta_sweeps):
        print(i)
        plot_theta_sweep_trajectory(
            d_aggregated_theta_sweeps[direction][start_theta_sweep_ind:end_theta_sweep_ind], 
            target_color=sweep_colors[i], 
            ax=ax, 
            linewidth=linewidth, 
            use_smooth_line=use_smooth_trajectories,
        )
    
    if rays:
        plot_rays(center=centre, radius=radius, ax=ax, alpha=alpha)
    
    xlim = ax.get_xlim()
    ylim = ax.get_ylim()
    
    bar_length = 1  # 5 cm
    x_start = xlim[1] - (xlim[1] - xlim[0]) * 0.2
    y_pos = ylim[1] - (ylim[1] - ylim[0]) * 0.1

    # Plot the scale bar
    if plot_scalar_bar:
        ax.plot([x_start, x_start + bar_length], [y_pos, y_pos], 
                    'k-', linewidth=2)  # 'k-' means black solid line

        # Add the label
        # Position the text centered below the line
        ax.text(x_start + bar_length/2, y_pos + (ylim[1] - ylim[0]) * 0.02, 
                    '5 cm', ha='center', va='top', fontsize=fontsize)
    
    if plot_md:
        arrow_x = ax.get_xlim()[0] + 1  # Adjust the x position as needed
        arrow_y = (ax.get_ylim()[0] + ax.get_ylim()[1]) / 2 - 4  # Middle of y-axis

        # Add the arrow
        ax.annotate('M.D.', 
                    xy=(arrow_x, arrow_y),  # Arrow tip (where annotation will be)
                    xytext=(arrow_x, arrow_y + 2),     # Arrow start
                    arrowprops=dict(facecolor='black', 
                                shrink=0,
                                width=1,
                                headwidth=8,
                                headlength=10),
                    ha='right',
                    va='center')
    
    ax.set_aspect("equal")
    ax.axis("off")
    
    
def plot_theta_sweep_trajectory(
    locs: np.ndarray,
    target_color: str = "", 
    cmap: Optional[str] = None, 
    ax=None,
    linewidth: float = 2,
    smoothing: str = 'spline',
    interpolation_factor: int = 50,  # Increase for smoother curves
    gaussian_sigma: float = 1.0,
    edge_width: float = 3.5,
    highlight_start_end_inds: Optional[List[int]] = None, 
    highlight_color: str = '#4D4D4D',
    highlight_alpha: float = 0.3,
    use_smooth_line: bool = True,
    truncation: Optional[Tuple[float, float]] = None,
):
    """
    Plot smoothed theta sweep trajectories with proper color gradients.
    """
    if ax is None:
        ax = plt.gca()
        
    if len(locs) < 2:
        return
        
    # Store original number of points for color mapping
    original_length = len(locs)
        
    # Create more points for smoother interpolation
    if smoothing == 'spline' and len(locs) >= 4:
        # Use spline interpolation for truly smooth curves
        tck, u = robust_spline_interpolation(
            trajectory=locs, 
            sigma=gaussian_sigma,
            min_distance=0.0001,
            s=0
        )
        # tck, u = splprep([locs[:, 0], locs[:, 1]], s=0, k=min(3, len(locs)-1))
        u_new = np.linspace(0, 1, len(locs) * interpolation_factor)
        smooth_coords = splev(u_new, tck)
        locs = np.column_stack(smooth_coords)
        if truncation is not None:
            start_ind = int(truncation[0] * len(locs))
            end_ind = int(truncation[1] * len(locs))
            locs = locs[start_ind:end_ind]
        
    elif smoothing == 'gaussian':
        # Apply Gaussian smoothing to x and y coordinates separately
        x_smooth = gaussian_filter1d(locs[:, 0], sigma=gaussian_sigma)
        y_smooth = gaussian_filter1d(locs[:, 1], sigma=gaussian_sigma)
        
        # Create more points through linear interpolation
        t = np.linspace(0, len(locs)-1, len(locs)*interpolation_factor)
        x_interp = np.interp(t, np.arange(len(locs)), x_smooth)
        y_interp = np.interp(t, np.arange(len(locs)), y_smooth)
        
        locs = np.column_stack([x_interp, y_interp])
    
    # Get or create the colormap
    if cmap is not None:
        custom_cmap = plt.get_cmap(cmap) if isinstance(cmap, str) else cmap
    else:
        custom_cmap = create_custom_trajectory_cmap(target_color)
    
    if use_smooth_line:
        # Create a smooth path with gradient coloring
        # Use path-based approach for truly smooth lines
        
        # Create color values that map to temporal progression
        time_values = np.linspace(0, 1, len(locs))
        colors = custom_cmap(time_values)
        
        # For gradient effect, we'll plot multiple overlapping lines with varying alpha
        # This creates a smooth gradient effect
        
        # First plot the black edge as a single smooth line
        ax.plot(locs[:, 0], locs[:, 1], color='black', linewidth=edge_width, 
                solid_capstyle='round', solid_joinstyle='round', zorder=2)
        
        # Create gradient by plotting segments with individual colors
        # But use fewer segments to maintain smoothness
        n_segments = min(100, len(locs)//5)  # Reduce number of segments
        if n_segments < 2:
            n_segments = len(locs) - 1
            
        segment_indices = np.linspace(0, len(locs)-1, n_segments+1).astype(int)
        
        for i in range(len(segment_indices)-1):
            start_idx = segment_indices[i]
            end_idx = segment_indices[i+1]
            
            # Get the segment
            segment_locs = locs[start_idx:end_idx+1]
            
            # Get color for this segment (midpoint)
            color_idx = (start_idx + end_idx) // 2
            color_value = time_values[color_idx]
            color = custom_cmap(color_value)
            
            # Plot this segment
            ax.plot(segment_locs[:, 0], segment_locs[:, 1], 
                   color=color, linewidth=linewidth,
                   solid_capstyle='round', solid_joinstyle='round', zorder=3)
        
        return None
        
    else:
        # Use original line collection approach (unchanged)
        locs_reshaped = locs.reshape(-1, 1, 2)
        segments = np.concatenate([locs_reshaped[:-1], locs_reshaped[1:]], axis=1)
        
        # Create color gradient that maps to original temporal progression
        if smoothing:
            time_values = np.linspace(0, original_length-1, len(segments))
            norm = plt.Normalize(0, original_length-1)
        else:
            time_values = np.arange(len(segments))
            norm = plt.Normalize(0, len(segments))
        
        # First plot the black edge
        lc_edge = LineCollection(segments, colors='black', linewidth=edge_width, zorder=2)
        ax.add_collection(lc_edge)
        
        # Then plot the colored line on top with gradient
        lc_color = LineCollection(segments, cmap=custom_cmap, norm=norm, zorder=3)
        lc_color.set_array(time_values)
        lc_color.set_linewidth(linewidth)
        
        # Add to plot
        ax.add_collection(lc_color)
        
        return lc_color
    
    
def robust_spline_interpolation(trajectory, sigma=0.3, min_distance=0.0001, s=0):
    """
    Robust spline interpolation that handles duplicates and edge cases.
    
    Parameters:
    -----------
    trajectory : np.ndarray
        2D array of shape (n_points, 2) containing x,y coordinates
    sigma : float
        Gaussian filter sigma for smoothing
    min_distance : float
        Minimum distance between consecutive points
    s : float
        Smoothing parameter for spline (0 = interpolation, >0 = approximation)
    
    Returns:
    --------
    tck : tuple
        Spline parameters, or None if fitting failed
    u : np.ndarray
        Parameter values, or None if fitting failed
    """
    
    # Apply smoothing
    smoothed_trajectory = gaussian_filter1d(trajectory, sigma=sigma, axis=0)
    
    # Remove consecutive points that are too close
    keep_indices = [0]  # Always keep first point
    for i in range(1, len(smoothed_trajectory)):
        distance = np.sqrt(np.sum((smoothed_trajectory[i] - smoothed_trajectory[keep_indices[-1]])**2))
        if distance > min_distance:
            keep_indices.append(i)
    
    # Ensure we always keep the last point if it's different from the last kept point
    if keep_indices[-1] != len(smoothed_trajectory) - 1:
        last_distance = np.sqrt(np.sum((smoothed_trajectory[-1] - smoothed_trajectory[keep_indices[-1]])**2))
        if last_distance > min_distance * 0.5:  # Use smaller threshold for last point
            keep_indices.append(len(smoothed_trajectory) - 1)
    
    filtered_trajectory = smoothed_trajectory[keep_indices]
    
    # Check if we have enough points for spline fitting
    if len(filtered_trajectory) < 4:
        print(f"Warning: Only {len(filtered_trajectory)} unique points, cannot fit cubic spline")
        return None, None
    
    try:
        # Determine spline degree
        k_degree = min(3, len(filtered_trajectory) - 1)
        
        # Fit spline
        tck, u = splprep([filtered_trajectory[:, 0], filtered_trajectory[:, 1]], 
                         s=s, k=k_degree)
        return tck, u
        
    except ValueError as e:
        print(f"Spline fitting failed: {e}")
        return None, None


def plot_rays(
    center: np.ndarray, 
    radius: int=1, 
    num_rays: int=12, 
    alpha: float=0.5, 
    colors=None, 
    ax=None, 
):
    if ax is None:
        ax = plt.gca()
    
    angles = np.linspace(0, 2*np.pi, num_rays, endpoint=False)
    
    x_ends = center[0] + radius * np.cos(angles)
    y_ends = center [1] + radius * np.sin(angles)
    
    if colors is None:
        colors = get_tab20_colors(num_rays)[::-1]
        colors = colors[-3:] + colors[:-3]
    elif len(colors) != num_rays:
        raise ValueError(f"Number of colors ({len(colors)}) must match number of angular bins!")
    
    for x, y in zip(x_ends, y_ends):
        ax.plot([center[0], x], [center[1], y], "k-", alpha=0.2)
        
    for start, end, color in zip(angles[:-1], angles[1:], colors):
        start = np.rad2deg(start)
        end = np.rad2deg(end)
        wedge = Wedge(center, radius, start, end, fc=color, ec='black', alpha=alpha)
        ax.add_artist(wedge)
    # last bin
    start = np.rad2deg(angles[-1])
    end = np.rad2deg(angles[0])
    wedge = Wedge(center, radius, start, end, fc=colors[-1], ec="black", alpha=alpha)
    ax.add_artist(wedge)
    
    ax.plot(
        center[0], 
        center[1], 
        marker="^", 
        color="white", 
        markersize=20, 
        markeredgecolor="black",
        markeredgewidth=1.0, 
        zorder=10, 
    )
    
    return ax


def get_tab20_colors(num_segments=12):
    """
    Get colors from tab20 colormap
    """
    # tab20 has 20 colors, repeating if needed
    cmap = plt.cm.tab20
    colors = [cmap(i % 20) for i in range(num_segments)]
    return colors


def create_custom_trajectory_cmap(target_color):
    """
    Create a custom colormap that transitions from white to the target color
    
    Parameters:
    -----------
    target_color : tuple or str
        The target color from tab20
        
    Returns:
    --------
    matplotlib.colors.LinearSegmentedColormap
        Custom colormap transitioning from white to target color
    """
    from matplotlib.colors import LinearSegmentedColormap
    
    # Create colormap transitioning from white to target color
    return LinearSegmentedColormap.from_list('custom_cmap', 
                                           ['white', target_color])
    
    
def plot_paired_boxplot(data1, data2, pos1=1, pos2=2, labels=['Centre', 'Goal'], ylabel="Mean Resultant Vector Length", 
                       figsize=(1.2, 2.5), width=0.5, plot_paired_lines=True, plot_significance=True, 
                       pair_color='gray', pair_alpha=0.3, fontsize=15, ylim=None, show_outliers=False, 
                       colors=None, alternative="greater", fig=None, ax=None, paired_test=False, alpha=1.0, mhu_test=False, y_max=None):
    """
    Create a paired boxplot with statistical comparison.
    
    Parameters:
    -----------
    data1, data2 : array-like
        Arrays containing paired measurements
    labels : list of str
        Labels for the two groups
    figsize : tuple
        Figure size (width, height)
    width : float
        Width of the boxes
    pair_color : str
        Color of the connection lines
    pair_alpha : float
        Alpha transparency for connection lines
    """
    # Perform statistical test
    if paired_test:
        t_stat, p_val = stats.ttest_rel(data1, data2, alternative=alternative)
    else:
        t_stat, p_val = stats.ttest_ind(data1, data2, alternative=alternative)
        
    if mhu_test:
        # Perform Mann-Whitney U test
        u_stat, p_val = stats.mannwhitneyu(data1, data2, alternative=alternative)
        t_stat = u_stat
        effective_size, _, _ = compute_mann_whitney_effect_size(data1, data2, alternative=alternative)
        print(f"Mann-Whitney U test statistic: {u_stat}, p-value: {p_val}, effect size: {effective_size}")
    
    # Create figure
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize, dpi=300)
    
    # Create boxplot
    bp = ax.boxplot([data1, data2], 
                    positions=[pos1, pos2], 
                    widths=width,
                    patch_artist=True,
                    medianprops=dict(color='black'),
                    boxprops=dict(facecolor='white', 
                                edgecolor='black'),
                    whiskerprops=dict(color='black'),
                    capprops=dict(color='black'),
                    flierprops=dict(marker='o', 
                                  markerfacecolor='black', 
                                  markersize=4), 
                    showfliers=show_outliers, 
        )
    
    if colors is not None:
        for i, (patch, color) in enumerate(zip(bp['boxes'], colors)):
            patch.set_facecolor(color)
            if isinstance(alpha, List):
                patch.set_alpha(alpha[i])
            else:
                patch.set_alpha(alpha)
    
    # Add paired dots and connection lines
    if plot_paired_lines:
        for i in range(len(data1)):
            ax.plot([pos1, pos2], [data1[i], data2[i]], 
                    color=pair_color, alpha=pair_alpha, 
                    linewidth=.7, zorder=1)
            
        # Add dots for individual data points
        x_jitter_1 = np.random.normal(0, 0.04, size=len(data1))
        x_jitter_2 = np.random.normal(0, 0.04, size=len(data2))
        ax.scatter(pos1 + x_jitter_1, data1, color='black', s=10, alpha=0.8, zorder=2)
        ax.scatter(pos2 + x_jitter_2, data2, color='black', s=10, alpha=0.8, zorder=2)
    
    # Add significance markers
    if plot_significance:
        if ylim is not None:
            y_min, y_max = ylim
            y_max *= 0.9
        else:
            if y_max is None:
                y_max = max(max(data1), max(data2)) # 3
            y_min = min(min(data1), min(data2))
        y_range = y_max - y_min
        
        sig_text = ''
        if p_val < 0.001:
            sig_text = '***'
        elif p_val < 0.01:
            sig_text = '**'
        elif p_val < 0.05:
            sig_text = '*'
        elif p_val > 0.05:
            sig_text = "n.s."
        
        print(p_val)
        
        if sig_text:
            sig_text = sig_text
            ax.plot([pos1, pos1, pos2, pos2], 
                    [y_max + y_range*0.05, y_max + y_range*0.1, 
                    y_max + y_range*0.1, y_max + y_range*0.05],
                    color='black', linewidth=0.75)
            ax.text((pos1+pos2)/2, y_max + y_range*0.12, sig_text, 
                    ha='center', va='bottom', fontsize=fontsize)
    
    # Customize plot
    ax.set_xticks([pos1, pos2])
    ax.set_xticklabels(labels, fontsize=fontsize)
    
    # Remove top and right spines
    ax.spines["left"].set_linewidth(2)
    ax.spines["left"].set_color("black")
    ax.spines["bottom"].set_linewidth(2)
    ax.spines["bottom"].set_color("black")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    
    ax.tick_params(width=2, color="black", direction="out")
    ax.xaxis.label.set_color("black")
    ax.yaxis.label.set_color("black")
    ax.title.set_color("black")
    
    ax.tick_params(axis="x", colors="black")
    ax.tick_params(axis="y", colors="black")
    ax.tick_params(axis="y", labelsize=fontsize)
    ax.tick_params(axis="x", labelsize=fontsize)
    
    ax.set_ylabel(ylabel, fontsize=fontsize)
    
    if ylim is not None:
        ax.set_ylim(ylim)
    
    # Adjust layout
    plt.tight_layout()
    
    return fig, ax


def compute_mann_whitney_effect_size(data1, data2, alternative="two-sided"):
    """
    Compute effect size for Mann-Whitney U test using rank-biserial correlation.
    
    Returns:
    --------
    r : float
        Rank-biserial correlation (-1 to 1)
        Small: 0.1, Medium: 0.3, Large: 0.5
    """
    n1, n2 = len(data1), len(data2)
    
    # Perform Mann-Whitney U test
    u_stat, p_val = stats.mannwhitneyu(data1, data2, alternative=alternative)
    
    # Calculate rank-biserial correlation
    r = 1 - (2 * u_stat) / (n1 * n2)
    
    return r, u_stat, p_val


def plot_phase_precession_wrt_nd(
    SpikePhase_decreasing, 
    SpikePhase_increasing, 
    Dist2G_decreasing, 
    Dist2G_increasing, 
    phi_decreasing, 
    slope_decreasing, 
    R_decreasing, 
    phi_increasing, 
    slope_increasing, 
    R_increasing, 
    n_bins: int = 120, 
    fontsize: int = 20, 
):
    fig = plt.figure(figsize=(12.5, 5), dpi=300)
    gs = gridspec.GridSpec(1, 2)

    ax = plt.subplot(gs[0])

    # Wrap phase by duplicating with +2π to handle circularity
    SpikePhase_wrapped = np.concatenate([SpikePhase_decreasing, SpikePhase_decreasing + 2 * np.pi, SpikePhase_decreasing + 4*np.pi]) 
    Dist2G_wrapped = np.concatenate([Dist2G_decreasing, Dist2G_decreasing, Dist2G_decreasing])
    # Compute 2D histogram
    heatmap, xedges, yedges = np.histogram2d(Dist2G_wrapped, SpikePhase_wrapped, bins=[n_bins, n_bins])
    # Smooth with Gaussian filter
    heatmap_smooth = gaussian_filter(heatmap, sigma=4)

    sum_heatmap = np.sum(heatmap_smooth, axis=1)
    valid_x_indices = sum_heatmap > 0.5 * np.max(sum_heatmap)
    # Apply column filter
    heatmap_smooth_4_plot = heatmap_smooth[valid_x_indices, :]

    # Adjust xedges and X accordingly
    xedges_filtered = xedges[:-1][valid_x_indices]  # remove last bin edge, select valid
    xedges_filtered = np.append(xedges_filtered, xedges_filtered[-1] + np.diff(xedges).mean())  # match dimension

    # Create matching meshgrid
    X, Y = np.meshgrid(xedges_filtered, yedges)

    # Plot
    pcm = ax.pcolormesh(X, Y, heatmap_smooth_4_plot.T, shading='auto', cmap='jet')

    #!!!! flip x axis with small values on the right and large values on the left
    # Flip the x-axis
    ax.invert_xaxis()

    width = xedges_filtered.max() - xedges_filtered.min()
    xmin = xedges_filtered.min() + 0.0 * width
    xmax = xedges_filtered.max() - 0.0 * width
    # ax.set_xlim(xmin, xmax)
    ax.set_xticks([xmin, xmax])
    ax.set_xticklabels(['Close\nto\ngoal', 'Far\nfrom\ngoal'], fontsize=fontsize)

    ax.set_ylabel('Theta phase', fontsize=fontsize)
    # #activity colorba
    ax.set_ylim(np.pi, 5*np.pi) #start from np.pi since we want to make MUA trough as 0 degree, previous is 180 degree
    ax.set_yticks([np.pi, 2*np.pi, 3*np.pi, 4*np.pi, 5*np.pi])
    ax.set_yticklabels([r'$-360$', r'$-180$', r'$0$', r'$180$', r'$360$'], fontsize=fontsize)   
    ax.set_title('Towards', fontsize=fontsize)
    ax.tick_params(axis='both', which='major', labelsize=fontsize)

    # Create line in data coordinates, then map to bin coordinates
    x_line = np.linspace(xedges_filtered.min(), xedges_filtered.max(), 100)  # bin indices
    x_data = np.linspace(Dist2G_decreasing.min(), Dist2G_decreasing.max(), 100)  # corresponding data values

    y_fitted = np.mod(slope_decreasing * x_line - 0, 8 * np.pi)

    ax.plot(x_line[::-1], y_fitted, 'k', linewidth=4, alpha=0.8)
    ax.plot(x_line[::-1], y_fitted - 2 * np.pi, 'k', linewidth=4, alpha=0.8)
    ax.plot(x_line[::-1], y_fitted - 4 * np.pi, 'k', linewidth=4, alpha=0.8)


    #------------------------------- ------------------------------------- --------------------------------- ------------------------------------      
    fontsize = 20
    ax = plt.subplot(gs[1])

    # Wrap phase by duplicating with +2π to handle circularity
    SpikePhase_wrapped = np.concatenate([SpikePhase_increasing, SpikePhase_increasing + 2 * np.pi, SpikePhase_increasing + 4*np.pi]) 
    Dist2G_wrapped = np.concatenate([Dist2G_increasing, Dist2G_increasing, Dist2G_increasing])
    # Compute 2D histogram
    heatmap, xedges, yedges = np.histogram2d(Dist2G_wrapped, SpikePhase_wrapped, bins=[n_bins, n_bins])
    # Smooth with Gaussian filter
    heatmap_smooth = gaussian_filter(heatmap, sigma=4)
    sum_heatmap = np.sum(heatmap_smooth, axis=1)
    valid_x_indices = sum_heatmap > 0.5 * np.max(sum_heatmap)
    # Apply column filter
    heatmap_smooth_4_plot = heatmap_smooth[valid_x_indices, :]

    # Adjust xedges and X accordingly
    xedges_filtered = xedges[:-1][valid_x_indices]  # remove last bin edge, select valid
    xedges_filtered = np.append(xedges_filtered, xedges_filtered[-1] + np.diff(xedges).mean())  # match dimension

    # Create matching meshgrid
    X, Y = np.meshgrid(xedges_filtered, yedges)

    # Plot
    pcm = ax.pcolormesh(X, Y, heatmap_smooth_4_plot.T, shading='auto', cmap='jet')

    width = xedges_filtered.max() - xedges_filtered.min()
    xmin = xedges_filtered.min() + 0.0 * width
    xmax = xedges_filtered.max() - 0.0 * width
        
    # ax.set_ylabel('Theta phase', fontsize=fontsize)
    ax.set_ylim(np.pi, 5*np.pi)  #start from np.pi since we want to make MUA trough as 0 degree, previous is 180 degree
    ax.set_yticks([np.pi, 2*np.pi, 3*np.pi, 4*np.pi, 5*np.pi]) 
    ax.set_yticklabels(["", "", "", "", ""])
    ax.set_title('Away', fontsize=fontsize)
    #tickfontsize as 6
    ax.tick_params(axis='both', which='major', labelsize=8)

    # Create line in data coordinates, then map to bin coordinates
    x_line = np.linspace(xedges_filtered.min(), xedges_filtered.max(), 100)  # bin indices
    x_data = np.linspace(Dist2G_increasing.min(), Dist2G_increasing.max(), 100)  # corresponding data values

    y_fitted = np.mod(slope_increasing * x_line + 0, 8 * np.pi)

    ax.plot(x_line, y_fitted, 'k', linewidth=4, alpha=0.8)
    ax.plot(x_line, y_fitted + 2 * np.pi, 'k', linewidth=4, alpha=0.8)
    ax.plot(x_line, y_fitted + 4 * np.pi, 'k', linewidth=4, alpha=0.8)

    ax.set_xticklabels(['Close\nto\ngoal', 'Far\nfrom\ngoal'], fontsize=fontsize)
    ax.set_xticks([xmin, xmax])

    plt.tight_layout()

    plt.show()


def hexagon_vertices_and_plot(
    ax, 
    center: np.ndarray, 
    left_vertex: np.ndarray, 
    linewidth: float= 1.0, 
    edgecolor="black", 
    alpha: float = 1.0, 
):
    """
    Compute the vertices of a hexagon and add it to the given axis.
    
    Parameters:
    ax (matplotlib.axes.Axes): The axis to add the hexagon to.
    center (tuple): The (x, y) coordinates of the hexagon's center.
    left_vertex (tuple): The (x, y) coordinates of the leftmost vertex.
    
    Returns:
    list: A list of (x, y) tuples representing the hexagon's vertices.
    """
    # Calculate the radius (distance from center to any vertex)
    radius = np.linalg.norm(left_vertex - center)
    
    # Calculate the angle between the center and the left vertex
    angle = np.arctan2(left_vertex[1] - center[1], left_vertex[0] - center[0])
    
    # Generate the angles for all vertices
    angles = angle + np.arange(0, 2*np.pi, np.pi/3)
    
    # Calculate the coordinates of all vertices
    vertices = center + radius * np.column_stack((np.cos(angles), np.sin(angles)))
    
    # Create a Polygon patch and add it to the axis
    hexagon = Polygon(vertices, closed=True, fill=False, 
                      linewidth=linewidth, edgecolor=edgecolor, alpha=alpha)
    ax.add_patch(hexagon)
    
    # Return the list of vertex coordinates
    return vertices.tolist()