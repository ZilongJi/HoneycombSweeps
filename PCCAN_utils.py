import matplotlib.pyplot as plt
import numpy as np
from scipy.ndimage import gaussian_filter
from matplotlib.animation import FuncAnimation

def find_patches(array):
    '''Find connected patches of 1s in a binary array.
    
    A patch is defined as a continuous set of pixels (>=1 pixel). 
    For inclusion in a patch, a candidate must share at least an edge 
    with a pixel known to be a member of the patch; a corner is not enough.
    
    '''
    
    # Define 4-connected neighborhood
    neighbors = [(0, 1), (1, 0), (0, -1), (-1, 0)]

    # Function to get neighboring pixels
    def get_neighbors(pixel):
        i, j = pixel
        return [(i + di, j + dj) for di, dj in neighbors]

    patches = []
    visited = set()
    unvisited = set(zip(*np.where(array == 1)))

    while unvisited:
        # Start a new patch
        current_patch = set()
        stack = [unvisited.pop()]
        while stack:
            pixel = stack.pop()
            if pixel not in visited:
                visited.add(pixel)
                current_patch.add(pixel)
                for neighbor in get_neighbors(pixel):
                    if neighbor in unvisited:
                        stack.append(neighbor)
                        unvisited.remove(neighbor)
        patches.append(current_patch)

    return patches

def get_firing_area(map, firing_rate_thres):
    return np.sum(map>firing_rate_thres)
    
def get_spatial_cohenrence(map):
    '''
    Coherence is the z-transform of the correlation between a list of firing rates in each pixel 
    and a corresponding list of firing rates averaged over the 8 nearest-neighbors of each pixel. 
    '''
    
    #Consider 1:-1 to avoid the border
    firing_rates = map[1:-1,1:-1].flatten()
    neighbors = np.zeros_like(firing_rates)
    
    id = 0
    for i in range(1, map.shape[0]-1):
        for j in range(1, map.shape[1]-1):
            neighbors_8 = map[i-1:i+2, j-1:j+2].flatten()
            #exclue the center  
            neighbors_8 = np.delete(neighbors_8, 4)
            #take the average of the neighbors
            neighbors[id] = np.mean(neighbors_8)
            id += 1
    
    return np.corrcoef(firing_rates, neighbors)[0,1]

def get_patchiness(map):
    '''
    get the number of patches in the map
    '''
    
    # step 1:
    # Each element in the rate array is put into one of 6 categories, 1,2,3,4,5,6. The breakpoints
    # between two adjacent categories are not fixed from map to map. Instead, the breakpoints are set so that 
    # the number of pixels assigned to a category is eaual to 0.8 times the number of nixels in the 
    # next lower firing category. This rank-scaling ensures that all 6 categories will be used to 
    # cover the full firing rate range, independent of the absolute firing rate.
    
    #get the breakpoints
    num_pixels = map.shape[0]*map.shape[1]
    pixels_in_low_categories = int(num_pixels/(1+0.8+0.8**2+0.8**3+0.8**4+0.8**5))
    pixels_in_category_6 = pixels_in_low_categories*0.8**5
    #get the breakpoint for category 6 so that the number of pixels in category 6 is pixels_in_category_6
    breakpoint_forcategory_6 = np.percentile(map, 100-100*pixels_in_category_6/num_pixels)
    
    patch_6 = map>breakpoint_forcategory_6
    
    # step 2:
    patches = find_patches(patch_6)

    return len(patches)

def spatial_information_content(map, FR_In_Position, Time_In_Position):
    '''
    calculate spatial information content following Skaggs 1993 paper
    SIC = sum_i(p_i*\lambda_i/\lambda)log_2(\lambda_i/\lambda)
    where p_i is the probablity of the mouse visiting the i-th bin, \lambda_i is the activity in the i-th bin, \lambda is the mean activity across all bins
    '''
    
    prob_visit = np.divide(Time_In_Position, np.sum(Time_In_Position))
    mean_map = np.sum(FR_In_Position)/np.sum(Time_In_Position)
    SIC = prob_visit * map/mean_map * np.log2(map/mean_map+1e-10)
    
    return np.nansum(SIC)

    
def get_tuningMap(activity, positions, cellindex_x, cellindex_y, shift, 
                 filter=False, firing_rate_thres=0.05, samples_per_sec=10, dim=40):
    '''
    Get the tuning map of a cell
    Input:
        activity: the activity of the network across time
        positions: the position of the animal
        cellindex_x: the x index of the cell
        cellindex_y: the y index of the cell
        shift: the shift of the activity; Positive values shift the activity to the right, negative values to the left
        firing_rate_thres: the threshold of the firing rate
    '''
    
    activity_4_cell_i = activity[:, cellindex_x, cellindex_y]
    
    activity_4_cell_i = np.roll(activity_4_cell_i, shift) #positive shift means shift to the right!!
    
    #position in bins
    position_x = positions[:,0]; position_x = (position_x*dim).astype(int)
    position_y = positions[:,1]; position_y = (position_y*dim).astype(int)
    
    #Calculate the summarized  activity and the occupancy time in each position bin 
    n_bins_x = dim; n_bins_y = dim
    Time_In_Position = np.zeros((n_bins_x, n_bins_y))
    FR_In_Position = np.zeros((n_bins_x, n_bins_y))

    diffTimeStamps = np.asarray([1/samples_per_sec]*len(positions))
    np.add.at(Time_In_Position, (position_x, position_y), diffTimeStamps)
    np.add.at(FR_In_Position, (position_x, position_y), activity_4_cell_i)
    
    map = np.divide(FR_In_Position, Time_In_Position, out=np.zeros_like(FR_In_Position), where=Time_In_Position!=0)
    
    if filter is True:
        #Gaussian smoothing the map
        map = gaussian_filter(map, sigma=2)
    
    # #get the quantity of the rate map
    # map_criteria = {}
    # #first, firing area
    # map_criteria['firing_area'] = get_firing_area(map, firing_rate_thres)
    # #second, patchiness
    # map_criteria['patchiness'] = get_patchiness(map)
    # #third, spatial coherence
    # map_criteria['spatial_coherence'] = get_spatial_cohenrence(map)
    # #fourth, spatial information content
    # map_criteria['spatial_information_content'] = spatial_information_content(map, FR_In_Position, Time_In_Position)
    
    # return map, map_criteria
    
    return map


def animate_sweeps(Position, pc_activity, num, duration, Speed, m0, n_step=10, goal_loc=None, save_path='./animations/', filename_prefix='GD_adaptation_'):
    """
    Creates an animated heatmap with position and speed information.

    Parameters:
    - Position: 2D numpy array of (x, y) positions.
    - num: Integer scaling factor for position data.
    - duration: Integer representing the duration for data slicing.
    - pc_activity: 3D numpy array representing activity heatmap data.
    - goal_loc: List or array-like of goal location coordinates (x, y).
    - Speed: 1D numpy array of speed values for each time point.
    - m0: Identifier for the file name.
    - n_step: Sampling interval for down-sampling the data (default is 10).
    - save_path: Path to save the generated GIF file (default is './animations/').
    - filename_prefix: Prefix for the filename (default is 'GD_adaptation_').
    """
    # Initialize plot
    fig, ax = plt.subplots(figsize=(3, 3), dpi=100)

    # Rescale and sample position data
    position_x_int = (Position[:, 0] * num).astype(int)
    position_y_int = (Position[:, 1] * num).astype(int)
    
    start = 0
    end = int(duration + 1)
    pc_activity_sampled = pc_activity[start:end:n_step]
    position_x_int = position_x_int[start:end:n_step]
    position_y_int = position_y_int[start:end:n_step]
    if goal_loc is not None:
        goal_loc_int = (np.array(goal_loc) * num).astype(int)

    # Set up the fixed ticks and labels
    ax.set_xticks([0, 50, 99])
    ax.set_xticklabels([0, 0.5, 1])
    ax.set_yticks([0, 50, 99])
    ax.set_yticklabels([0, 0.5, 1])

    # Plot the static trajectory as a grey line
    ax.plot(Position[:, 0] * num, Position[:, 1] * num, color='grey', linewidth=1, label='Trajectory')

    # Initialize the elements to update
    position_marker = ax.scatter(position_x_int[0], position_y_int[0], color='#009FB9', s=50)
    if goal_loc is not None:
        goal_marker = ax.scatter(goal_loc_int[0], goal_loc_int[1], color='#F18D00', marker='*', s=200)
    heatmap = ax.imshow(pc_activity_sampled[0], cmap='Blues', origin='lower')
    title = ax.text(0.5, 1.05, "", transform=ax.transAxes, ha="center")

    def update(i):
        # Update the position marker
        position_marker.set_offsets([position_x_int[i], position_y_int[i]])
        
        if goal_loc is not None:
            goal_marker.set_offsets([goal_loc_int[0], goal_loc_int[1]])
        
        # Update the heatmap data
        heatmap.set_array(pc_activity_sampled[i])

        # Update the title with current time and speed
        title.set_text(f'Time: {np.round(i / 100, 2)}s; Speed: {np.round(Speed[i] * 1000, 2)} m/s')
        
        return position_marker, heatmap, title

    # Create animation
    ani = FuncAnimation(fig, update, frames=len(pc_activity_sampled), interval=100)

    # Define filename and save animation
    save_filename = f"{save_path}{filename_prefix}{m0}.gif"
    ani.save(save_filename, writer='imagemagick', fps=10)
    
    print(f'Animation saved to {save_filename}')
    plt.close(fig)  # Close the figure to free memory