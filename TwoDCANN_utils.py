import matplotlib.pyplot as plt
import jax
import brainpy as bp
import brainpy.math as bm
import numpy as np
import numpy as np
from scipy.ndimage import gaussian_filter

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
    
    #get the quantity of the rate map
    map_criteria = {}
    #first, firing area
    map_criteria['firing_area'] = get_firing_area(map, firing_rate_thres)
    #second, patchiness
    map_criteria['patchiness'] = get_patchiness(map)
    #third, spatial coherence
    map_criteria['spatial_coherence'] = get_spatial_cohenrence(map)
    #fourth, spatial information content
    map_criteria['spatial_information_content'] = spatial_information_content(map, FR_In_Position, Time_In_Position)
    
    return map, map_criteria

class adaptiveCANN2D(bp.dyn.NeuDyn):
  def __init__(self, length, tau=1., tauv=2, m0=1., k=8.1, a=0.5, A=10., J0=4.,
               z_min=-bm.pi, z_max=bm.pi, name=None):
    super(adaptiveCANN2D, self).__init__(size=(length, length), name=name)

    # parameters
    self.length = length
    self.tau = tau  # The synaptic time constant
    self.tauv = tauv  # The time constant of the adaptation
    self.m = tau/tauv*m0  # The adaptation strength
    self.k = k  # Degree of the rescaled inhibition
    self.a = a  # Half-width of the range of excitatory connections
    self.A = A  # Magnitude of the external input
    self.J0 = J0  # maximum connection value

    # feature space
    self.z_min = z_min
    self.z_max = z_max
    self.z_range = z_max - z_min
    self.x = bm.linspace(z_min, z_max, length)  # The encoded feature values
    self.rho = length / self.z_range  # The neural density
    self.dx = self.z_range / length  # The stimulus density

    # The connections
    self.conn_mat = self.make_conn()

    # variables
    self.r = bm.Variable(bm.zeros((length, length)))
    self.u = bm.Variable(bm.zeros((length, length)))
    self.v = bm.Variable(bm.zeros((length, length)))
    self.input = bm.Variable(bm.zeros((length, length)))

  def dist(self, d):
    v_size = bm.asarray([self.z_range, self.z_range])
    return bm.where(d > v_size / 2, v_size - d, d)

  def make_conn(self):
    x1, x2 = bm.meshgrid(self.x, self.x)
    value = bm.stack([x1.flatten(), x2.flatten()]).T

    @jax.vmap
    def get_J(v):
      d = self.dist(bm.abs(v - value))
      d = bm.linalg.norm(d, axis=1)
      # d = d.reshape((self.length, self.length))
      Jxx = self.J0 * bm.exp(-0.5 * bm.square(d / self.a)) / (bm.sqrt(2 * bm.pi) * self.a)
      return Jxx

    return get_J(value)

  def get_stimulus_by_pos(self, pos):
    assert bm.size(pos) == 2
    x1, x2 = bm.meshgrid(self.x, self.x)
    value = bm.stack([x1.flatten(), x2.flatten()]).T
    d = self.dist(bm.abs(bm.asarray(pos) - value))
    d = bm.linalg.norm(d, axis=1)
    d = d.reshape((self.length, self.length))
    return self.A * bm.exp(-0.25 * bm.square(d / self.a))

  def update(self):
    r1 = bm.square(self.u)
    r2 = 1.0 + self.k * bm.sum(r1)
    self.r.value = r1 / r2
    interaction = (self.r.flatten() @ self.conn_mat).reshape((self.length, self.length))
    self.u.value = self.u + (-self.u + self.input + interaction - self.v) / self.tau * bp.share['dt']
    self.u = bm.where(self.u>0, self.u, 0)
    self.v.value = self.v + (-self.v + self.m * self.u) / self.tauv * bp.share['dt']
    self.input[:] = 0.