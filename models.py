import jax
import brainpy as bp
import brainpy.math as bm

class PC_cell(bp.DynamicalSystem):
    def __init__(
        self,
        num=100,
        tau=10.0,
        tauv=100.0,
        m0=10.0,
        k=1.0,
        a=0.08,
        goal_a = 0.08, 
        A=10.0,
        J0=4.0,
        goal_J0 = 500.,
        z_min=0,
        z_max=1,
        goal_loc=None,
    ):

        super(PC_cell, self).__init__()

        # Hyper-parameters
        self.num = num  # number of neurons at each dimension
        self.tau = tau  # The synaptic time constant
        self.tauv = tauv  # The time constant of firign rate adaptation
        self.m = tau / tauv * m0  # The adaptation strength
        self.k = k  # Degree of the rescaled inhibition
        self.a = a  # Half-width of the range of excitatory connections
        self.A = A  # Magnitude of the external input
        self.J0 = J0  # maximum connection value
        self.goal_J0 = goal_J0
        self.goal_a = goal_a

        # feature space
        self.z_range = z_max - z_min
        linspace_z = bm.linspace(z_min, z_max, num + 1)
        self.z = linspace_z[:-1]
        x, y = bm.meshgrid(self.z, self.z)  # x y index
        self.value_index = bm.stack([x.flatten(), y.flatten()]).T

        # Synaptic connections
        self.conn_mat = self.make_conn()
        
        if goal_loc is not None:
            self.goal_loc = bm.array(goal_loc).reshape(1,2)
            self.gd_conn = self.make_gd_conn(self.goal_loc)  
            self.conn_mat = self.conn_mat + self.gd_conn

        # Define variables we want to update
        self.r = bm.Variable(bm.zeros((num, num)))  # firing rate of all PCs
        self.u = bm.Variable(bm.zeros((num, num)))  # presynaptic input of all PCs
        self.v = bm.Variable(bm.zeros((num, num)))  # firing rate adaptation of all PCs
        self.center = bm.Variable(bm.zeros(2))  # center of the bump
        self.loc_input = bm.Variable(
            bm.zeros((num, num))
        )  # Location dependent sensory input to the networks

        # define the integrator
        self.integral = bp.odeint(method="exp_euler", f=self.derivative)

    @property
    def derivative(self):
        du = lambda u, t, Irec: (-u + Irec + self.loc_input - self.v) / self.tau
        dv = lambda v, t: (-v + self.m * self.u) / self.tauv
        return bp.JointEq([du, dv])

    def dist(self, d):
        v_size = bm.asarray([self.z_range, self.z_range])
        return bm.where(d > v_size / 2, v_size - d, d)

    def make_conn(self):
        @jax.vmap
        def get_J(v):
            d = self.dist(bm.abs(v - self.value_index))
            d = bm.linalg.norm(d, axis=1)
            # d = d.reshape((self.length, self.length))
            Jxx = (
                self.J0
                * bm.exp(-0.5 * bm.square(d / self.a))
                / (bm.sqrt(2 * bm.pi) * self.a)
            )
            return Jxx

        return get_J(self.value_index)

    def make_gd_conn(self, goal_loc):
        #add a goal directed connection to the neurons at the goal location with a Gaussian profile
        @jax.vmap
        def get_J(v):
            d = self.dist(bm.abs(v - goal_loc))
            d = bm.linalg.norm(d, axis=1)
            Jxx = (
                self.goal_J0
                * bm.exp(-0.5 * bm.square(d / self.goal_a))
                / (bm.sqrt(2 * bm.pi) * self.goal_a)
            )
            return Jxx
        
        conn_vec = get_J(self.value_index)
        
        '''
        #asymmetric connection of a neuron
        #find the cloest index in self.value_grid to the goal location
        distances = bm.linalg.norm(self.goal_loc - self.value_index, axis=1)
        closest_index = bm.argmin(distances)
        
        #geterante a zeros matrix the same size as self.conn_mat, and out only the closest_index column as 
        goal_conn_mat = bm.zeros_like(self.conn_mat)
        goal_conn_mat[:,closest_index] = conn_vec.reshape(-1,)
        '''
        
        #asymmetric connection of multiple neurons
        d = self.dist(bm.abs(self.goal_loc - self.value_index))
        distances = bm.linalg.norm(d, axis=1)
        #rank the distances from low to high and get the index
        closest_index = bm.argsort(distances)
        goal_conn_mat = bm.zeros_like(self.conn_mat)
        for i, index in enumerate(closest_index):
            # if i < 5000:
            if True:
                rank_dist = distances[index]
                alpha = 1 - rank_dist / (bm.max(distances)-bm.min(distances))
                goal_conn_mat[:,index] = conn_vec.reshape(-1,) * alpha 
        
        return goal_conn_mat

    def location_input(self, animal_loc, theta_mod):
        # return bump input (same dim as neuroal space) from a x-y location

        assert bm.size(animal_loc) == 2

        d = self.dist(bm.abs(bm.asarray(animal_loc) - self.value_index))
        d = bm.linalg.norm(d, axis=1)
        d = d.reshape((self.num, self.num))
        # Gaussian bump input
        input_ = self.A * bm.exp(-0.25 * bm.square(d / self.a))

        # further theta modulation
        input_ = input_ * theta_mod

        return input_

    def update(self, Animal_location, ThetaModulator):

        self.loc_input = self.location_input(Animal_location, ThetaModulator)
        
        Irec = bm.matmul(self.conn_mat, self.r.flatten()).reshape((self.num, self.num))

        # update the system
        u, v = self.integral(self.u, self.v, None, Irec)

        self.u.value = bm.where(u > 0, u, 0)
        self.v.value = v
        r1 = bm.square(self.u)
        r2 = 1.0 + self.k * bm.sum(r1)
        self.r.value = r1 / r2
        
        #get the center of the bump from self.r which is num x num matrix
        self.center[1], self.center[0] = bm.unravel_index(bm.argmax(self.r.value), [self.num, self.num])
        

class PC_cell_topdown(bp.DynamicalSystem):
    def __init__(
        self,
        num=100,
        tau=10.0,
        tauv=100.0,
        m0=10.0,
        k=1.0,
        a=0.08,
        A=10.0,
        J0=4.0,
        goal_a=0.5,
        goal_A=1.0,
        z_min=0,
        z_max=1,
        conn_noise=0.,
        rec_noise=0.5,
        goal_loc=(0.75, 0.75),
        topdown=False,
    ):

        super(PC_cell_topdown, self).__init__()

        # Hyper-parameters
        self.num = num  # number of neurons at each dimension
        self.tau = tau  # The synaptic time constant
        self.tauv = tauv  # The time constant of firign rate adaptation
        self.m = tau / tauv * m0  # The adaptation strength
        self.k = k  # Degree of the rescaled inhibition
        self.a = a  # Half-width of the range of excitatory connections
        self.A = A  # Magnitude of the external input
        self.J0 = J0  # maximum connection value
        self.topdown = topdown #if turn on the topdown input
        self.rec_noise = rec_noise #the noise level of the recurrent connections
        self.goal_loc = goal_loc #the location of the goal
        self.goal_a = goal_a #the half-width of the range of top down connections from the reward cell
        self.goal_A = goal_A #the magnitude of the top down input
        # feature space
        self.z_range = z_max - z_min
        linspace_z = bm.linspace(z_min, z_max, num + 1)
        self.z = linspace_z[:-1]
        x, y = bm.meshgrid(self.z, self.z)  # x y index
        self.value_index = bm.stack([x.flatten(), y.flatten()]).T

        # Synaptic connections
        conn_mat = self.make_conn()
        #add noise to connection matrix
        gaussiannoise_matrix = bm.random.normal(loc=0, scale=conn_noise, size=(self.num**2, self.num**2))
        self.conn_mat = conn_mat + gaussiannoise_matrix
        # Define variables we want to update
        self.r = bm.Variable(bm.zeros((num, num)))  # firing rate of all PCs
        self.u = bm.Variable(bm.zeros((num, num)))  # presynaptic input of all PCs
        self.v = bm.Variable(bm.zeros((num, num)))  # firing rate adaptation of all PCs
        self.center = bm.Variable(bm.zeros(2))  # center of the bump
        self.loc_input = bm.Variable(bm.zeros((num, num)))  # Location dependent sensory input to the networks
        self.td_input = bm.Variable(bm.zeros((num, num))) # Top down goal location input to the networks
        
        # define the integrator
        self.integral = bp.odeint(method="exp_euler", f=self.derivative)

    def dist(self, d):
        v_size = bm.asarray([self.z_range, self.z_range])
        return bm.where(d > v_size / 2, v_size - d, d)

    def make_conn(self):
        @jax.vmap
        def get_J(v):
            d = self.dist(bm.abs(v - self.value_index))
            d = bm.linalg.norm(d, axis=1)
            # d = d.reshape((self.length, self.length))
            Jxx = (
                self.J0
                * bm.exp(-0.5 * bm.square(d / self.a))
                / (bm.sqrt(2 * bm.pi) * self.a)
            )
            return Jxx

        return get_J(self.value_index)

    def location_input(self, animal_loc, theta_mod):
        # return bump input (same dim as neuroal space) from a x-y location

        assert bm.size(animal_loc) == 2

        d = self.dist(bm.abs(bm.asarray(animal_loc) - self.value_index))
        d = bm.linalg.norm(d, axis=1)
        d = d.reshape((self.num, self.num))
        # Gaussian bump input
        input_ = self.A * bm.exp(-0.25 * bm.square(d / self.a))

        # further theta modulation
        input_ = input_ * theta_mod

        return input_

    def topdown_input(self, Topdown_mod):
        # return bump input (same dim as neuroal space) from a x-y location
        assert bm.size(self.goal_loc) == 2

        d = self.dist(bm.abs(bm.asarray(self.goal_loc) - self.value_index))
        d = bm.linalg.norm(d, axis=1)
        d = d.reshape((self.num, self.num))
        # Gaussian bump input
        input_ = self.goal_A * bm.exp(-0.25 * bm.square(d / self.goal_a))

        # further theta modulation
        input_ = input_ * Topdown_mod

        return input_

    @property
    def derivative(self):
        du = lambda u, t, Irec: (-u + Irec + self.total_input - self.v) / self.tau
        dv = lambda v, t: (-v + self.m * self.u) / self.tauv
        return bp.JointEq([du, dv])

    def update(self, Animal_location, ThetaModulator_BU, Topdown_mod):

        #bottom up sensory input
        self.loc_input = self.location_input(Animal_location, ThetaModulator_BU)
        
        if self.topdown: #top down control
            self.td_input = self.topdown_input(Topdown_mod)
            self.total_input = self.loc_input + self.td_input
        else:
            self.total_input = self.loc_input
        
        Irec = bm.matmul(self.conn_mat, self.r.flatten()).reshape((self.num, self.num)) + self.rec_noise * bm.random.randn(self.num, self.num)

        # update the system
        u, v = self.integral(self.u, self.v, None, Irec)

        self.u.value = bm.where(u > 0, u, 0)
        self.v.value = v
        r1 = bm.square(self.u)
        r2 = 1.0 + self.k * bm.sum(r1)
        self.r.value = r1 / r2
        
        #get the center of the bump from self.r which is num x num matrix
        self.center[1], self.center[0] = bm.unravel_index(bm.argmax(self.r.value), [self.num, self.num])
        
