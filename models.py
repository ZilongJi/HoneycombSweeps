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
        
class PC_cell_topdown_asym(bp.DynamicalSystem):
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
        goal_b=0.5,
        goal_A=1.0,
        asym_J0 = 2,
        asym_a = 0.5,
        z_min=0,
        z_max=1,
        conn_noise=0.,
        rec_noise=0.5,
        goal_loc=(0.75, 0.75),
        topdown=False,
        asymmetry=False,
        tdstyle='Gaussian'
    ):

        super(PC_cell_topdown_asym, self).__init__()

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
        self.goal_b = goal_b
        self.goal_A = goal_A #the magnitude of the top down input
        self.tdstyle = tdstyle
        
        self.asym_J0 = asym_J0
        self.sym_a = asym_a
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
        
        if asymmetry is True:
            self.goal_loc = bm.array(goal_loc).reshape(1,2)
            self.gd_conn = self.make_gd_conn(self.goal_loc)  
            self.conn_mat = self.conn_mat + self.gd_conn
        
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

    def make_gd_conn(self, goal_loc):
        #add a goal directed connection to the neurons at the goal location with a Gaussian profile
        @jax.vmap
        def get_J(v):
            d = self.dist(bm.abs(v - goal_loc))
            d = bm.linalg.norm(d, axis=1)
            Jxx = (
                self.asym_J0
                * bm.exp(-0.5 * bm.square(d / self.sym_a))
                / (bm.sqrt(2 * bm.pi) * self.sym_a)
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

    def topdown_input(self, Topdown_mod, tdstyle='Gaussian'):
        # return bump input (same dim as neuroal space) from a x-y location
        assert bm.size(self.goal_loc) == 2

        d = self.dist(bm.abs(bm.asarray(self.goal_loc) - self.value_index))
        d = bm.linalg.norm(d, axis=1)
        d = d.reshape((self.num, self.num))
        if tdstyle == 'Gaussian':
            # Gaussian bump input
            input_ = self.goal_A * bm.exp(-0.25 * bm.square(d / self.goal_a))
        elif tdstyle == 'linear':
            # linear bump input (not Gaussian), which just decay with the distance
            input_ = self.goal_A * (1 - d / bm.max(d)*self.goal_b)

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
            self.td_input = self.topdown_input(Topdown_mod, tdstyle=self.tdstyle)
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
        
class HD_cell_L1(bp.DynamicalSystem):
    def __init__(
        self,
        num,
        noise_stre=0.01,
        tau=1.0,
        tau_v=10.0,
        k=1.0,
        mbar=15.,
        a=0.4,
        A=3.0,
        J0=4.0,
        z_min=-bm.pi,
        z_max=bm.pi,
        goal_a=0.5,
        goal_A=1.0,
        goal_dir=0,
        topdown=False,
    ):
        super(HD_cell_L1, self).__init__()

        # parameters
        self.tau = tau  # The synaptic time constant
        self.tau_v = tau_v
        self.k = k / num * 20  # Degree of the rescaled inhibition
        self.a = a  # Half-width of the range of excitatory connections
        self.A = A  # Magnitude of the external input
        self.J0 = J0 / num * 20  # maximum connection value
        self.m = mbar * tau / tau_v
        self.noise_stre = noise_stre

        # neuron num
        self.num = num  # head-direction cell
        # feature space
        self.z_min = z_min
        self.z_max = z_max
        self.z_range = z_max - z_min
        x1 = bm.linspace(z_min, z_max, num + 1)  # The encoded feature values
        self.x = x1[0:-1]
        
        self.topdown = topdown #if turn on the topdown input
        self.goal_dir = goal_dir #direction to the goal
        self.goal_a = goal_a #the half-width of the range of top down connections from the reward cell
        self.goal_A = goal_A #the magnitude of the top down input       

        # connection matrix
        conn_mat = self.make_conn()
        self.conn_fft = bm.fft.fft(conn_mat)

        # neuron state variables
        self.r = bm.Variable(bm.zeros(num))
        self.u = bm.Variable(bm.zeros(num))  # head direction cell
        self.v = bm.Variable(bm.zeros(num))
        self.input = bm.Variable(bm.zeros(num))
        self.center = bm.Variable(bm.zeros(1))
        self.centerI = bm.Variable(bm.zeros(1))
        self.total_input = bm.Variable(bm.zeros(num))

        # 定义积分器
        self.integral = bp.odeint(method="exp_euler", f=self.derivative)

    def topdown_input(self, Topdown_mod):
        # return bump input which is a one-dim bump (same dim as neuroal space)
        assert bm.size(self.goal_dir) == 1

        d = self.dist(bm.asarray(self.goal_dir) - self.x)
        # d = bm.linalg.norm(d)
        # Gaussian bump input
        input_ = self.goal_A * bm.exp(-0.25 * bm.square(d / self.goal_a))

        # further topdown modulation
        input_ = input_ * Topdown_mod

        return input_

    def dist(self, d):
        d = self.circle_period(d)
        d = bm.where(d > 0.5 * self.z_range, d - self.z_range, d)
        return d

    def make_conn(self):
        d = self.dist(bm.abs(self.x[0] - self.x))
        Jxx = self.J0 * bm.exp(-0.5 * bm.square(d / self.a)) / (2 * bm.pi * self.a ** 2)
        return Jxx

    def get_center(self, r, x):
        exppos = bm.exp(1j * x)
        center = bm.angle(bm.sum(exppos * r))
        return center.reshape(-1,)

    @property
    def derivative(self):
        du = lambda u, t, input: (-u + input - self.v) / self.tau
        dv = lambda v, t: (-v + self.m * self.u) / self.tau_v
        return bp.JointEq([du, dv])

    def circle_period(self, A):
        B = bm.where(A > bm.pi, A - 2 * bm.pi, A)
        B = bm.where(B < -bm.pi, B + 2 * bm.pi, B)
        return B

    def input_HD(self, HD):
        # integrate self motion
        return self.A * bm.exp(-0.25 * bm.square(self.dist(self.x - HD) / self.a))

    def reset_state(self, HD_truth):
        self.r.value = bm.Variable(bm.zeros(self.num))
        self.u.value = bm.Variable(bm.zeros(self.num))
        self.v.value = bm.Variable(bm.zeros(self.num))
        self.center.value = bm.Variable(bm.zeros(1,) + HD_truth)

    def update(self, HD, ThetaInput, Topdown_mod):
        
        
        self.center = self.get_center(r=self.r, x=self.x)
        Iext = ThetaInput * self.input_HD(HD)
        
        if self.topdown: #top down control
            self.td_input = self.topdown_input(Topdown_mod)
            self.total_input = Iext + self.td_input
        else:
            self.total_input = Iext
        
        # Calculate input
        r_fft = bm.fft.fft(self.r)
        Irec = bm.real(bm.fft.ifft(r_fft * self.conn_fft))
        input_total = self.total_input + Irec + bm.random.randn(self.num) * self.noise_stre
        
        # Update neural state
        u, v = self.integral(self.u, self.v, bp.share.load("t"), input_total, bm.dt)
        
        self.u.value = bm.where(u > 0, u, 0)
        self.v.value = v
        r1 = bm.square(self.u)
        r2 = 1.0 + self.k * bm.sum(r1)
        self.r.value = r1 / r2
        
class PC_cell_L2(bp.DynamicalSystem):
    def __init__(
        self,
        noise_stre=0.,
        num=100,
        tau=10.0,
        tau_v=100.0,
        mbar=75.0,
        a=0.5,
        A=1.0,
        J0=5.0,
        k=1,
        g = 1000,
        x_min=-bm.pi,
        x_max=bm.pi,
        num_hd = 100,
        Phase_Offset = 1/9,
    ):
        super(PC_cell_L2, self).__init__()

        # dynamics parameters
        self.tau = tau  # The synaptic time constant
        self.tau_v = tau_v  # The time constant of the adaptation variable
        self.num_x = num  # number of excitatory neurons for x dimension
        self.num_y = num  # number of excitatory neurons for y dimension
        self.num = self.num_x * self.num_y
        self.num_hd = num_hd
        self.k = k  # Degree of the rescaled inhibition
        self.a = a  # Half-width of the range of excitatory connections
        self.A = A  # Magnitude of the external input
        self.g = g
        self.J0 = J0/g  # maximum connection value
        self.m = mbar * tau / tau_v
        self.noise_stre = noise_stre
        self.Phase_Offset = Phase_Offset

        # feature space
        self.x_range = x_max - x_min
        phi_x = bm.linspace(x_min, x_max, self.num_x + 1)  # The encoded feature values
        self.x = phi_x[0:-1]
        self.y_range = self.x_range
        phi_y = bm.linspace(x_min, x_max, self.num_y + 1)  # The encoded feature values
        self.y = phi_y[0:-1]
        x_grid, y_grid = bm.meshgrid(self.x, self.y)
        self.x_grid = x_grid.flatten()
        self.y_grid = y_grid.flatten()
        self.value_grid = bm.stack([self.x_grid, self.y_grid]).T

        # initialize conn matrix
        self.conn_mat = self.make_conn()
        
        # initialize dynamical variables
        self.r = bm.Variable(bm.zeros(self.num))
        self.u = bm.Variable(bm.zeros(self.num))
        self.v = bm.Variable(bm.zeros(self.num))
        self.input = bm.Variable(bm.zeros(self.num))
        self.center_I = bm.Variable(bm.zeros(2))
        self.center_bump = bm.Variable(bm.zeros(2))

        # 定义积分器
        self.integral = bp.odeint(method="exp_euler", f=self.derivative)

    def circle_period(self, d):
        d = bm.where(d > bm.pi, d - 2 * bm.pi, d)
        d = bm.where(d < -bm.pi, d + 2 * bm.pi, d)
        return d

    def dist(self, d):
        dis = self.circle_period(d)
        delta_x = dis[:, 0]
        delta_y = dis[:, 1]
        dis = bm.sqrt(delta_x ** 2 + delta_y ** 2)
        return dis

    def make_conn(self):
        @jax.vmap
        def get_J(v):
            d = self.dist(v - self.value_grid)
            Jxx = (
                self.J0
                * bm.exp(-0.5 * bm.square(d / self.a))
                / (bm.sqrt(2 * bm.pi) * self.a)
            )
            return Jxx

        return get_J(self.value_grid)

    def Postophase(self, pos):
        phase = pos + bm.pi  # 坐标变换
        phase_x = bm.mod(phase[0], 2 * bm.pi) - bm.pi
        phase_y = bm.mod(phase[1], 2 * bm.pi) - bm.pi
        Phase = bm.array([phase_x, phase_y])
        return Phase
    

    def input_by_conjG(self, Animal_location, HD_activity, ThetaModulator, HD_truth):
        assert bm.size(Animal_location) == 2
        num_hd = self.num_hd
        hd = bm.linspace(-bm.pi,bm.pi,num_hd) 
        # each head-direction cell corresponds to a group of Conjunctive grid cells, which in turn projects to pure grid cells with assymetric connections determined by offset(hd)
        lagvec = -bm.array([bm.cos(HD_truth), bm.sin(HD_truth)]) * self.Phase_Offset * 1.4
        offset = bm.array([bm.cos(hd), bm.sin(hd)]) * self.Phase_Offset + 0*lagvec.reshape(-1,1)
        self.center_conjG = self.Postophase(
            Animal_location.reshape([-1,1]) + offset.reshape(-1,num_hd)
        )  # Ideal phase using mapping function
        input = bm.zeros([num_hd, self.num])
        for i in range(num_hd):
            d = self.dist(bm.asarray(self.center_conjG[:,i]) - self.value_grid)
            input[i] = self.A * bm.exp(-0.25 * bm.square(d / self.a))
            
        max_hd = bm.max(HD_activity)
        hd_weight = bm.where(HD_activity>max_hd/3, HD_activity, 0)
        hd_weight = hd_weight/bm.sum(hd_weight)

        total_input = bm.matmul(input.transpose(), hd_weight).reshape(-1,) * ThetaModulator
        
        #add animal location as a sensory inout which do not receive theta modulation
        loc_ = self.Postophase(Animal_location)
        d = self.dist(bm.asarray(loc_) - self.value_grid)
        loc_input = self.A * bm.exp(-0.25 * bm.square(d / self.a))
        total_input = total_input + loc_input
        
        return total_input


    def get_center(self):
        exppos_x = bm.exp(1j * self.x_grid)
        exppos_y = bm.exp(1j * self.y_grid)
        r = bm.where(self.r > bm.max(self.r) * 0.1, self.r, 0)
        
        #get the center of the activity bump
        self.center_bump[0] = bm.angle(bm.sum(exppos_x * r))
        self.center_bump[1] = bm.angle(bm.sum(exppos_y * r))
        
        #get the center of the moddle layer input (offset input)
        self.center_I[0] = bm.angle(bm.sum(exppos_x * self.input))
        self.center_I[1] = bm.angle(bm.sum(exppos_y * self.input))


    @property
    def derivative(self):
        du = (
            lambda u, t, Irec: (
                -u
                + Irec
                + self.input
                - self.v
            )
            / self.tau
        )
        dv = lambda v, t: (-v + self.m * self.u) / self.tau_v
        return bp.JointEq([du, dv])

    def reset_state(self):
        self.r.value = bm.Variable(bm.zeros(self.num))
        self.u.value = bm.Variable(bm.zeros(self.num))
        self.v.value = bm.Variable(bm.zeros(self.num))
        self.input.value = bm.Variable(bm.zeros(self.num))

    def update(self, Animal_location, HD_activity, ThetaModulator, HD_truth):
        #update the center and store them
        self.get_center()
        
        #
        input_conjG = self.input_by_conjG(Animal_location, HD_activity, ThetaModulator, HD_truth)
        
        self.input = input_conjG
        
        Irec = bm.matmul(self.conn_mat, self.r) + self.noise_stre * bm.random.randn(
            (self.num)
        )
        # Update neural state
        u, v = self.integral(self.u, self.v, bp.share["t"], Irec, bm.dt)
        self.u.value = bm.where(u > 0, u, 0)
        self.v.value = v
        r1 = bm.square(self.u)
        r2 = 1.0 + self.k * bm.sum(r1)
        self.r.value = self.g*r1 / r2