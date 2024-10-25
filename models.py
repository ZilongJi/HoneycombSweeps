
import jax
import brainpy as bp
import brainpy.math as bm

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