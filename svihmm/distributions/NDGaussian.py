# external packages
from numpy import random as nprand
from scipy.stats import multivariate_normal

# internal packages
from Exponential import Exponential
import Gaussian_impl as impl

class NDGaussian(Exponential):
  """
  NDGaussian is a class representing an N dimensional Gaussian, where
  N > 1.

  Attributes:
    mu - the mean; an N dimensional np.array
    sigma - the covariance matrix; an NxN np.array 
    prior - the prior Normal Inverse Wishart distribution
  """

  def __init__(self, params, prior=None):
    """
    Initializes the distribution with given parameters and prior.

    Assume params is of the form [mu, sigma]. If prior is not specified
    SAY WHAT IT DOES IF PRIOR UNSPECIFIED HERE
    """
    if ((len(params) != 2) or (len(params[0].shape) != 1)
      or (len(params[1].shape) != 2)
      or (params[0].shape[0] != params[1].shape[0])
      or (params[0].shape[0] != params[1].shape[1])):
      raise ValueError("NDGaussian initialization failed. Malformed arguments
        to the constructor")

    self.mu = params[0]
    self.sigma = params[1]

    if prior==None:
      raise NotImplementedError("Must implement NIWs")

    self.prior = prior

  def gen_sample(self):
    """
    Generates a sample from this distribution.
    """
    return nprand.multivariate_normal(self.mu, self.sigma)

  def get_natural(self):
    """
    Returns a list of the natural parameters, namely [sigma^(-1)*mu,
      -sigma^(-1)/2].
    """
    sigmainv = np.linalg.inv(self.sigma)
    return [np.dot(sigmainv, self.mu), (-0.5)*sigmainv]

  def set_natural(self, w):
    """
    Sets the vector of natural parameters. Expects that w is a
    list of the form [x, A] where x is of a N dimensional numpy
    array and A is an NxN numpy array.
    """
    if ((len(params) != 2) or (len(params[0].shape) != 1)
      or (len(params[1].shape) != 2)
      or (params[0].shape[0] != params[1].shape[0])
      or (params[0].shape[0] != params[1].shape[1])):
      raise ValueError("Invalid input into set_natural.")

    self.sigma = -2.*w[1]

    self.mu = self.sigma*w[0]

  def gen_log_expected(self):
    """
    Generates the log expected distribution according to the
    current prior. NOTE THIS IS CURRENTLY OFF THE CORRECT
    DISTRIBUTION BY A CONSTANT FACTOR; THIS SHOULD ONLY AFFECT
    THE VALUES OF STATES.LL() BY AN ADDITIVE CONSTANT; CHECK
    WITH NICK
    """
    mu, sigma, kappa, nu = self.prior.get_params()

    est_params = [mu, nu*sigma]

    return NDGaussian(est_params)

  def get_expected_local_suff(self, S, j, a, b):
    """
    Returns the vector of expected sufficient statistics from subchain
    [a,b] according to a given states object; assums this dist is the one
    corresponding to the jth hidden state.
    """
    return impl._GaussianSuffStats.get_stats(S, j, a, b, self.mu.shape[0])

  def get_expected_suff(self, S, j):
    """
    Returns the vector of the expected sufficient statistics from
    a given states object; assumes this dist is the one corresponding to
    the jth hidden state.
    """
    T = len(S.data[0])

    return self.get_expected_local_suff(S, j, 0, T - 1)

  def maximize_likelihood(self, S, j):
    """
    Updates the parameters of this distribution to maximize the likelihood
    of it being the jth hidden state's emitter.
    """
    T = len(S.data[0])
    dim = self.mu.shape[0]

    self.mu, self.sigma = impl.__maximize_likelihood(S, j, 0, T - 1, dim)

  def mass(self, x):
    """
    Computes the probability of an observation x.

    Args:
      x: a single observation

    Returns:
      p(x)
    """ 
    return multivariate_normal.pdf(x, self.mu, self.sigma)
