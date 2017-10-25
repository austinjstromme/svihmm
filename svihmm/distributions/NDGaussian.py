# external packages
import numpy as np
from numpy import random as nprand
from scipy.stats import multivariate_normal

# internal packages
from Distribution import Distribution
from Exponential import Exponential
from NDGaussian_impl import _NDGaussianSuffStats as impl
from NormalInverseWishart import NormalInverseWishart

class NDGaussian(Exponential):
  """
  NDGaussian is a class representing an N dimensional Gaussian, where
  N > 1.

  Attributes:
    mu: the mean; an N dimensional np.array
    sigma: the covariance matrix; an NxN np.array 
    prior: the prior Normal Inverse Wishart distribution
  """

  def __init__(self, params, prior=None):
    """
    Initializes the distribution with given parameters and prior.

    Args:
      params: [mu, sigma]
      prior: NIW, if not specified is initialize to [0, I, 1, 1]
    """
    if ((len(params) != 2) or (len(params[0].shape) != 1)
      or (len(params[1].shape) != 2)
      or (params[0].shape[0] != params[1].shape[0])
      or (params[0].shape[0] != params[1].shape[1])):
      raise ValueError("Malformed arguments to NDGaussian constructor")

    self.mu = params[0]
    self.sigma = params[1]

    dim = self.mu.shape[0]

    if prior == None:
      prior = NormalInverseWishart([np.zeros(dim), np.eye(dim), 1., 1.])

    self.prior = prior

  def gen_sample(self):
    """
    Generates a sample from this distribution.

    Returns:
      x: a sample from this.
    """
    return nprand.multivariate_normal(self.mu, self.sigma)

  def get_natural(self):
    """
    Returns the natural parameters of this distribution.

    Returns:
      l: [sigma^{-1} mu, -0.5 sigma^{-1}]
    """
    sigmainv = np.linalg.inv(self.sigma)
    return [np.dot(sigmainv, self.mu), (-0.5)*sigmainv]

  def set_natural(self, w):
    """
    Updates the parameters so the natural parameters become w.

    Args:
      w: [sigma^{-1} mu, -0.5 sigma^{-1}]
    """
    if ((len(params) != 2) or (len(params[0].shape) != 1)
      or (len(params[1].shape) != 2)
      or (params[0].shape[0] != params[1].shape[0])
      or (params[0].shape[0] != params[1].shape[1])):
      raise ValueError("Invalid input into set_natural.")

    self.sigma = -0.5*np.linalg.inv(w[1])

    self.mu = self.sigma*w[0]

  def gen_log_expected(self):
    """
    Generates the log expected distribution according to the
    current prior.

    Returns:
      p: a distribution such that p(x) = exp(E[ln(q(x))]) where the expectation
      is over the distribution on q via the prior.

    NOTE: the returned distribution may only implement Distribution.py.
    """
    mu, sigma, kappa, nu = self.prior.get_params()

    D = self.mu.shape[0]
    lambduh = self.prior.lambduh_tilde()
    precision = np.linalg.inv(sigma)

    mass = lambda x : ((lambduh**0.5)*np.exp(-D/(2*kappa) - 0.5*nu*
      (np.inner(x - mu, precision.dot(x - mu))))*((2*np.pi)**(-D/2)))

    return Distribution(mass)

  def get_expected_local_suff(self, S, j, a, b):
    """
    Returns the vector of expected sufficient statistics from subchain
    [a,b].

    Args:
      S: States object.
      j: the hidden state this distribution corresponds to.
      a: state of the subchain.
      b: end of the subchain.

    Returns:
      w: vector of expected sufficient statistics
    """
    l = impl.to_list(impl.get_stats(S, j, a, b, self.mu.shape[0]),
      self.mu.shape[0])
    return impl.to_vec(impl.NIW_normal_to_natural(l), self.mu.shape[0])

  def get_expected_suff(self, S, j):
    """
    Returns the vector of the expected sufficient statistics from
    a given states object; assume this dist is the one corresponding to
    the jth hidden state.

    Args:
      S: States object.
      j: the hidden state this distribution corresponds to.

    Returns:
      w: a np.array of length L where is the number of parameters of the prior.
    """
    T = len(S.data[0])

    return self.get_expected_local_suff(S, j, 0, T - 1)

  def maximize_likelihood(self, S, j):
    """
    Updates the parameters of this distribution to maximize the likelihood
    of it being the jth hidden state's emitter.

    Args:
      S: States object.
      j: the hidden state this distribution corresponds to.
    """
    T = len(S.data[0])
    dim = self.mu.shape[0]

    self.mu, self.sigma = impl.maximize_likelihood_helper(S, j, 0, T - 1, dim)

  def mass(self, x):
    """
    Computes the probability of an observation x.

    Args:
      x: a single observation

    Returns:
      p(x)
    """ 
    return multivariate_normal.pdf(x, self.mu, self.sigma)
