# external packages
from numpy import random as nprand
from scipy.misc import logsumexp
from scipy.special import digamma as dg
from scipy.stats import norm
import numpy as np

# internals
from Gaussian_impl import _GaussianSuffStats
from NormalInverseWishart import NormalInverseWishart
from Exponential import Exponential
import LogMatrixUtil as lm

class Gaussian(Exponential):
  """
  Gaussian is a class representing a single variable Gaussian distribution.
  It implementes the Exponential interface.
 
  Attributes:
    mu: mean
    sigma: standard deviation
    prior: pointer to NIW prior.
  """

  def __init__(self, params, prior=None):
    """
    See Exponential.py. Params = [mu, sigma]; prior should be
    a Normal Inverse Wishart.
    """
    if (len(params) != 2): raise ValueError("Gaussian initialization failed."
      + " Malformed arguments to the constructor.")

    self.mu = params[0]
    self.sigma = params[1]

    if prior == None:
      l = [np.zeros(1), np.ones((1,1)), 1., 1.]
      prior = NormalInverseWishart(l)

  def gen_sample(self):
    """
    See Exponential.py.
    """
    return nprand.normal(self.mu, self.sigma)

  def get_natural(self):
    """
    See Exponential.py.
    """
    return np.array([self.mu, -.5])/(self.sigma**2)

  def set_natural(self, w):
    """
    See Exponential.py.
    """
    print(" w = " + str(w) + "  and w.shape = " + str(w.shape))
    if ((len(w.shape) != 1) or w.shape[0] != 2): raise ValueError("Malformed"
      + " arguments to set_natural")

    self.mu = (w[0]/(-2.*w[1]))
    self.sigma = np.pow(w[0]/self.mu, -0.5)

  def gen_log_expected(self):
    """
    Generates the log expected distribution according to the
    current prior.
    """
    # TODO: make sure this works; honestly should be able to just use NDGaussian
    mu, sigma, kappa, nu = self.prior.get_params()

    est_params = [mu, nu*sigma]

    return Gaussian(est_params)


  def get_expected_local_suff(self, S, j, a, b):
    """
    Returns the vector of expected sufficient statistics from subchain
    [a,b] according to a given states object; assums this dist is the one
    corresponding to the jth hidden state.
    """
  
    return _GaussianSuffStats.get_stats(S, j, a, b, 1)

  def get_expected_suff(self, S, j):
    """
    Returns the vector of the expected sufficient statistics from
    a given states object; assumes this dist is the one corresponding to
    the jth hidden state.
    """
    T = len(S.data[0])

    return self.get_expected_local_suff(S, j, 0, T - 1)

  def mass(self, x):
    """
    Computes the probability of an observation x.

    Args:
      x: a single observation

    Returns:
      p(x)
    """
    return norm.pdf(x, self.mu, self.sigma)

