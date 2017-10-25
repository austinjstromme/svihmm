# external packages
from numpy import random as nprand
from scipy.misc import logsumexp
from scipy.special import digamma as dg
from scipy.stats import norm
import numpy as np

# internals
import context
import Gaussian_impl as impl
from NormalInverseWishart import NormalInverseWishart
from NormalInverseChiSquared import NormalInverseChiSquared
from Distribution import Distribution
from Exponential import Exponential
import utils.LogMatrixUtil as lm

class Gaussian(Exponential):
  """
  Gaussian is a class representing a single variable Gaussian distribution. It
  implementes the Exponential interface.
 
  Attributes:
    mu: mean
    sigma: standard deviation
    prior: Normal-Inverse-Chi-Squared distribution.
  """

  def __init__(self, params, prior=None):
    """
    Initializes the distribution with given parameters and prior.

    Args:
      params: [mu, sigma]
      prior: Normal-Inverse-Chi-Squared distribution
    """
    if (len(params) != 2): raise ValueError("Gaussian initialization failed."
      + " Malformed arguments to the constructor.")

    self.mu = params[0]
    self.sigma = params[1]

    if prior == None:
      l = [0., 1., 2., 2.]
      prior = NormalInverseChiSquared(l)

  def gen_sample(self):
    """
    Generates a sample from this distribution.

    Returns:
      x: a sample from this.
    """
    return nprand.normal(self.mu, self.sigma)

  def get_natural(self):
    """
    Returns the natural parameters of this distribution.

    Returns:
      w: np.array of length L, natural parameters for this.
    """
    return np.array([self.mu, -.5])/(self.sigma**2)

  def set_natural(self, w):
    """
    Updates the parameters so the natural parameters become w.

    Args:
      w: np.array of length L of new natural parameters
    """
    if ((len(w.shape) != 1) or w.shape[0] != 2): raise ValueError("Malformed"
      + " arguments to set_natural")

    self.mu = (w[0]/(-2.*w[1]))
    self.sigma = np.pow(w[0]/self.mu, -0.5)

  def gen_log_expected(self):
    """
    Generates the log expected distribution according to the
    current prior.

    Returns:
      p: a distribution such that p(x) = exp(E[ln(q(x))]) where the expectation
      is over the distribution on q via the prior.

    NOTE: the returned distribution may only implement Distribution.py.
    """
    mu, sigmasq, kappa, nu = self.prior.get_params()

    mass = lambda x : np.exp(-0.5/kappa - 0.5*nu*
      ((x - mu)**2)/sigmasq)*((2*np.pi*sigmasq)**(-0.5))

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
    res = impl._GaussianSuffStats.get_stats(S, j, a, b)

    return res

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

  def mass(self, x):
    """
    Computes the probability of an observation x.

    Args:
      x: a single observation

    Returns:
      p(x)
    """
    return norm.pdf(x, self.mu, self.sigma)

  def maximize_likelihood(self, S, j):
    """
    Updates the parameters of this distribution to maximize the likelihood
    of it being the jth hidden state's emitter.

    Args:
      S: States object.
      j: the hidden state this distribution corresponds to.
    """
    T = len(S.data[0])

    mu, sigma = impl._GaussianSuffStats.maximize_likelihood_helper(
                S, j, 0, T - 1)

    self.mu = mu
    self.sigma = sigma

  def __str__(self):
    """
    String representation of the Gaussian
    """
    return ("mu, sigma = " + str(self.mu) + " " + str(self.sigma))
