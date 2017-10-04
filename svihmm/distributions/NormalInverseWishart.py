# external packages
import numpy as np
from scipy.special import digamma as dg
from scipy.special import gamma as Gamma

# internals
from Exponential import Exponential
from Gaussian_impl import _GaussianSuffStats

class NormalInverseWishart(Exponential):
  """
  A Normal Inverse Wishart distribution; functions as a prior for the
  1D and NDGaussian.

  Attributes:
    w: np.array of natural parameters in the form specified by
    _GaussianSuffStats
    dim: dimension of this NIW
    prior: NOT IMPLEMENTED
  """

  def __init__(self, params, prior=None):
    """
    Initializes the distribution with given parameters and prior.
    Prior defaults to sensible value if not specified. See
    specific distributions for more information.

    Assumes params is of the form [mu_0, sigma_0, kappa_0, nu_0]
    """
    self.dim = params[0].shape[0]
    self.w = _GaussianSuffStats.to_vec(params, self.dim)

    # priors are unsupported
    self.prior = None

  def gen_sample(self):
    """
    Generates a sample from this distribution.
    """
    raise NotImplementedError()

  def get_natural(self):
    """
    Returns the vector of natural parameters.
    """
    return self.w

  def set_natural(self, w):
    """
    Sets the vector of natural parameters.
    """
    self.w = w

  def get_params(self):
    """
    Returns a list of [mu_0, sigma_0, kappa_0, nu_0]
    """
    return _GaussianSuffStats.to_list(self.w, self.dim)

  def gen_log_expected(self):
    """
    Generates the log expected distribution according to the
    current prior.
    """
    raise NotImplementedError("NIWs have no prior implemented")

  def get_expected_suff(self, S, j):
    """
    Returns the vector of the expected sufficient statistics from
    a given states object; assume this dist is the one corresponding to
    the jth hidden state.
    """
    raise NotImplementedError()

  def KL_div(self, other):
    """
    Computes the KL divergence between self and other; i.e.
    KL(other || self).
    """
    mu_0, sigma_0, kappa_0, nu_0 = self.get_params()
    mu_k, sigma_k, kappa_k, nu_k = other.get_params()

    lambduh = other.lambduh_tilde()

    lambduh_k = np.linalg.inv(sigma_k)

    #res = self.dim*0.5*np.log(kappa_0/kappa_k) \
    #    - self.dim*(kappa_0/kappa_k - 0.5) \
    #    - kappa_0*nu_k*((mu_k - mu_0).transpose().dot(
    #        lambduh_k.dot(mu_k - mu_0))).sum() \
    #    + (nu_0 - self.dim - 1)*0.5*np.log(lambduh) \
    #    - 0.5*nu_k*(np.trace(sigma_0.dot(lambduh_k))) \
    #    + other.inv_wishart_entropy() \
    #    + self.inv_wishart_log_partition()

    res = -0.5*(np.log(lambduh) + self.dim*(np.log(kappa_k/(2*np.pi))
      - 1)) + other.inv_wishart_entropy() \
      + 0.5*(self.dim*np.log(kappa_0/(2*np.pi)) + np.log(lambduh) \
      - self.dim*kappa_0/kappa_k - kappa_0*nu_k*((mu_k
      - mu_0).transpose().dot(lambduh_k.dot(mu_k - mu_0))).sum()) \
      + self.inv_wishart_log_partition() \
      + (nu_0 - self.dim - 1)*0.5*np.log(lambduh) - 0.5*nu_k \
      *np.trace(lambduh_k.dot(sigma_0))

    return res

  def inv_wishart_log_partition(self):
    """
    Returns the log partition function of self's inverse wishart.
    """
    #TODO: verify correctness. Following pybasicbayes here
    mu, sigma, kappa, nu = self.get_params()
    D = self.dim
    return -(nu*0.5*np.log(np.linalg.det(sigma))
      - nu*D*0.5*np.log(2) - D*(D - 1.)*0.25*np.log(np.pi)
      - sum([np.log(abs(Gamma(0.5*(nu + 1. - i)))) for i in range(1, D + 1)]))

  def inv_wishart_entropy(self):
    """
    Returns the entropy of self's inverse wishart.
    """
    #TODO: verify correctness. Following pybasicbayes here
    mu, sigma, kappa, nu = self.get_params()
    D = self.dim
    
    log_lambduh = np.log(self.lambduh_tilde())
    return (self.inv_wishart_log_partition() - (nu - D - 1)*0.5*log_lambduh
      + nu*D*0.5)

  def lambduh_tilde(self):
    """
    Returns lambduh tilde; see Bishop 10.65
    """
    mu, sigma, kappa, nu = self.get_params()
    D = self.dim

    log_lambduh = (sum([dg(0.5*(nu + 1. - i)) for i in range(1, D + 1)])
      + D*np.log(2.) - np.log(np.linalg.det(sigma)))

    return np.exp(log_lambduh)

  def maximize_likelihood(self, S, j):
    """
    Updates the parameters of this distribution to maximize the likelihood
    of it being the jth hidden state's emitter.
    """
    raise NotImplementedError()

  def mass(self, x):
    """
    Computes the probability of an observation x.

    Args:
      x: a single observation

    Returns:
      p(x)
    """
    raise NotImplementedError()
