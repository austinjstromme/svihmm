# external packages
import numpy as np
from scipy.special import digamma as dg
from scipy.special import gamma as Gamma

# internals
from .Exponential import Exponential
from .NDGaussian_impl import _NDGaussianSuffStats

class NormalInverseWishart(Exponential):
  """
  A Normal Inverse Wishart distribution; functions as a prior for NDGaussian.

  Attributes:
    w: np.array of natural parameters in the form specified by
    _NDGaussianSuffStats
    dim: dimension of this NIW
    prior: prior for this
  """

  def __init__(self, params, prior=None):
    """
    Initializes the distribution with given parameters and prior.

    Args:
      params: [mu_0, sigma_0, kappa_0, nu_0]
      prior: not implemented yet
    """
    self.dim = params[0].shape[0]
    self.w = _NDGaussianSuffStats.to_vec(params, self.dim)

    # priors are unsupported
    self.prior = None

  def gen_sample(self):
    """
    Generates a sample from this distribution.

    Returns:
      x: a sample from this.
    """
    raise NotImplementedError()

  def get_natural(self):
    """
    Returns the natural parameters of this distribution.

    Returns:
      w: np.array of length L, natural parameters for this.
    """
    normal_params = _NDGaussianSuffStats.to_list(self.w, self.dim)
    natural_params = _NDGaussianSuffStats.NIW_normal_to_natural(normal_params)
    return _NDGaussianSuffStats.to_vec(natural_params, self.dim)

  def set_natural(self, w):
    """
    Updates the parameters so the natural parameters become w.

    Args:
      w: np.array of length L of new natural parameters
    """
    natural_params = _NDGaussianSuffStats.to_list(w, self.dim)
    normal_params = _NDGaussianSuffStats.NIW_natural_to_normal(natural_params)
    self.w = _NDGaussianSuffStats.to_vec(normal_params, self.dim)

  def get_params(self):
    """
    Returns the normal parameters.

    Return:
      l: [mu_0, sigma_0, kappa_0, nu_0]
    """
    return _NDGaussianSuffStats.to_list(self.w, self.dim)

  def gen_log_expected(self):
    """
    Generates the log expected distribution according to the
    current prior.

    Returns:
      p: a distribution such that p(x) = exp(E[ln(q(x))]) where the expectation
      is over the distribution on q via the prior.

    NOTE: the returned distribution may only implement Distribution.py.
    """
    raise NotImplementedError("NIWs have no prior implemented")

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
    raise NotImplementedError()

  def KL_div(self, other):
    """
    Computes the KL divergence between self and other; i.e.
      KL(other || self).

    Returns:
      x: KL(other || self).
    """
    mu_0, sigma_0, kappa_0, nu_0 = self.get_params()
    mu_k, sigma_k, kappa_k, nu_k = other.get_params()

    lambduh = other.lambduh_tilde()

    lambduh_k = np.linalg.inv(sigma_k)

    # following Matt Johnson's code in pybasicbayes here

    q_entropy = -0.5*(np.log(lambduh) + self.dim*(np.log(kappa_k/(2*np.pi))
      - 1)) + other.inv_wishart_entropy()
    p_avgengy = 0.5*(self.dim*np.log(kappa_0/(2*np.pi)) + np.log(lambduh) \
      - self.dim*kappa_0/kappa_k - kappa_0*nu_k*((mu_k
      - mu_0).transpose().dot(lambduh_k.dot(mu_k - mu_0))).sum()) \
      + self.inv_wishart_log_partition() \
      + (nu_0 - self.dim - 1)/(2.*np.log(lambduh)) - 0.5*nu_k \
      *np.trace(lambduh_k.dot(sigma_0))

    return -(p_avgengy + q_entropy)

  def inv_wishart_log_partition(self):
    """
    Log partition function of self's Inverse-Chi-Squared distribution.

    Returns:
      x: log partition of self's Inverse-Chi-Squared distribution.
    """
    mu, sigma, kappa, nu = self.get_params()
    D = self.dim
    return -(nu*0.5*np.log(np.linalg.det(sigma))
      - nu*D/2.*np.log(2) - D*(D - 1.)/4.*np.log(np.pi)
      - sum([np.log(abs(Gamma(0.5*(nu + 1. - i)))) for i in range(1, D + 1)]))

  def inv_wishart_entropy(self):
    """
    Entropy of self's Inverse-Chi-Squared distribution.

    Returns:
      x: entropy of self's Inverse-Chi-Squared distribution.
    """
    mu, sigma, kappa, nu = self.get_params()
    D = self.dim
    
    log_lambduh = np.log(self.lambduh_tilde())
    return (self.inv_wishart_log_partition() - (nu - D - 1)/2.*log_lambduh
      + nu*D*0.5)

  def lambduh_tilde(self):
    """
    Computes lambduh tilde; see Bishop 10.65.

    Returns:
      x: lambduh_tilde
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

    Args:
      S: States object.
      j: the hidden state this distribution corresponds to.
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
