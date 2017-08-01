# external packages
import numpy as np

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
    return 0.0

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
