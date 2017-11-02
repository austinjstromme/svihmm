# external packages
import numpy as np
from scipy.special import digamma as dg
from scipy.special import gamma as Gamma

# internals
from .Exponential import Exponential
from .Gaussian_impl import _GaussianSuffStats

class NormalInverseChiSquared(Exponential):
  """
  A Normal (Scaled) Inverse chi-squared distribution; functions as a prior for
  Gaussian.

  Attributes:
    w: np.array of natural parameters in the form specified by
    _GaussianSuffStats
    prior: prior for this
  """

  def __init__(self, params, prior=None):
    """
    Initializes the distribution with given parameters and prior.

    Args:
      params: [mu_0, sigmasq_0, kappa_0, nu_0]
      prior: not implemented yet
    """
    self.w = _GaussianSuffStats.to_vec(params)

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
    normal_params = _GaussianSuffStats.to_list(self.w)
    natural_params = _GaussianSuffStats.NICS_normal_to_natural(normal_params)
    return _GaussianSuffStats.to_vec(natural_params)

  def set_natural(self, w):
    """
    Updates the parameters so the natural parameters become w.

    Args:
      w: np.array of length L of new natural parameters
    """
    natural_params = _GaussianSuffStats.to_list(w)
    normal_params = _GaussianSuffStats.NICS_natural_to_normal(natural_params)
    self.w = _GaussianSuffStats.to_vec(normal_params)

  def get_params(self):
    """
    Returns a list of [mu_0, sigma_0, kappa_0, nu_0]
    """
    return _GaussianSuffStats.to_list(self.w)

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
    """
    mu_0, sigmasq_0, kappa_0, nu_0 = self.get_params()
    mu_k, sigmasq_k, kappa_k, nu_k = other.get_params()

    #print("u_D params = " + str(self.get_params()))
    #print("D params = " + str(other.get_params()))

    # we use the fact that KL div between products of normals N_1, N_2 and
    # inverse chi squared distributions s_1, s_2, is equal to
    #
    #   KL(s_2N_2 \| s_1 N_1) = E_{s \sim s_2}[ KL( N_1 \| N_2)]
    #                          + KL(s_2 \| s_1)
    # 
    # We compute the first term:
    expected_KL_between_normals = 0.5*np.log(kappa_k/kappa_0) + 0.5*kappa_0/kappa_k \
      - 0.5 + 0.5*kappa_0*((mu_k - mu_0)**2)/sigmasq_k

    # For the second term we use the fact that the KL divergence between two
    # members of the same exponential family which has log partition g
    # is
    # 
    # KL(q \| p) = g(\theta_p) - g(\theta_q) - (\theta_p - \theta_q)\nabla g (\theta_q)
    #
    # see ``Entropies and cross-entropies of exponential families" by Nielsen
      # and Nock

    KL_between_inv_chi_sq = self.inv_chi_squared_log_partition()\
        - other.inv_chi_squared_log_partition() \
        + 0.5*(nu_0*sigmasq_0 - nu_k*sigmasq_k)/sigmasq_k \
        - 0.5*(nu_0 - nu_k)*(dg(nu_k/2) - np.log(nu_k*sigmasq_k*0.5))

    #KL = self.inv_chi_squared_log_partition() \
    #  - other.inv_chi_squared_log_partition() \
    #  + 0.5*np.log(kappa_k/kappa_0) \
    #  + (sigmasq_0 - sigmasq_k) + 0.5*(kappa_0 - kappa_k)/kappa_k \
    #  + (nu_0 - nu_k)*(0.5*(np.log(sigmasq_k*nu_k/2)) + 0.5 - dg(nu_k/2))

    return (expected_KL_between_normals + KL_between_inv_chi_sq)

  def inv_chi_squared_log_partition(self):
    """
    Log partition function of self's Inverse-Chi-Squared distribution.

    Returns:
      x: log partition of self's Inverse-Chi-Squared distribution.
    """
    mu, sigmasq, kappa, nu = self.get_params()

    return np.log(abs(Gamma(0.5*nu))) - nu*0.5*np.log(sigmasq*nu*0.5)

  def inv_chi_squared_entropy(self):
    """
    Entropy of self's Inverse-Chi-Squared distribution.

    Returns:
      x: entropy of self's Inverse-Chi-Squared distribution.
    """
    mu, sigmasq, kappa, nu = self.get_params()

    return (nu*0.5 + np.log(sigmasq*nu*0.5*abs(Gamma(0.5*nu)))
      - (1. + 0.5*nu)*dg(0.5*nu))

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
