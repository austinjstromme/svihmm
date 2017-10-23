# external packages
import numpy as np

class _GaussianSuffStats():
  """
  A static implementation helper class. Converts between
  a list of sufficient statistics and a single numpy array.
  """

  @staticmethod
  def NICS_normal_to_natural(l):
    """
    Takes in a list of normal params of NICS (i.e. [mu_0, sigmasq_0, kappa_0,
      nu_0]) and returns the list of natural parmas
    """
    mu, sigmasq, kappa, nu = l
    return [kappa*mu, nu*sigmasq + (mu**2)*kappa, kappa, nu]

  @staticmethod
  def NICS_natural_to_normal(l):
    """
    Takes in a list of natural params of NIW (i.e. [e1, e2, e3, e4])
      and returns the list of normal params
    """
    e1, e2, e3, e4 = l
    kappa = e3
    nu = e4
    mu = e1/kappa
    sigmasq = (e2 - (mu**2)*kappa)/nu

    return [mu, sigmasq, kappa, nu]

  @staticmethod
  def get_stats(S, j, a, b):
    mu_0 = _GaussianSuffStats.__get_mu_0(S, j, a, b)
    sigmasq_0 = _GaussianSuffStats.__get_sigmasq_0(S, j, a, b)
    kappa_0 = _GaussianSuffStats.__get_kappa_or_nu_0(S, j, a, b)
    nu_0 = kappa_0

    return _GaussianSuffStats.to_vec([mu_0, sigmasq_0, kappa_0, nu_0])

  @staticmethod
  def to_vec(l):
    """
    Converts a list of the form [mu_0, sigma_0, kappa_0, nu_0]
    to a single numpy array. Used by the get expected sufficient
    statistics method.
    """
    return np.array(l)

  @staticmethod
  def to_list(v):
    """
    Converts a single numpy array to a list of the form
    [mu_0, sigmasq_0, kappa_0, nu_0]
    """
    mu_0 = v[0]
    sigmasq_0 = v[1]
    kappa_0 = v[2]
    nu_0 = v[3]

    return [mu_0, sigmasq_0, kappa_0, nu_0]

  @staticmethod
  def __get_mu_0(S, j, a, b):
    obs = S.data[0][a : (b + 1)]
    L = b - a + 1  # length of this subsequence
    gammas = np.array([np.exp(g[j]) for g in S.gamma[0][a : (b + 1)]])

    res = 0.
    for t in range(0, L):
      res += gammas[t]*obs[t]

    return res/_GaussianSuffStats.__get_kappa_or_nu_0(S, j, a, b)

  @staticmethod
  def __get_sigmasq_0(S, j, a, b):
    obs = S.data[0][a : (b + 1)]  
    L = b - a + 1  # length of this subsequence
    gammas = np.array([np.exp(g[j]) for g in S.gamma[0][a : (b + 1)]])

    kappa = _GaussianSuffStats.__get_kappa_or_nu_0(S, j, a, b)

    mu_0 = _GaussianSuffStats.__get_mu_0(S, j, a, b)

    res = 0.
    for t in range(0, L):
      res += gammas[t]*((obs[t] - mu_0)**2)

    return res/kappa

  @staticmethod
  def __get_kappa_or_nu_0(S, j, a, b):
    return np.sum(np.array([np.exp(g[j]) for g in S.gamma[0][a : (b + 1)]]))

  @staticmethod
  def maximize_likelihood_helper(S, j, a, b):
    """
    Returns [mu, sigma] which maximize the likelihood of it being
    the jth hidden state's emitter for time interval [a, b]
    """
    kappa = _GaussianSuffStats.__get_kappa_or_nu_0(S, j, a, b)
  
    mu = _GaussianSuffStats.__get_mu_0(S, j, a, b)/kappa

    obs = S.data[0][a : (b + 1)]
    L = b - a + 1  # length of this subsequence
    gammas = np.array([np.exp(g[j]) for g in S.gamma[0][a : (b + 1)]])
  
    sigmasq = 0.
    for t in range(0, L):
      sigmasq += gammas[t]*(np.outer(obs[t] - mu, obs[t] - mu))
  
    sigmasq /= kappa
  
    return [mu, sigmasq**(0.5)]
