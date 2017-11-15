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
    Converts from normal parameters to natural parameters.

    Args:
      l: [mu_0, sigmasq_0, kappa_0, nu_0].

    Returns:
      r: list of corresponding natural parameters.
    """
    mu, sigmasq, kappa, nu = l
    return [kappa*mu, (nu*sigmasq + (mu**2)*kappa), kappa, nu]

  @staticmethod
  def NICS_natural_to_normal(l):
    """
    Converts from natural parameters to normal parameters.

    Args:
      l: list of natural parameters.

    Returns:
      r: [mu_0, sigmasq_0, kappa_0, nu_0].
    """
    e1, e2, e3, e4 = l
    kappa = e3
    nu = e4
    mu = e1/kappa
    sigmasq = (e2 - (mu**2)*kappa)/nu

    return [mu, sigmasq, kappa, nu]

  @staticmethod
  def get_stats(S, j, a, b):
    mu_0 = 0.
    sigmasq_0 = 0.
    kappa_0 = 0.
    nu_0 = 0.

    for i in range(0, len(S.data)):
      mu_0 += _GaussianSuffStats.__get_mu_0(S, i, j, a, b)
      sigmasq_0 += _GaussianSuffStats.__get_sigmasq_0(S, i, j, a, b)
      kappa_0 += _GaussianSuffStats.__get_kappa_or_nu_0(S, i, j, a, b)
      nu_0 += kappa_0

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
  def __get_mu_0(S, i, j, a, b):
    obs = S.data[i][a : (b + 1)]
    L = b - a + 1  # length of this subsequence
    gammas = np.array([np.exp(g[j]) for g in S.gamma[i][a : (b + 1)]])

    res = 0.
    for t in range(0, L):
      res += gammas[t]*obs[t]

    return res

  @staticmethod
  def __get_sigmasq_0(S, i, j, a, b):
    obs = S.data[i][a : (b + 1)]  
    L = b - a + 1  # length of this subsequence
    gammas = np.array([np.exp(g[j]) for g in S.gamma[i][a : (b + 1)]])

    res = 0.
    for t in range(0, L):
      res += gammas[t]*(obs[t]**2)

    return res

  @staticmethod
  def __get_kappa_or_nu_0(S, i, j, a, b):
    return np.sum(np.array([np.exp(g[j]) for g in S.gamma[i][a : (b + 1)]]))

  @staticmethod
  def maximize_likelihood_helper(S, j):
    """
    Returns parameters which maximize the likelihood of it being the jth hidden
    state's emitter.

    Args:
      S: states object.
      j: hidden state we correspond to

    Returns:
      r: [mu, sigma]
    """
    kappa = 0.
    mu = 0.
    sigmasq = 0.

    for i in range(0, len(S.data)):
      a = 0
      b = len(S.data[i]) - 1

      kappa += _GaussianSuffStats.__get_kappa_or_nu_0(S, i, j, a, b)
      mu += _GaussianSuffStats.__get_mu_0(S, i, j, a, b)

    mu /= kappa

    for i in range(0, len(S.data)):
      obs = S.data[i][a : (b + 1)]
      gammas = np.array([np.exp(g[j]) for g in S.gamma[i][a : (b + 1)]])
      L = len(S.data[i]) - 1

      for t in range(0, L):
        sigmasq += gammas[t]*((obs[t] - mu)**2.)
  
    sigmasq /= kappa
  
    return [mu, sigmasq**(0.5)]
