# external packages
import numpy as np

class _NDGaussianSuffStats():
  """
  A static implementation helper class. Converts between
  a list of sufficient statistics and a single numpy array.
  """

  @staticmethod
  def NIW_normal_to_natural(l):
    """
    Converts from normal parameters to natural parameters.

    Args:
      l: [mu_0, sigma_0, kappa_0, nu_0].

    Returns:
      r: list of corresponding natural parameters.
    """
    # taking this from Dillon Laird's code
    mu, sigma, kappa, nu = l
    eta3 = sigma + np.outer(mu, mu)*kappa
    return [kappa*mu, eta3, kappa, nu + 2 + mu.shape[0]]

  @staticmethod
  def NIW_natural_to_normal(l):
    """
    Converts from natural parameters to normal parameters.

    Args:
      l: list of natural parameters.

    Returns:
      r: [mu_0, sigma_0, kappa_0, nu_0].
    """
    e1, e2, e3, e4 = l
    kappa = e3
    mu = e1/kappa
    sigma = e2 - np.outer(mu, mu)*kappa
    nu = e4 - 2 - mu.shape[0]

    return [mu, sigma, kappa, nu]

  @staticmethod
  def get_stats(S, j, a, b, dim):
    mu_0 = _NDGaussianSuffStats.__get_mu_0(S, j, a, b, dim)
    sigma_0 = _NDGaussianSuffStats.__get_sigma_0(S, j, a, b, dim)
    kappa_0 = _NDGaussianSuffStats.__get_kappa_or_nu_0(S, j, a, b, dim)
    nu_0 = kappa_0

    return _NDGaussianSuffStats.to_vec([mu_0, sigma_0, kappa_0, nu_0], dim)

  @staticmethod
  def to_vec(l, dim):
    """
    Converts a list of the form [mu_0, sigma_0, kappa_0, nu_0]
    to a single numpy array. Used by the get expected sufficient
    statistics method.
    """
    mu_0, sigma_0, kappa_0, nu_0 = l

    L = [x for x in mu_0]

    for i in xrange(0, dim):
      for j in xrange(0, dim):
        L.append(sigma_0[i][j])

    L.append(kappa_0)
    L.append(nu_0)

    return np.array(L)

  @staticmethod
  def to_list(v, dim):
    """
    Converts a single numpy array to a list of the form
    [mu_0, sigma_0, kappa_0, nu_0]
    """
    idx = 0
    mu_0 = v[0 : dim]

    idx += dim

    sigma_0 = np.zeros((dim, dim))
    for i in range(0, dim):
      for j in range(0, dim):
        sigma_0[i][j] = v[idx]
        idx += 1

    kappa_0 = v[idx]
    idx += 1
    nu_0 = v[idx]

    return [mu_0, sigma_0, kappa_0, nu_0]

  @staticmethod
  def __get_mu_0(S, j, a, b, dim):
    obs = S.data[0][a : (b + 1)]
    L = b - a + 1  # length of this subsequence
    gammas = np.array([np.exp(g[j]) for g in S.gamma[0][a : (b + 1)]])

    res = np.zeros(dim)
    for t in range(0, L):
      res += gammas[t]*obs[t]

    return res

  @staticmethod
  def __get_sigma_0(S, j, a, b, dim):
    obs = S.data[0][a : (b + 1)]  
    L = b - a + 1  # length of this subsequence
    gammas = np.array([np.exp(g[j]) for g in S.gamma[0][a : (b + 1)]])

    res = np.zeros((dim, dim))
    for t in range(0, L):
      res += gammas[t]*(np.outer(obs[t], obs[t]))

    return res

  @staticmethod
  def __get_kappa_or_nu_0(S, j, a, b, dim):
    return np.sum(np.array([np.exp(g[j]) for g in S.gamma[0][a : (b + 1)]]))

  @staticmethod
  def maximize_likelihood_helper(S, j, a, b, dim):
    """
    Returns parameters which maximize the likelihood of it being the jth
    hidden state's emitter for time interval [a,b].

    Args:
      S: states object.
      j: hidden state we correspond to
      a: beginning of subchain
      b: end of subchain

    Returns:
      r: [mu, sigma]
    """
    kappa = _NDGaussianSuffStats.__get_kappa_or_nu_0(S, j, a, b, dim)
  
    mu = _NDGaussianSuffStats.__get_mu_0(S, j, a, b, dim)/kappa

    obs = S.data[0][a : (b + 1)]
    L = b - a + 1  # length of this subsequence
    gammas = np.array([np.exp(g[j]) for g in S.gamma[0][a : (b + 1)]])
  
    sigma = np.zeros((dim, dim))
    for t in range(0, L):
      sigma += gammas[t]*(np.outer(obs[t] - mu, obs[t] - mu))
  
    sigma /= kappa
  
    return [mu, sigma]
