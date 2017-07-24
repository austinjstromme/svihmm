import numpy as np

class _GaussianSuffStats():
  """
  A static implementation helper class. Converts between
  a list of sufficient statistics and a single numpy array.
  """

  @staticmethod
  def get_stats(S, j, a, b, dim):
    mu_0 = _GaussianSuffStats.__get_mu_0(S, j, a, b, dim)
    sigma_0 = _GaussianSuffStats.__get_sigma_0(S, j, a, b, dim)
    kappa_0 = _GaussianSuffStats.__get_kappa_or_nu_0(S, j, a, b, dim)
    nu_0 = kappa_0

    return _GaussianSuffStats.to_vec([mu_0, sigma_0, kappa_0, nu_0], dim)

  @staticmethod
  def to_vec(l, dim):
    """
    Converts a list of the form [mu_0, sigma_0, kappa_0, nu_0]
    to a single numpy array. Used by the get expected sufficient
    statistics method.
    """
    mu_0, sigma_0, kappa_0, nu_0 = l
   
    L = [x for x in mu_0]


    for i in range(0, dim):
      for j in range(0, dim):
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

    sigma_0 = np.zeros(dim, dim)
    for i in range(0, dim):
      for j in range(0, dim):
        sigma[i][j] = v[idx]
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
    return np.sum(np.exp(S.gamma[0][a : (b + 1)]))

