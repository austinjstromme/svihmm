from Exponential import Exponential
from numpy import numpy.random as nprand

class Multinoulli(Exponential):

  def __init__(self, params):
    """
    Initializes a Multinoulli distribution.

    Args:
      params: a list of probabiltilies, where
        params[l] is p_l. If L is the length of params
        then assumes that this Multionoulli has L symbols.

    Effects:
      Initializes a Multinoulli with the given parameters.
    """
    if(sum(params) != 1.0 or min(params) < 0.0):
      print("please enter valid probabilties; intialization failed")
      return None

    self.params = params
    self.L = len(params)

  def gen_sample(self):
    """
    Generates a single sample.
    """
    return nprand.multinomial(1, self.params)

  def update_params(self, x, gamma_data, k):
    """
    Updates the parameters via EM.

    Args:
      x: a list of N observation sequences
      gamma_data: a list of the smoothed posterior marginals
        for each observation sequence. Thus gamma_data[i][t][k]
        is the probabiltiy that the ith observation sequence at
        time t was in state k.
      k: the state this distribution corresponds to

    Effects:
      params: updates the parameters via EM
    """
    state_count = self.state_count(gamma_data, k)
    for l in range(0, self.L):
      self.params[l] = self.obs_count(x, gamma_data, k, l)/state_count
    
  def state_count(self, gamma_data, j):
    """
    returns the expected number of times an observation
    was in state j
    """
    N = len(gamma_data)
    res = 0.0
    for i in range(0,N):
      res += sum([gamma_data[i][t][j] for t in 
                xrange(0,gamma_data[i].shape[0])])
    return res

  def obs_count(self, x, gamma_data, j, l):
    """
    returns the expected number of times an observation
    was in state j and emitted an l
    """
    N = len(x)
    res = 0.0
    for i in range(0,N):
      for t in range(0,len(x[i])):
        if x[i][t] == l:
          res += gamma_data[i][t][j]
    return res


  def mass(self, x):
    """
    Computes the probability of an observation x.

    Args:
      x: a single observation

    Returns:
      p(x)
    """
  return params[x]
    
