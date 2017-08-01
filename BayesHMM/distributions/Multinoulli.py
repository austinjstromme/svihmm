# exteneral packages
import sys
sys.path.append('../utils')
from numpy import random as nprand
from scipy.misc import logsumexp
from scipy.special import digamma as dg
import numpy as np

# internals
from Exponential import Exponential
from Dirichlet import Dirichlet
import LogMatrixUtil as lm

class Multinoulli(Exponential):
  """
  Multinoulli is a class representing multinoulli distributions.
 
  Fields:
    params: a list of L probabilities
    prior: a Dirichlet distribution governing the prior on this Multinoulli
  """

  def __init__(self, params, prior=None):
    """
    Initializes a Multinoulli distribution.

    Args:
      params: a list of probabiltilies, where
        params[l] is p_l. If L is the length of params
        then assumes that this Multionoulli has L symbols.
      prior: a distribution (see Distribution.py) that gives
        the prior for this distribution. Optional.

    Effects:
      self: Initializes with the given parameters. If a prior
        is not specified, defaults to a Dir(alpha) prior where alpha
        is all 1's
    """
    if (np.min(params) < 0.0):
      raise ValueError("please enter valid probabilties; Multinoulli"
              + " initialization failed")

    self.params = np.array(params)
    self.L = self.params.shape[0]
    if (prior is None):
      self.prior = Dirichlet(2*np.ones(self.L))
    else:
      self.prior = prior

  def gen_log_expected(self):
    """
    Returns log expected distribution according to self.prior.
    """
    w = self.prior.get_natural()
    temp = dg(np.sum(w))
    params = np.array([np.exp(dg(w[l]) - temp) for l in range(0, len(w))])
    return Multinoulli(params)

  def gen_sample(self):
    """
    Generates a single sample.
    """
    gen = nprand.multinomial(1, self.params)
    for l in range(0, self.L):
      if gen[l] == 1:
         return l
    # if it makes it to here, something has gone very wrong
    raise ValueError("Multinoulli not initialized properly; can't"
           + " generate samples")

  def get_natural(self):
    """
    Returns the natural parameters of this Multinoulli.
    """
    return np.array([np.log(self.params[k]/self.params[self.L - 1])
        for k in range(0, self.L)])

  def set_natural(self, w):
    """
    Updates the parameters so the natural parameters become w

    Args:
      w: np.array of length L of new natural parameters
    """
    self.params = np.exp(w)/np.sum(np.exp(w))

  def maximize_likelihood(self, S, j):
    """
    Updates the parameters of this distribution to maximize the likelihood
    of it being the jth hidden state's emitter.
    """
    w = self.get_expected_suff(S, j)
    self.set_natural(np.log(w))

  def update_params(self, x, gamma_data, k):
    """
    Updates the parameters via EM.

    Args:
      x: a list of N observation sequences
      gamma_data: a list of the smoothed posterior marginals
        for each observation sequence. Thus gamma_data[i][t][k]
        is the probabiltiy that the ith observation sequence at
        time t was in state k. IN LOG DOMAIN.
      k: the state this distribution corresponds to

    Effects:
      params: updates the parameters via EM
    """
    state_count = self.state_count(gamma_data, k)
    obs_count = self.obs_count(x, gamma_data, k)
    self.params = np.exp(obs_count)/np.exp(state_count)

  def get_expected_suff(self, S, j):
    """
    Returns the vector of the expected sufficient statistics from
    a given states object; assume this dist is the one corresponding to
    the jth hidden state.
    """
    return np.exp(self.obs_count(S.data, S.gamma, j))

  def get_expected_local_suff(self, S, j, a, b):
    """
    Returns the vector of expected sufficient statistics from subchain
    [a,b] according to a given states object; assums this dist is the one
    corresponding to the jth hidden state.
    """
    return np.exp(self.local_obs_count(S.data, S.gamma, j, a, b))

  def state_count(self, gamma_data, j):
    """
    returns the log of expected number of times an observation
    was in state j
    """
    N = len(gamma_data)
    res = []
    for i in range(0,N):
      res += [gamma_data[i][t][j] for t in 
                xrange(0,len(gamma_data[i]))]
    return logsumexp(res)

  def local_obs_count(self, x, gamma_data, j, a, b):
    """
    returns numpy vector whose lth coordinate is the log of the expected
    number of times an observation was in state j and emitted an l;
    a <= t <= b
    """
    res = np.zeros((self.L, b - a + 1)) - np.inf

    for i in range(0, 1):
      for t in range(a, b + 1):
        res[x[i][t]][t - a] = gamma_data[i][t][j]

    return logsumexp(res, axis=1)

  def obs_count(self, x, gamma_data, j):
    """
    Returns a vector whose lth coord is the log of the expected
    number of times an observation was in state j and emitted an l
    """
    N = len(x)
    res = []

    for i in range(0, N):
      res.append(self.local_obs_count([x[i]], [gamma_data[i]], j,
                  0, len(x[i]) - 1))
    res = np.array(res)
    return logsumexp(res, axis=0)

  def mass(self, x):
    """
    Computes the probability of an observation x.

    Args:
      x: a single observation

    Returns:
      p(x)
    """
    return self.params[x]

  def __str__(self):
    return str(self.params)
    
