# external packages
import numpy as np
import math
import scipy as sc

# internals
import context  # for utils and distributions
from distributions.Multinoulli import Multinoulli
import utils.LogMatrixUtil as lm

class HMM(object):
  """
  A HMM is an abstraction of a Hidden Markov Model.

  Attributes:
    K: number of hidden states.
    A: transition matrix IN LOG DOMAIN.
    pi: start vector IN LOG DOMAIN.
    D: list of K distributions.
  """

  def __init__(self, K, A, pi, D):
    """
    Initializes a Hidden Markov Model (HMM) with K hidden states.

    K: a positive integer givin the number of hidden states.
    A: a KxK np.array of transition probabilities between states. NOT
      in log domain.
    pi: a 1xK np.array of start probabilities. NOT in log domain.
    D: a list of K Exponential Distributions (see Exponential.py), one for
      each state
    """
    #TODO: add checks that the arguments are of the right form
    self.K = K
    self.A = np.log(A)
    self.pi = np.log(pi)
    self.D = D

  def get_params(self):
    """
    Returns:
      params: [K, A, pi, D] NOT IN LOG DOMAIN.
    """
    return [self.K, np.exp(self.A), np.exp(self.pi), self.D]

  def compare(self, M, local=False):
    """
    Compares this HMM to another; symmetric operation. Heuristic for
    verifying structure learning.

    Args:
      M: other HMM to compare to.
      local: if true, don't compare pi vectors.

    Returns:
      x: nonnegative real number representing similarity of the two.
    """
    res = 0.
    if not local:
      res = np.linalg.norm(np.exp(self.pi) - np.exp(M.pi))
    res += np.linalg.norm(np.exp(self.A) - np.exp(M.A))
    for k in range(0, self.K):
      res += np.linalg.norm(self.D[k].params - M.D[k].params)
    return res

  def M_step(self, S):
    """
    Does a single M step.

    Args:
      S: a States object which points to this HMM.

    Effects:
      self: updates this HMM's parameter values
    """
    self.update_start(S)
    self.update_trans(S)
    for k in range(0, self.K):
      self.D[k].update_params(S.data, S.gamma, k)

  def EM_step(self, S):
    """
    Does a single step of EM.

    Args:
      S: states object to do EM with; must point to this HMM

    Effects:
      self: updates parameters according to M step
      S: gamma and xi are update according to E step
    """
    # e-step
    S.e_step()
    
    # update A and pi
    self.update_trans(S)
    self.update_start(S)
    # update emissions
    for k in range(0, self.K):
      self.D[k].maximize_likelihood(S, k)
    
  def update_trans(self, S):
    """
    Updates the transition matrix A.

    Args:
      S: states object to use for updating.
    """
    A_expect = S.get_trans()

    #normalize and put in
    for j in range(0, self.K):
      self.A[j] = lm.norm(A_expect[j])[0]

  def update_start(self, S):
    """
    Updates the start probability vector pi.

    Args:
      S: states object to use for updating.
    """
    self.pi = S.get_start() - np.log(len(S.gamma))

  def EM(self, S, N):
    """
    Args:
      S: states object.
      N: number of steps of EM to do.
    """
    for i in range(0, N):
      self.EM_step(S)

  def gen_obs(self, n):
    """
    Args:
      n: number of observations to generate.

    Returns:
      l: list of n observations.
    """
    m = Multinoulli(np.exp(self.pi))
    curr = m.gen_sample()
    obs = [self.D[curr].gen_sample()]
    for i in range(1, n):
      m = Multinoulli(np.exp(self.A[curr]))
      curr = m.gen_sample()
      obs.append(self.D[curr].gen_sample())
    return obs

  def __str__(self):
    """
    Returns:
      s: string representing this.
    """
    res = "HMM with K = " + str(self.K) + "\n"
    res += "  A = " + str(np.exp(self.A)) + "\n"
    res += "  pi = " + str(np.exp(self.pi)) + "\n"
    res += "  D = "
    for k in range(0, self.K):
      res += str(self.D[k]) + "\n"
    return res
