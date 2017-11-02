# external packages
import numpy as np
import scipy as sc
import math
import random

# internals
from ..utils import LogMatrixUtil as lm

class States(object):
  """
  A States object represents the states of a given HMM, both the hidden
  and observed. Each States object supports adding data, subsequent to
  which it updates its latent estimates.

  Attributes:
    M: the HMM this States object is associated with
    data: a list of N observations sequences; data[n] is a list of length T_n
    gamma: represents log probabilities of posterior marginals;
      gamma[n][t][i] is the log of the probability that the nth observation
      sequence at time t was in state i. gamma is a list of lists of
      np.arrays of shape (K)
    xi: represents log probabilities of two-slice marginals;
      xi[n][t][i][j] is the log of the probability that the nth observation
      sequence was in latent state i at time t and latent state j at time
      t + 1. Thus xi[n] is a (T_n - 1)xKxK table. xi is a list of lists of
      np.arrays of shape (K,K)
  """

  def __init__(self, M, x):
    """
    Initializes a States object with a given HMM and observation sequence
    (possibly empty) x.

    Args:
      M (HMM): the HMM we wnat this States object to be associated with
      x: a list of N observation seqeunces; x[i] is a list of length T_i
      observations
    """
    self.M = M
    self.data = x
    N = len(self.data)
    K = M.K
    #initialize
    self.gamma = [[np.ones(K) for t in range(0, len(x[n]))]
                  for n in range(0, N)]
    self.xi = [[np.ones((K,K)) for t in range(0, len(x[n]) - 1)]
                  for n in range(0, N)]
    self.e_step()

  def add_data(self, x):
    """
    Adds observation sequences x to the data. DOES NOT do any inference
    on the new sequences before returning.

    Args:
      x: list of N observation sequences; x[i] is a list of length T_i
      observations.
    """
    for seq in x:
      self.data.append(seq)

  def e_step(self):
    """
    Runs forwards-backwards to compute gamma and xi with the given M.
    """
    for i in range(0, len(self.data)):
      self.e_step_row(i)

  def e_step_sub_chain(self, a, b, buf):
    """
    Runs local forwards-backwards.

    Args:
      a: start of subchain.
      b: end of subchain.
      buf: buffer size for subchain.
    """
    for i in range(0, len(self.data)):
      if b + buf >= len(self.data[i]) or a < 0:
        raise ValueError("faulty parameters for local update interval")
        # no point in doing anything fancy; just update them all
        self.e_step_row(i)
      # run FB on this subchain
      self.e_step_row_sub_chain(i, a, b, buf)

  def e_step_row_sub_chain(self, i, a, b, buf):
    """
    Updates gamma and xi for t in [a,b] for the ith data point by
    performing a forward-backward pass on [a-buf, b + buf].

    Args:
      i: observation sequence index.
      a: start of subchain.
      b: end of subchain.
      buf: buffer size for subchain.
    """
    x = self.data[i][(a - buf) : (b + buf + 1)]

    psi = [[np.log(self.M.D[k].mass(x[t]))
            for k in range(0, len(self.M.D))]
            for t in range(0, len(x))]

    [alpha, Z] = self.forward_alg(x, psi)
    beta = self.backward_alg(x, psi)

    #gamma updates
    for t in range(buf, len(x) - buf):
      res = lm.norm(lm.mult(alpha[t], beta[t]))[0]
      self.gamma[i][t + a - buf] = res

    #xi updates
    xi = []
    for t in range(buf, len(x) - buf - 1):
      res = lm.norm(lm.mult(self.M.A, lm.outer(alpha[t],
        lm.mult(psi[t + 1], beta[t + 1]))))[0]
      self.xi[i][t + a - buf] = res

  def e_step_row(self, i):
    """
    Updates gamma and xi for the ith data point.
    """
    self.e_step_row_sub_chain(i, 0, len(self.data[i]) - 1, 0)

  def backward_alg(self, x, psi):
    """
    Runs the backwards algorithm and returns beta.

    Args:
      x: observation sequence.
      psi: local evidence vector; psi[t][k] is the log probability that the
      kth hidden state emitted x[t] at time t.

    Returns:
      beta: see forward backward discussion in Murphy, chapter 12.
    """
    beta = [np.zeros(self.M.K)]
    for t in range(len(x) - 2, -1, -1):
      res = lm.dot(self.M.A, lm.mult(psi[t + 1], beta[0]))
      beta.insert(0, res)
    return beta

  def forward_alg(self, x, psi):
    """
    Runs the backwards algorithm and returns beta.

    Args:
      x: observation sequence.
      psi: local evidence vector; psi[t][k] is the log probability that the
      kth hidden state emitted x[t] at time t.

    Returns:
      [alpha, Z]: see forward backward discussion in Murphy, chapter 12.
    """
    res = lm.norm(lm.mult(psi[0], self.M.pi))
    alpha = [res[0]]
    Z = [res[1]]
    for t in range(1, len(x)):
      res = lm.norm(lm.mult(psi[t], lm.dot(np.transpose(self.M.A),
              alpha[t - 1])))
      alpha.append(res[0])
      Z.append(res[1])
    return [alpha, Z]

  def get_local_trans(self, a, b):
    """
    Returns matrix of local transition probabilities.

    Args:
      a: start of subchain.
      b: end of subchain.

    Returns:
      X: the KxK matrix whose entries are
      log(\sum_{t = a}^b exp(xi[0][t][i][j]))
    """
    K = self.M.K
    N = len(self.xi)

    res = np.zeros((K,K))
    for j in range(0, K):
      for k in range(0, K):
        res[j][k] = sc.misc.logsumexp([self.xi[0][t][j][k]
          for t in range(a, b)])

    return res

  def get_start(self):
    """
    Returns matrix of start probabilities.

    Returns:
      y: the K vector whose entries are
      log( \sum_{n = 1}^N exp(gamma[n][0][k]))
    """
    N = len(self.gamma)
    K = self.M.K
    res = np.zeros(K)
    for k in range(0, K):
      sum_k = [self.gamma[n][0][k] for n in range(0, N)]
      res[k] = sc.misc.logsumexp(sum_k)
    return res

  def get_trans(self):
    """
    Returns matrix of transition probabilities.

    Returns:
      X: the KxK matrix whose entries are
      log(\sum_{t = a}^b exp(xi[0][t][i][j]))
    """
    K = self.M.K
    N = len(self.xi)

    res = np.zeros((K,K))
    for j in range(0, K):
      for k in range(0, K):
        res[j][k] = sc.misc.logsumexp([[self.xi[n][t][j][k]
          for t in range(0, len(self.xi[n]))] for n in range(0, N)])

    return res

  def LL(self):
    """
    Returns log likelihood of current data.
    """
    res = 0.
    for n in range(0, len(self.data)):
      x = self.data[n]
      psi = [[np.log(self.M.D[k].mass(x[t]))
              for k in xrange(0, len(self.M.D))]
              for t in xrange(0, len(x))]
      Z = self.forward_alg(x, psi)[1]
      for t in range(0, len(Z)):
        res += np.log(Z[t])
    return res
