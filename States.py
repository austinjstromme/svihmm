import numpy as np
import LogMatrixUtil as lm
import math

class States(object):
  """
  A States object represents the states of a given HMM, both the hidden
  and observed. Each States object supports adding data, subsequent to
  which it updates its latent estimates.

  Fields:
    M: the HMM this States object is associated with
    data: a list of N observations sequences; data[n] is a list of length T_n
    gamma: represents log probabilities of posterior marginals
      gamma[n][t][i] is the log of the probability that the nth observation
      sequence at time t was in state i. gamma is a list of lists of
      np.arrays of shape (K)
    xi: represents log probabilities of two-slice marginals
      xi[n][t][i][j] is the log of the probability that the nth observation
      sequence was in latent state i at time t and latent state j at time
      t + 1. Thus xi[n] is a (T_n - 1)xKxK table. xi is a list of lists of
      np.arrays of shape (K,K)
  """

  def __init__(self, M, x):
    """
    Initializes a States object with a given HMM and observation sequence
    (possibly empty) x.

    M: the HMM to which this States object is associated
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
    self.xi = [[np.ones((K,K)) for t in range(0, len(x[n]))]
                  for n in range(0, N)]
    self.e_step()

  def e_step(self):
    """
    Runs forwards-backwards to compute gamma and xi given the current data.
    """
    for i in range(0, len(self.data)):
      self.e_step_row(i)

  def e_step_row(self, i):
    """
    Updates gamma and xi for the ith data point.
    """
    x = self.data[i]

    psi = [[np.log(self.M.D[k].mass(x[t]))
            for k in range(0, len(self.M.D))]
            for t in range(0, len(x))]

    [alpha, Z] = self.forward_alg(x, psi)
    beta = self.backward_alg(x, psi)

    #build gamma
    gamma = []
    for t in range(0, len(x)):
      res = lm.norm(lm.mult(alpha[t], beta[t]))[0]
      gamma.append(res)
    #update gamma
    self.gamma[i] = gamma

    #build xi:
    xi = []
    for t in range(0, len(x) - 1):
      xi.append(lm.norm(lm.mult(self.M.A, lm.outer(alpha[t],
        lm.mult(psi[t + 1], beta[t + 1]))))[0])
    #update xi:
    self.xi[i] = xi
        
  def backward_alg(self, x, psi):
    """
    Runs the backwards algorithm and returns beta.
    """
    beta = [np.zeros(self.M.K)]
    for t in range(len(x) - 2, -1, -1):
      res = lm.dot(self.M.A, lm.mult(psi[t + 1], beta[0]))
      beta.insert(0, res)
    return beta

  def forward_alg(self, x, psi):
    """
    Runs the forwards algorithm and returns alpha and the Z values.
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

  def LL(self):
    """
    Returns the log likelihood of the current data 
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

