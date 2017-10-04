# external packages
from scipy.special import digamma as dg
from random import randrange
import numpy as np

# internals
import context  # for utils and distributions
from HMM import HMM
from distributions.Dirichlet import Dirichlet
from distributions.Multinoulli import Multinoulli
import utils.LogMatrixUtil as lm

class VBHMM(object):
  """
    A VBHMM is a HMM that supports variational bayesian inference.

    Fields:
      K - the number of hidden states
      u_pi - Dirichlet distribution governing the hyperparameter
        on the start vector
      u_A - Dirichlet distribution governing the hyperparameter
        on the rows of A
      u_D - list of distributions governing the hyperparmeters
        on the emissions
      w_pi - Dirichlet distribution governing the current variational
        approximation q
      w_A - a list of K Dirichlet distributions governing the current
        variational approximation q
      D - list of K exponential family distributions (see Exponential.py)
        governing emissions from the hidden states. Each with a prior.
  """

  def __init__(self, K, u_A, u_pi, u_D, D):
    """
    Initializes a VBHMM with K hidden states.

    Params:
      K: the number of hidden states
      u_A: the Dirichlet distribution governing the prior on the rows of A
      u_pi: a Dirichlet distribution which is the prior on the start
      u_D: list of priors governing hyperparameters on the emissions
      D: list of K exp family dists (see Exponential.py) governing
        emissions
    """
    self.K = K
    #save hyperparams u_A, u_pi, u_D
    self.u_A = u_A
    self.u_pi = u_pi
    self.u_D = u_D
    #create the rest of the fields
    self.w_A = [Dirichlet(u_A.get_natural() + 1.) for k in range(0, K)]
    self.w_pi = Dirichlet(u_pi.get_natural() + 1.)
    self.D = D
    for k in range(0, K):
      self.D[k].prior = u_D[k]

  def gen_a_b(self, S, buf, L):
    """
    Randomly generates an interval of length L with buffer size buf
    """
    T = len(S.data[0])
    a = randrange(buf, T - L - buf)
    b = a + L
    return [a, b]

  def batch_SVI_step(self, S, buf, L, rho, M):
    """
    Does a single SVI step using buf as the buffer size and a minibatch
    of size M.

    Args:
      S: a States object. We will only do inference on the first data
        sequence
      buf: integer determining the size of the buffer for our local
        update step
      L: length of the subchain
      rho: stepsize

    Effects:
      self: updates this VBHMM's fields.
      S: updates the gamma and xi tables
    """
    trans = np.zeros((self.K, self.K))
    emissions = [np.zeros(len(self.D[k].prior.get_natural()))
      for k in range(0, self.K)]
    #emission = np.zeros((self.K, self.D[0].L))

    # generate the subchains:
    subchains = [self.gen_a_b(S, buf, L) for i in range(0, M)]

    # do local e-steps:
    S.M = self.gen_M(local=True)
    for i in range(0, M):
      S.e_step_row_sub_chain(0, subchains[i][0], subchains[i][1], buf)

    # now get sufficient statistics
    for i in range(0, M):
      a, b = subchains[i]
      trans += np.exp(S.get_local_trans(a, b))
      for j in range(0, self.K):
        emissions[j] += self.D[j].get_expected_local_suff(S, j, a, b)

    # now update natural params for A
    c_A = float(len(S.data[0]) - L + 1)/(L - 1)
    for j in range(0, self.K):
      new_row = (self.u_A.get_natural() + (c_A/float(M))*trans[j])
      temp = (1. - rho)*self.w_A[j].get_natural() + rho*new_row
      self.w_A[j].set_natural(temp)

    # update the natural params of the priors on our emissions
    c_phi = float(len(S.data[0]) - L + 1)/L
    for j in range(0, self.K):
      dist = self.D[j]
      new_row = (c_phi/float(M))*emissions[j]
      new_row += self.u_D[j].get_natural()
      temp = (1. - rho)*dist.prior.get_natural() + rho*new_row
      dist.prior.set_natural(temp)

  def SVI_step(self, S, buf, L, rho):
    """
    Does a single SVI step using buf as the buffer size.

    Args:
      S: a States object. We will only do inference on the first data
        sequence
      buf: integer determining the size of the buffer for our local
        update step
      L: length of the subchain
      rho: stepsize

    Effects:
      self: updates this VBHMM's fields
      S: updates the gamma and xi tables
    """
    self.batch_SVI_step(S, buf, L, rho, 1)

  def VB_step(self, S):
    """
    Does a single VB step.

    Args:
      S: a States object

    Effects:
      self: updates this VBHMM's fields.
    """
    #update A
    trans = np.exp(S.get_trans())
    for j in range(0, self.K):
      new_row = self.u_A.get_natural() + trans[j]
      self.w_A[j].set_natural(new_row)

    # update emissions
    for j in range(0, self.K):
      dist = self.D[j]
      dist.prior.set_natural(dist.get_expected_suff(S, j)
                             + self.u_D[j].get_natural())

    # update pi
    new_pi = self.u_pi.get_natural() + np.exp(S.get_start())
    self.w_pi.set_natural(new_pi)

    #do e step
    S.M = self.gen_M()
    S.e_step()

  def gen_M(self, local=False):
    """
    Generates an HMM M to be used in message passing.
    """
    #compute A
    A = []
    for j in range(0, self.K):
      w = self.w_A[j].get_natural()
      temp = dg(np.sum(w))
      row = np.array([np.exp(dg(w[k]) - temp) for k in range(0, self.K)])
      A.append(row)
    A = np.array(A)

    #compute pi depending on value for local
    pi = []
    if local:
      eigval, eigvec = np.linalg.eig(A)
      maxind = max((l, i) for i, l in enumerate(eigval))[1]
      pi = eigvec[:,maxind]
      if np.min(pi) < 0:
        pi = -pi
      pi = (1./np.sum(pi))*pi
    else:
      w = self.w_pi.get_natural()
      temp = dg(np.sum(w))
      pi = np.array([np.exp(dg(w[k]) - temp) for k in range(0, self.K)])

    #make distributions
    Dists = [self.D[j].gen_log_expected() for j in range(0, self.K)]

    return HMM(self.K, A, pi, Dists)
 
      
  def elbo(self, S):
    """
    Returns the Evidence Lower Bound (elbo) of this VBHMM with states
    S. Note: this only valid immediately after an e-step has been done.

    Args:
      S: a States object which points to a current auxiliary HMM for this
        VBHMM

    Returns:
      elbo
    """
    res = -self.u_pi.KL_div(self.w_pi)
    for j in range(0, self.K):
      res -= self.u_A.KL_div(self.w_A[j]) + self.u_D[j].KL_div(self.D[j].prior)

    res += S.LL()
    return res

  def __str__(self):
    res = "VBHMM with K = " + str(self.K) + "\n"
    res += "  w_pi = " + str(self.w_pi.get_natural()) + "\n"
    res += "  w_A = " 
    for k in range(0, self.K):
      res += str(self.w_A[k].get_natural() + 1.) + "\n"
    res += "  D = "
    for k in range(0, self.K):
      res += str(self.D[k].prior.get_natural()) + "\n"
    return res
