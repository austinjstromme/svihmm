# external packages
from scipy.special import digamma as dg
from random import randrange
import numpy as np

# internals
from .HMM import HMM
from .States import States
from ..distributions.Dirichlet import Dirichlet
from ..distributions.Multinoulli import Multinoulli
from ..utils import LogMatrixUtil as lm

class VBHMM(object):
  """
    A VBHMM is a HMM that supports variational bayesian inference.

    Attributes:
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
      S - a States object holding our local beliefs about the observations.
  """

  def __init__(self, K, u_A, u_pi, u_D, D, obs):
    """
    Initializes a VBHMM with K hidden states.

    Params:
      K: the number of hidden states
      u_A: the Dirichlet distribution governing the prior on the rows of A
      u_pi: a Dirichlet distribution which is the prior on the start
      u_D: list of priors governing hyperparameters on the emissions
      D: list of K exp family dists (see Exponential.py) governing
        emissions
      obs: list of observations associated this will learn on
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
      # initialize params of D[k] to the passed in hyperparams
      self.D[k].prior.set_natural(u_D[k].get_natural())

    self.S = States(self.gen_M(), obs)

  def gen_a_b(self, buf, L):
    """
    Randomly generates an interval of length L with buffer size buf.

    Args:
      buf: buffer size.
      L: interval length.

    Returns:
      [a,b]: uniformly random interval of length L in [0, T], T being the
      length of the first observation sequence of S.
    """
    T = len(self.S.data[0])
    a = randrange(buf, T - L - buf)
    b = a + L
    return [a, b]

  def batch_SVI_step(self, buf, L, rho, M):
    """
    Does a single SVI step using buf as the buffer size and a minibatch
    of size M.

    Args:
      buf: integer determining the size of the buffer for our local
        update step
      L: length of the subchain
      rho: stepsize

    Effects:
      self: updates variational parameters along with beliefs in self.S.
    """
    trans = np.zeros((self.K, self.K))
    emissions = [np.zeros(len(self.D[k].prior.get_natural()))
      for k in range(0, self.K)]
    #emission = np.zeros((self.K, self.D[0].L))

    # generate the subchains:
    subchains = [self.gen_a_b(buf, L) for i in range(0, M)]

    # do local e-steps:
    self.S.M = self.gen_M(local=True)
    for i in range(0, M):
      self.S.e_step_row_sub_chain(0, subchains[i][0], subchains[i][1], buf)

    # now get sufficient statistics
    for i in range(0, M):
      a, b = subchains[i]
      trans += np.exp(self.S.get_local_trans(a, b))
      for j in range(0, self.K):
        emissions[j] += self.D[j].get_expected_local_suff(self.S, j, a, b)

    # now update natural params for A
    c_A = float(len(self.S.data[0]) - L + 1)/(L - 1)
    for j in range(0, self.K):
      new_row = (self.u_A.get_natural() + (c_A/float(M))*trans[j])
      temp = (1. - rho)*self.w_A[j].get_natural() + rho*new_row
      self.w_A[j].set_natural(temp)

    # update the natural params of the priors on our emissions
    c_phi = float(len(self.S.data[0]) - L + 1)/L
    for j in range(0, self.K):
      dist = self.D[j]
      new_row = (c_phi/float(M))*emissions[j]
      new_row += self.u_D[j].get_natural()
      temp = (1. - rho)*dist.prior.get_natural() + rho*new_row
      dist.prior.set_natural(temp)

  def SVI_step(self, buf, L, rho):
    """
    Does a single SVI step using buf as the buffer size.

    Args:
      buf: integer determining the size of the buffer for our local
        update step
      L: length of the subchain
      rho: stepsize

    Effects:
      self: updates this VBHMM's variational parameters and beliefs in self.S
    """
    self.batch_SVI_step(buf, L, rho, 1)

  def VB_step(self):
    """
    Does a single VB step.

    Effects:
      self: updates this VBHMM's variational parameters and beliefs in self.S
    """
    #update A
    trans = np.exp(self.S.get_trans())
    for j in range(0, self.K):
      new_row = self.u_A.get_natural() + trans[j]
      self.w_A[j].set_natural(new_row)

    # update emissions
    for j in range(0, self.K):
      dist = self.D[j]
      dist.prior.set_natural(dist.get_expected_suff(self.S, j)
        + self.u_D[j].get_natural())

    # update pi
    new_pi = self.u_pi.get_natural() + np.exp(self.S.get_start())
    self.w_pi.set_natural(new_pi)

    #do e step
    self.S.M = self.gen_M()
    self.S.e_step()

  def gen_M(self, local=False):
    """
    Generates an HMM M to be used in message passing.

    Args:
      local: if local is true, sets the pi vector to be an eigenvector of A.

    Returns:
      M: HMM generated via variational parameters. See section S4 of the SVIHMM
      paper.
    """
    # compute A
    A = []
    for j in range(0, self.K):
      w = self.w_A[j].get_natural()
      temp = dg(np.sum(w))
      row = np.array([np.exp(dg(w[k]) - temp) for k in range(0, self.K)])
      A.append(row)
    A = np.array(A)

    # compute pi depending on value for local
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

    # make distributions
    Dists = [self.D[j].gen_log_expected() for j in range(0, self.K)]

    return HMM(self.K, A, pi, Dists)

  def elbo(self):
    """
    Returns the Evidence Lower Bound (elbo) of this VBHMM with states
    S. Note: this only valid immediately after an e-step has been done.

    Returns:
      elbo
    """
    res = -self.u_pi.KL_div(self.w_pi)
    for j in range(0, self.K):
      res -= (self.u_A.KL_div(self.w_A[j])
        + self.u_D[j].KL_div(self.D[j].prior))

    res += self.S.LL()
    return res

  def __str__(self):
    """
    Returns string representation of this.
    """
    res = "VBHMM with K = " + str(self.K) + "\n"
    res += "  w_pi = " + str(self.w_pi.get_natural()) + "\n"
    res += "  w_A = " 
    for k in range(0, self.K):
      res += str(self.w_A[k].get_natural() + 1.) + "\n"
    res += "  D = "
    for k in range(0, self.K):
      res += str(self.D[k].prior.get_params()) + "\n"
    return res
