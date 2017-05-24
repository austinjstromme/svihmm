from HMM import HMM
import numpy as np
from Dirichlet import Dirichlet
from Multinoulli import Multinoulli
import LogMatrixUtil as lm
from scipy.special import digamma as dg
from random import randrange

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
      u_D: list of Dirichlets governing hyperparameters on the emissions
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
    return [a,b]

  def SVI_step(self, S, buf, L, rho):
    """
    Does a single SVI step using buf as the buffer size.

    Args:
      S: a States object. We will only do inference on the first data
        sequence
      buf: integer determining the size of the buffer for our local
        update step

    Effects:
      self: updates this VBHMM's fields.
      S: updates the gamma and xi tables
    """
    #TODO: implement this so that it isn't dependent on Multinoulli
      #emissions (CURRENTLY DEPENDEONT ON MULTINOULLIS)
    #generate the subchain interval
    a, b = self.gen_a_b(S, buf, L)
    #start with E_step because we need that to do the M_step:
    #do FB on subchain interval:
    S.M = self.gen_M(local=True)
    S.e_step_row_sub_chain(0, a, b, buf)
  
    #now do the M_step
    #first update A:
    #sum over all sequences x; use update natural param methods for A
    trans = np.exp(S.get_local_trans(a, b))
    c_A = float(len(S.data[0]) - L + 1)/(L - 1)
    for j in range(0, self.K):
      new_row = (self.u_A.get_natural() + c_A*trans[j]
                  - self.w_A[j].get_natural())
      temp = (1. - rho)*self.w_A[j].get_natural() + rho*new_row
      self.w_A[j].set_natural(temp)

    # update the natural params of the priors on our emissions
    c_phi = float(len(S.data[0]) - L + 1)/L
    for j in range(0, self.K):
      dist = self.D[j]
      new_row = c_phi*np.array([np.exp(dist.local_obs_count(S.data,
                                  S.gamma, j, l, a, b))
          for l in range(0, dist.L)])
      new_row += self.u_D[j].get_natural()
      new_row -= self.D[j].prior.get_natural()
      temp = (1. - rho)*self.w_A[j].get_natural() + rho*new_row
      dist.prior.set_natural(temp)

  def VB_step(self, S):
    """
    Does a single VB step.

    Args:
      S: a States object

    Effects:
      self: updates this VBHMM's fields.
    """
    #TODO: implement this so that it isn't dependent on Multinoulli
      #emissions (CURRENTLY DEPENDEONT ON MULTINOULLIS)
  
    #now do the M_step
    #first update A:
    #sum over all sequences x; use update natural param methods for A
    trans = np.exp(S.get_trans())
    for j in range(0, self.K):
      new_row = self.u_A.get_natural() + trans[j]
      self.w_A[j].set_natural(new_row)

    # update the natural params of the priors on our emissions
    for j in range(0, self.K):
      dist = self.D[j]
      new_row = np.array([np.exp(dist.obs_count(S.data, S.gamma, j, l))
          for l in range(0, dist.L)])
      dist.prior.set_natural(new_row + self.u_D[j].get_natural())

    #next do simple calculation for pi
    new_pi = self.u_pi.get_natural() + np.exp(S.get_start())
    self.w_pi.set_natural(new_pi)

    #now do the e step
    #first set S so it points to our auxiliary HMM
    S.M = self.gen_M()
    #next do an e_step on the states
    S.e_step()

  def gen_M(self, local=False):
    """
    Generates an HMM M to be used in message passing.
    """
    #TODO: extend this so it works across emissions; THIS IS CURRENTLY
      #DEPENDENT ON MULTINOULLI EMISSIONS; won't be too hard to extend though
      # just need to make a method like get_expected_log which returns
      # a distribution

    #compute A
    A = []
    for j in range(0, self.K):
      w = self.w_A[j].get_natural()
      temp = dg(np.sum(w))
      row = [np.exp(dg(w[k]) - temp) for k in range(0, self.K)]
      A.append(np.array(row))
    A = np.array(A)
    print("A = " + str(A))
    pi = []
    if local:
      #compute pi
      eigval, eigvec = np.linalg.eig(A)
      maxind = max((l, i) for i, l in enumerate(eigval))[1]
      pi = eigvec[:,maxind]
      if np.min(pi) < 0:
        pi = -pi
      print("pi = " + str(pi))
    else:  
      w = self.w_pi.get_natural()
      temp = dg(np.sum(w))
      row = [np.exp(dg(w[k]) - temp) for k in range(0, self.K)]
      pi = np.array(row)

    #make distributions
    Dists = []
    for j in range(0, self.K):
      w = self.D[j].prior.get_natural()
      temp = dg(np.sum(w))
      row = np.array([np.exp(dg(w[l]) - temp) for l in range(0, self.D[j].L)])
      curr = Multinoulli(row)
      Dists.append(curr)
    M = HMM(self.K, A, pi, Dists)
    return M 
      
  def elbo(self, S):
    """
    Returns the Evidence Lower Bound (elbo) of this VBHMM with states
    S.

    Args:
      S: a States object which points to a current auxiliary HMM for this
        VBHMM

    Returns:
      elbo
    """
    res = -self.u_pi.KL_div(self.w_pi)
    print("KL(w_pi || u_pi) = " + str(-res))
    for j in range(0, self.K):
      #print("find self.w_A[j] = " + str(self.w_A[j].get_natural() + 1.))
      #print("find self.u_pi = " + str(self.u_pi.get_natural() + 1.))
      temp = self.u_A.KL_div(self.w_A[j])
      print("A[" + str(j) + "]: find their KL_div is " + str(temp))
      res -= temp
      temp = self.u_D[j].KL_div(self.D[j].prior)
      if temp < 0:
        print("self.D[j].prior = " + str(self.D[j].prior.get_natural() + 1.))
        print("self.u_D[j] = " + str(self.u_D[j].get_natural() + 1.))

      print("D[" + str(j) + "]: find their KL_div is " + str(temp))
      res -= temp

    print("find S.LL() = " + str(S.LL()))
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
      res += str(self.D[k].prior.get_natural() + 1.) + "\n"
    return res

