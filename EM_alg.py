import numpy as np
import fb_main as fb


def EM_main(x, A, B, pi):
  """
  x is a list of N observations, each a T_ix1 np.array
  A is a KxK np.array of transition probabilities
  B is a Kx1 list of emitters (see emitter.py)
  pi is a Kx1 np.array of starting probabilites

  Performs one round of EM for HMM (Baum-Welch)
  with x, A, B, pi as the initial values.

  EFFECTS: updates A, B, pi using EM.
  """
  N = len(x)
  K = len(A)
  s = B[0].num_states()
  gamma_data = []
  xi_data = []
  for i in range(0,N):
    [gamma, xi] = fb.fb_main(x[i], A, B, pi)
    gamma_data.append(gamma)
    xi_data.append(xi)

  pi_expect = [start_count(gamma_data, k)/N for k in xrange(0,K)]
  A_expect = [[trans_count(xi_data, j, k) for k in xrange(0,K)] 
                for j in xrange(0,K)]
  B_expect = [[obs_count(x, gamma_data, j, l)/state_count(gamma_data, j)
                for l in range(0,s)] for j in xrange(0,K)]

  #in -place update everybody now
  for k in range(0, K):
    pi[k] = pi_expect[k]

  for j in range(0, K):
    row_sum = sum(A_expect[j])
    for k in range(0, K):
      A[j][k] = A_expect[j][k]/row_sum

  for j in range(0, K):
    for l in range(0, s):
      B[j][l] = B_expect[j][l]

   return

def trans_count(xi_data, j, k)
  """
  returns the expected number of transitions from
  state j to state k
  """
  N = len(gamma_data)
  return sum([sum(xi_data[i][:][j][k]) for i in xrange(0,N)])

def start_count(gamma_data, k):
  """
  returns the expected number of observation seqeunces
  which started at state k
  """
  N = len(gamma_data)
  return sum([gamma_data[i][0][k] for i in xrange(0,N)])

def state_count(gamma_data, j):
  """
  returns the expected number of times an observation
  was in state j
  """
  N = len(gamma_data)
  res = 0.0
  for i in range(0,N):
    res += sum([gamma_data[i][t][j] for t in xrange(0,N)])
  return res

def obs_count(x, gamma_data, j, l):
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
