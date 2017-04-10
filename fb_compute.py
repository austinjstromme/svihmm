import numpy as np

def fb_main(x, A, B, pi):
  """
  x is a 1xT np.array of observations
  A is a KxK np.array of transition probabilities
  B is an emitter (see emitter.py)
  pi is a 1xK np.array of starting probabilites

  Computes the smooth posterior marginals gamma[t][j]
   using the forwards backwards algorithm.

  RETURNS: gamma, a TxK np.array where gamma[t][j]
   is the probability z_t = j conditioned on the 
   observations x.
  """

  alpha = forward_alg(x, A, B, pi)[0]
  beta = backward_alg(x, A, B, pi)

  gamma = np.array([normalize(np.multiply(alpha[t], beta[t]))[0] 
    for t in range(0,len(x))])
      
  return gamma
  
def forward_alg(x, A, B, pi):
  """
  x is a 1xT np.array of observations
  A is a KxK np.array of transition probabilities; A[i][j] is prob.
   of i -> j
  B is a 1xK list of emitters; the kth on describes the
   emissions from state k
  pi is a 1xK np.array of initial state probabilites

  This method computes the forward direction of the
  forward backward algorithm.

  RETURNS: [alpha, Z] where alpha is a TxK np.array of 
    belief probabilities, and Z is a 1xT np.array of 
    normalization constants as in Murphy 17.4.2
  """
  alpha = []
  Z = []
  psi = np.array([B[i].prob_obs(x[0]) for i in range(0,len(B))])
  res = normalize(np.multiply(psi, pi))
  alpha.append(res[0])
  Z.append(res[1])

  for t in range(1, len(x)):
    psi = np.array([B[i].prob_obs(x[t]) for i in range(0,len(B))])
    res = normalize(np.multiply(psi, np.transpose(A).dot(alpha[t-1])))
    alpha.append(res[0])
    Z.append(res[1])
  
  return [np.vstack(alpha), np.array(Z)]

def backward_alg(x, A, B, pi):
  """
  x is a 1xT np.array of observations
  A is a KxK np.array of transition probabilities
  B is an emitter (see emitter.py)
  pi is a 1xK np.array of starting probabilites

  Does the backward algorithm as described in Murphy, 17.4.3.

  RETURNS: beta where beta is a TxK np.array 
    such that beta[t][j] is the conditional
    likelihood of future evidence given that
    the hidden state at time t is j. 
  """

  beta = []
  beta.append(np.array([1.0 for j in range(0, len(pi))]))
  
  for t in range(len(x) - 2, -1, -1):
    psi = np.array([B[i].prob_obs(x[t + 1]) for i in range(0, len(B))])
    beta.insert(0, normalize(A.dot(np.multiply(psi, beta[0])))[0])

  return np.vstack(beta)

def normalize(u):
  """
  u is a 1xn np.array

  RETURNS: [v,Z] where Z = sum(u)
  and v is a 1xn np.array such that v[i] = u[i]/Z
  """
  Z = sum(u)
  v = np.array([u[i]/Z for i in range(0,len(u))])
  return [v,Z]



