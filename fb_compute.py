import numpy as np

def fb_main(x, A, B, pi):
  """
  x is a Tx1 list of observations
  A is a KxK np.array of transition probabilities
  B is a list of K distributions specifying the observation
    model from each state (see Distribution.py)
  pi is a Kx1 np.array of starting probabilites

  Computes the smooth posterior marginals gamma[t][j]
   using the forwards backwards algorithm.

  RETURNS: [gamma, xi, Z_vals], where gamma is a TxK np.array 
   where gamma[t][j] is the probability z_t = j conditioned 
   on the observations x. xi is a (T - 1)xKxK np.array of the
   two slice marginals, i.e. xi[t][i][j] = 
   p(z_t = i, z_{t + 1} = j | x). Z_vals is a Tx1 np.array
   where Z_vals[j] = p(x_j | x_{1: j - 1})
  """
  #TODO: fix this
  #psi = np.array([[B[i].prob_obs(x[t]) for i in range(0,len(B))]
  #        for t in range(0, len(x))])
  psi = [[B[k].mass(x[t]) for k in xrange(0, len(B))]
    for t in xrange(0, len(x))]

  #print("psi = " + str(psi))

  [alpha, Z_vals] = forward_alg(x, A, pi, psi)
  beta = backward_alg(x, A, pi, psi, Z_vals)

  #print("alpha = " + str(alpha))
  #print("beta = " + str(beta))

  xi = [normalize(np.multiply(A, np.outer(alpha[t],
         np.multiply(psi[t + 1], beta[t + 1]))))[0]
         for t in range(0, len(x) - 1)]

  gamma = np.array([normalize(np.multiply(alpha[t], beta[t]))[0] 
    for t in range(0, len(x))])



  return [gamma, xi, Z_vals]

def normalize_xi(xi):
  """
  Normalizes xi; this special normalization is necessary because
  xi[i][j] = xi[j][i] and sum_{i <= j} xi[i][j] = 1
  """
  norm = 0.
  for i in range(0, xi.shape[0]):
    for j in range(i, xi.shape[1]):
      norm += xi[i][j]
  return xi*(1./norm)
  
def forward_alg(x, A, pi, psi):
  """
  x is a Tx1 list of observations
  A is a KxK np.array of transition probabilities
  pi is a Kx1 np.array of starting probabilites
  psi is a TxK np.array of local evidences, i.e. psi[t][i]
    is p(x[t] | z_i) in Murphy Chapter 17 notation.

  This method computes the forward direction of the
  forward backward algorithm.

  RETURNS: [alpha, Z] where alpha is a TxK np.array of 
    belief probabilities, and Z is a 1xT np.array of 
    normalization constants as in Murphy 17.4.2
  """
  alpha = []
  Z = []
  res = normalize(np.multiply(psi[0], pi))
  alpha.append(res[0])
  Z.append(res[1])

  for t in range(1, len(x)):
    res = normalize(np.multiply(psi[t], np.transpose(A).dot(alpha[t-1])))
    alpha.append(res[0])
    Z.append(res[1])
  
  return [np.vstack(alpha), np.array(Z)]

def backward_alg(x, A, pi, psi, Z_vals):
  """
  Does the backward algorithm as described in Murphy, 17.4.3.

  Params:
    x: Tx1 np.array of observations
    A: KxK np.array of transition probabilities
    pi: Kx1 np.array of starting probabilites
    psi: TxK np.array of local evidences, i.e. psi[t][i]
      is p(x[t] | z_i) in Murphy Chapter 17 notation.
    Z_vals: Tx1 np.array of Z values, where Z_vals[t] is the
      norm of alpha[t] before it was normalized

  RETURNS: beta where beta is a TxK np.array 
    such that beta[t][j] is the conditional
    likelihood of future evidence given that
    the hidden state at time t is j. 
  """

  beta = []
  beta.append(np.array([1.0 for j in range(0, len(pi))]))
  
  for t in range(len(x) - 2, -1, -1):
    beta.insert(0, A.dot(np.multiply(psi[t + 1],
      beta[0]*Z_vals[t + 1]))*(1./Z_vals[t + 1]))

  return np.vstack(beta)

def normalize(u):
  """
  u is a np.array

  RETURNS: [v,Z] where Z = sum(u)
  and v is a np.array with the same shape as v such that v = u/Z
  """
  Z = np.sum(u)
  if Z == 0:
    v = u
  else:
    v = u * (1.0/Z)
  v = np.array(v)
  return [v,Z]
