"""
 This file does some basic testing on the forward backward algorithm
 file. It tests the gamma outputed for two simple cases that can be
 worked out by hand. It also tests xi[0] for both cases against
 what scratch work suggests it should be. 
"""
# external packages
import numpy as np

# internals
from Multinoulli import Multinoulli as mn
from HMM import HMM
from States import States

def umbrella_example(eps):
  """
  Umbrella example from wikipedia
  """

  x = [[0, 0, 1, 0, 0]]
  A = np.array([[0.7,0.3], [0.3,0.7]])
  D = [mn([0.9, 0.1]), mn([0.2, 0.8])]
  pi = np.array([0.5,0.5])
  M = HMM(2, A, pi, D)
  S = States(M, x)
  S.e_step()
  gamma = np.exp(np.array(S.gamma[0]))
  xi = np.exp(np.array(S.xi[0]))
  gamma_correct = np.array([[0.8673,0.1327],[0.82,.18],
    [0.31,0.70],[0.82,0.18],[0.87,0.13]])
  xi_correct = np.array([[.75,.12],[.07,.06]])
  gamma_score = check_gamma(gamma, gamma_correct, eps)
  xi_score = check_gamma(xi[0], xi_correct, eps)
  #if not gamma_score:
  #  print("    gamma computation failed")
  #if not xi_score:
  #  print("    xi computation failed")
  return gamma_score and xi_score

def easy_case(eps):
  """
    When K = 2, the transition matrix is 0.5 everywhere,
    and state 0 emits only 0 and state 1 emits only 1s,
    forward backward should guess 0's (p = 1) whenever it sees
    a 0 and a 1 (p = 1) whenever it sees a 1.
  """
  #set hyperparams
  K = 2
  T = 10

  #initialize estimates
  x = [[0 for j in range(0,T)] + [1 for j in range(0,T)]]
  B = [mn(np.array([.999, 0.001])), mn(np.array([0.001, .999]))]
  A = np.array([[0.5,0.5],[0.5,0.5]])
  pi = [0.5,0.5]
  M = HMM(2, A, pi, B)
  S = States(M, x)
  S.e_step()
  gamma = np.exp(np.array(S.gamma[0]))
  xi = np.exp(np.array(S.xi[0]))
  gamma_correct = np.array([[1.0,0.0] for j in range(0,T)]
    + [[0.0,1.0] for j in range(0,T)])
  xi_correct = ([np.array([[1,0], [0,0]]) for j
                in range(0, T - 1)] + 
                [np.array([[0,1], [0,0]])] +
                [np.array([[0,0], [0,1]]) for j in range(0, T - 1)])

  gamma_score = check_gamma(gamma, gamma_correct, eps)
  xi_score = True
  for j in range(0, len(xi)):
    xi_score = xi_score and check_gamma(xi[j], xi_correct[j], eps)

  return gamma_score and xi_score

def check_gamma(gamma, gamma_correct, eps):
  """
  gamma and gamma_correct are NxK
  np.arrays of smooth marginal probabilities
  eps is a small float

  RETURN: true if each entry in gamma and 
  gamma_correct are within eps, and false otherwise
  """
  for i in range(0,gamma.shape[0]):
    for j in range(0,gamma.shape[1]):
      if abs(gamma[i][j] - gamma_correct[i][j]) >= eps:
        return False
  return True

def run():
  eps = 0.1
  print("Running forward backwards alg tests " + 
      "with error tolerance = " + str(eps) + "...")
  easy_result = int(easy_case(eps))
  umbrella_result = int(umbrella_example(eps))
  correct = easy_result + umbrella_result
  print("  easy case: " + str(easy_result) + "/1")
  print("  umbrella case: " + str(umbrella_result) + "/1")
  print("Forwards backwards results: " + str(correct) + "/2")
  return correct == 2
