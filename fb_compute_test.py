import numpy as np
import fb_compute as fb
from multinoulli_emitter import mn_emitter as mn

def umbrella_example():
  """
  Umbrella example from wikipedia
  """
  x = np.array([0,0,1,0,0])
  A = np.array([[0.7,0.3],[0.3,0.7]])
  B =[mn([0.9,0.1]), mn([0.2,0.8])]
  pi = np.array([0.5,0.5])
  gamma = fb.fb_main(x, A, B, pi)
  gamma_correct = np.array([[0.8673,0.1327],[0.82,.18],
    [0.31,0.70],[0.82,0.18],[0.87,0.13]])
  return check_gamma(gamma, gamma_correct, .1)

def easy_case():
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
  x = np.array([0 for j in range(0,T)] + [1 for j in range(0,T)])
  B = [mn(np.array([1.0, 0.0])), mn(np.array([0.0, 1.0]))]
  A = np.array([[0.5,0.5],[0.5,0.5]])
  pi = [0.5,0.5]
  gamma = fb.fb_main(x, A, B, pi)
  gamma_correct = np.array([[1.0,0.0] for j in range(0,T)] 
    + [[0.0,1.0] for j in range(0,T)])
  return check_gamma(gamma, gamma_correct, .1)

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

print("Running forward backwards alg tests...")
easy_result = int(easy_case())
umbrella_result = int(umbrella_example())
correct = easy_result + umbrella_result
print("  easy case: " + str(easy_result) + "/1")
print("  umbrella case: " + str(umbrella_result) + "/1")
print("Test results: " + str(correct) + "/2")
