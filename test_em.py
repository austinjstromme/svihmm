import numpy as np
from HMM import HMM
from Multinoulli import Multinoulli
import matplotlib.pyplot as plt

def biased_coins_HMM(p, q, t_1, t_2):
  """
  This method returns a 2 state HMM, where state 1 is a p-biased coin,
  state 2 is a q-biased coin, the probability of staying in state
  1 is t_1, and the probability of staying in state 2 is t_2.
  The initial distribution pi is uniform.
  """
  K = 2
  A = np.array([[t_1, 1. - t_1], [1. - t_2, t_2]])
  pi = np.array([.5, .5])
  D = [Multinoulli([p, 1. - p]), Multinoulli([q, 1. - q])]
  return HMM(K, A, pi, D)


p = 0.9 
q = 0.1
t_1 = 0.05
t_2 = 0.8
M_true = biased_coins_HMM(p, q, t_1, t_2)
num_steps = 100 # controls length of observation sequences
N = 100 # number of observation sequences

# generate observation seqeunces
x = [M_true.gen_obs(num_steps) for j in xrange(0, N)]

# now initialize the model we will be learning to uniform 
  # everything:
M_learner = biased_coins_HMM(.4, .6, .5, .5)
#print(" at first, M_learner = " + str(M_learner))

em_steps = 10
train_likelihood = []
for j in range(0, em_steps):
  M_learner.EM_step(x)
  neg_log_like = -M_learner.log_likelihood(x)
  train_likelihood.append(neg_log_like)

plt.plot(train_likelihood)
plt.title("NLL vs steps, EM for HMM")
#plt.savefig("demo")
plt.show()





