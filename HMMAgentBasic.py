'''
CODE NOT USED, IT DOESN'T WORK AS INTENDED

Code ajusted from the coding example:https://canvas.utwente.nl/courses/16751/assignments/145246
and from this example on wikipedia so we could ajust the model more in depth then with just parmeters:https://en.wikipedia.org/wiki/Viterbi_algorithm#Algorithm 


States:
0 -> straight
1 -> left
2 -> right
3 -> stop

Observations of gaze heading
0 -> forwards
1 -> left
2 -> right
3 -> back left
4 -> back right
'''

import numpy as np
from hmmlearn import hmm
from collections import deque

start_probs = np.array([0.7, 0.1, 0.1, 0.1])

trans_mat = np.array([
    [0.7, 0.1, 0.1, 0.1],       # Forwards to [Forwards, left, right, stop]
    [0.3, 0.4, 0.05, 0.25],     # Left to ...
    [0.3, 0.05, 0.4, 0.25],     # Right to ...
    [0.1, 0.1, 0.1, 0.7],       # Stop to ...
])

emission_mat = np.array([
    [0.6, 0.15, 0.15, 0.05, 0.05],  # P(obs|Forwards)
    [0.15, 0.4, 0.025, 0.4, 0.025],  # P(obs|Left)
    [0.15, 0.025, 0.4, 0.025, 0.4],  # P(obs|Right)
    [0.6, 0.15, 0.15, 0.05, 0.05],  # P(obs|Stop)
])

# Manually writing the math because we want to be able to set previous states as certain.

# 1. Number of hidden states
n_states = 4  # [Forwards, Left, Right, Stop]

# 2. Number of possible observation symbols
n_observations = 5  # [Forwards, left, right, backleft, backright]# 3. Define HMM model
model = hmm.CategoricalHMM(n_components=n_states, init_params="", params="")

# 4. Set model parameters manually
model.startprob_ = start_probs       # initial distribution
model.transmat_  = trans_mat
model.emissionprob_ = emission_mat

np.random.seed(42)  # for reproducibility
X, Z = model.sample(n_samples=100)

print("Generated Observations (symbol indices):", X.ravel())
print("True Hidden States:", Z)


logprob, state_sequence = model.decode(X, algorithm="viterbi")
print("Log Probability of the best state sequence:", logprob)
print("Most likely hidden states:", state_sequence)
