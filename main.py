import math
from pyfaidx import Fasta
import numpy as np
from sklearn import preprocessing
labelencoder = preprocessing.LabelEncoder() # use to encode the DNA sequence

genes = Fasta('GRCh38_latest_genomic.fna') # get the gene file

S = genes.get_seq('NC_000006.12', 100001, 1100000) # get S segment
S = str(S) # change to string
S = S.lower()  # change to lowercase
S = list(S) # change to list
S = labelencoder.fit_transform(S)  # encode S; a: 0, c: 1, g: 2, t: 3

T = genes.get_seq('NC_000007.14', 100001, 1100000) # get T segment
T = str(T) # change to string
T = T.lower() # change to lowercase
T = list(T)  # change to list
T = labelencoder.fit_transform(T)  # encode T; a: 0, c: 1, g: 2, t: 3

# Transition Probabilities
# two state in the problem: 0 and 1
# a = [[0.7, 0.3]
#      [0.2, 0.8]]
# a[i, j]: statei -> statej
a = np.array([[0.7, 0.3], [0.2, 0.8]])

# Emission Probabilities
# b = [[0.4, 0.3, 0.1, 0.2]
#      [0.2, 0.2, 0.3, 0.3]]
# b[i, j]: statei to generate j (a, t, c, g)
b = np.array([[0.4, 0.3, 0.1, 0.2], [0.2, 0.2, 0.3, 0.3]])

# Equal Probabilities for the initial distribution
pi = np.array((0.5, 0.5))

## plain Markov Model ( Variable Oder Markov Model, vomm) ##
def plainMarkov(O, n_state, order):
    
    p = np.zeros(n_state) # probability of each state

    if order == 1 or 2:
        p_1 = np.zeros((4, 4))   # probability in order 1
    if order == 2:
        p_2 = np.zeros((4, 4, 4))   # probability in order 2
    
    Olen = len(O)  # the length of O, the denominator of probability
    total_p = 0 # total probability, show in log base 2

    # count the total number of each state
    for i in range(0, Olen):
        p[O[i]] += 1
        if (order == 1 or 2) and (i != 0):
            p_1[O[i], O[i - 1]] += 1
        if (order == 2) and (i != 0 or 1):
            p_2[O[i], O[i - 1], O[i - 2]] += 1

    # probability of each state
    p = p / Olen
    if order == 1 or 2:
        p_1 = p_1 / Olen
    if order == 2:
        p_2 = p_2 / Olen

    if order == 0:
        # compute total state
        for i in range(0, Olen):
            total_p += math.log2(p[O[i]])
    elif order == 1:
        total_p += math.log2(p[O[0]])
        for i in range(1, Olen):
            total_p += math.log2(p_1[O[i], O[i - 1]])
    elif order == 2:   
        total_p += math.log2(p[O[0]])
        total_p += math.log2(p_1[O[1], O[0]])
        for i in range(2, Olen):
            total_p += math.log2(p_2[O[i], O[i - 1], O[i - 2]])
                
    return total_p

## Forward to learn alpha ##
## bacause the alpha will be too small in the large t
## here will use scaled alpha to replace alpha
## origin foward formula: alpha[t, j] = alpha[t - 1] @ a[:, j] * b[j, O[t]]

# O: target segment
# a: transition prabability
# b: emission probability
# pi: initial probability
def Forward(O, a, b, pi):
    for i in range(0, 5):
        print(O[i])
        
    # init alpha
    scaled_alpha = np.zeros((O.shape[0], a.shape[0])) # scaled alpha (hat alpha)
    alpha = np.zeros(a.shape[0]) # alpha
    C = np.zeros(O.shape[0])  # scaling factor
    C[0] = np.sum(pi * b[:, O[0]]) 
    scaled_alpha[0, :] = (pi * b[:, O[0]]) / C[0] # init scaled alpha
    P = 0 # probability of the observation

    # compute alpha, t: time t
    for t in range(1, O.shape[0]):
        for j in range (a.shape[0]):
            alpha[j] = scaled_alpha[t - 1] @ a[:, j] * b[j, O[t]]
            C[t] += alpha[j]

        C[t] = 1 / C[t]
        P += math.log2(C[t])
        scaled_alpha[t, :] = alpha * C[t] # update alpha

    P *= -1
    print("Oberservation Probability: ", P)

    return scaled_alpha

## Backward to learn beta ##
## use scaled beta to avoid underflow
def Backward(O, a, b):
    # init beta
    scaled_beta = np.zeros((O.shape[0], a.shape[0])) # scaled beta
    beta = np.zeros(a.shape[0])
    C = np.zeros(O.shape[0])
    P = 0  # probability of the backward observation

    # setting beta(T) = 1
    scaled_beta[O.shape[0] - 1] = np.ones((a.shape[0]))

    # compute beta (from time t-1 to 1)
    for t in range(O.shape[0] - 2, -1, -1):
        for j in range(a.shape[0]):
           beta[j] = (scaled_beta[t + 1] * b[:, O[t + 1]]) @ a[j, :]
           C[t + 1] += beta[j]
        C[t + 1] = 1 / C[t + 1]
        P += math.log2(C[t + 1])
        scaled_beta[t, :] = beta * C[t + 1]

    P *= -1
    print("Oberservation Backward Probability: ", P)

    return scaled_beta

## BaumWelch to learn HMM parameters ##
def BaumWelch(O, a, b, pi, n_iter=100):
    N = a.shape[0] # 
    T = len(O) # total time

    for i in range (n_iter):
        alpha = Forward(O, a, b, pi)
        print("alpha:", alpha)
        beta = Backward(O, a, b)
        print("beta:", beta)

        xi = np.zeros((N, N, T - 1)) #init xi

        # compute xi
        for t in range(T - 1):
            denominator = np.dot(
                np.dot(alpha[t, :].T, a) * b[:, O[t + 1]].T, beta[t + 1, :])
            for i in range(N):
                numerator = alpha[t, i] * a[i, :] * b[:, O[t + 1]].T * beta[t + 1, :].T
                xi[i, :, t] = numerator / denominator
        
        # use xi to compute gamma
        gamma = np.sum(xi, axis=1)

        # use xi and gamma to update a
        
        a = np.sum(xi, 2) / np.sum(gamma, axis=1).reshape((-1, 1))  # fix row, reshape column

        # Add additional T'th element in gamma (because we start form T - 1)
        gamma = np.hstack(
            (gamma, np.sum(xi[:, :, T - 2], axis=0).reshape((-1, 1))))  # fix row, reshape column

        # use gamma to update b
        K = b.shape[1]
        denominator = np.sum(gamma, axis=1)
        for l in range(K):
            b[:, l] = np.sum(gamma[:, O == l], axis=1)

        b = np.divide(b, denominator.reshape((-1, 1)))

    return a, b

## Execute plain Markov Model ##
# print("**plain Markov Model**")
# print("probability of order 0:", plainMarkov(S, 4, 0))
# print("probability of order 1:", plainMarkov(S, 4, 1))
# print("probability of order 2:", plainMarkov(S, 4, 2))

## Execute Baum-Welch Algorithm ## 
#train model
n_iter = 5
a_model, b_model = BaumWelch(S, a.copy(), b.copy(), pi.copy(), n_iter=n_iter)

print("**Baum-Welch Algorithm**")
print(f'Custom model A is \n{a_model} \n \nCustom model B is \n{b_model}')
