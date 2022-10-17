import math
import numpy as np

## Forward to learn alpha ##
## bacause the alpha will be too small in the large t
## here will use scaled alpha to replace alpha
## origin foward formula: alpha[t, j] = alpha[t - 1] @ a[:, j] * b[j, O[t]]

# O: target segment
# a: transition prabability
# b: emission probability
# pi: initial probability
def Forward(O, a, b, pi, isBW=False):
    # init parameters
    # scaled alpha (hat alpha)
    scaled_alpha = np.zeros((O.shape[0], a.shape[0]))
    alpha = np.zeros(a.shape[0])  # alpha
    C = np.zeros(O.shape[0])  # scaling factor
    C[0] = np.sum(pi * b[:, O[0]])
    scaled_alpha[0, :] = (pi * b[:, O[0]]) / C[0]  # init scaled alpha
    P = 0  # probability of the observation

    # compute alpha, t: time t
    for t in range(1, O.shape[0]):
        for j in range(a.shape[0]):
            alpha[j] = scaled_alpha[t - 1] @ a[:, j] * b[j, O[t]]
            C[t] += alpha[j]

        C[t] = 1 / C[t]
        P += math.log2(C[t])
        scaled_alpha[t, :] = alpha * C[t]  # update alpha

    P *= -1
    # print("Oberservation Probability: ", P)

    ## if is called by Baum-Welch, return scaled_alpha
    ## if is called by other function, return P
    if isBW:
        return scaled_alpha
    else:
        return P

## Backward to learn beta ##
## use scaled beta to avoid underflow
## orign formula: beta[t + 1] = (beta[t + 1] * b[:, O[t + 1]]) @ a[j, :]


def Backward(O, a, b):
    # init beta
    scaled_beta = np.zeros((O.shape[0], a.shape[0]))  # scaled beta
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
    # print("Oberservation Backward Probability: ", P)

    return scaled_beta

## BaumWelch to learn HMM parameters ##


def BaumWelch(O, a, b, pi, n_iter=100):
    N = a.shape[0]
    T = len(O)  # total time

    for i in range(n_iter):
        alpha = Forward(O, a, b, pi, True)
        # print("alpha:", alpha)
        beta = Backward(O, a, b)
        # print("beta:", beta)

        xi = np.zeros((N, N, T - 1))  # init xi

        # compute xi
        # scaled alpha and beta wont effect xi
        # xi: probability of being in state i at time t and state j at time t + 1
        for t in range(T - 1):
            denominator = np.dot(
                np.dot(alpha[t, :].T, a) * b[:, O[t + 1]].T, beta[t + 1, :])
            for i in range(N):
                numerator = alpha[t, i] * a[i, :] * \
                    b[:, O[t + 1]].T * beta[t + 1, :].T
                xi[i, :, t] = numerator / denominator

        # use xi to compute gamma
        # scaled alpha and beta wont effect xi
        # gamma : probability of being in state i at time t (fix i and t then sum all j)
        gamma = np.sum(xi, axis=1)

        # use xi and gamma to update a
        # sum xi and gamma over time t
        # new a = sum(xi) / sum(gamma)
        # because sum(xi, 2) is two demension and sum(gamma, axis=1) is one demension, gamma need to be expand to two demension
        # fix row, reshape column
        a = np.sum(xi, 2) / np.sum(gamma, axis=1).reshape((-1, 1))

        # Add additional T'th element in gamma (because xi is T-1)
        # hsack: stack arrays in sequence horizontally (column wise)
        # use the origin gamma formula to compute the last gamma
        denominator = alpha[T - 1] @ beta[T - 1]
        gamma = np.hstack((gamma, np.expand_dims(
            (alpha[T - 1] * beta[T - 1]) / denominator, axis=1)))

        # use gamma to update b
        # K is the number of observation types (four in this case)
        K = b.shape[1]
        denominator = np.sum(gamma, axis=1)
        for l in range(K):
            b[:, l] = np.sum(gamma[:, O == l], axis=1)

        b = np.divide(b, denominator.reshape((-1, 1)))

    return a, b
