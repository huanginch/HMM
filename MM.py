import numpy as np
import math

## plain Markov Model ( Variable Oder Markov Model, vomm) ##
def plainMarkov(O, n_state, order):

    p = np.zeros(n_state)  # probability of each state

    if order == 1 or 2:
        p_1 = np.zeros((4, 4))   # probability in order 1
    if order == 2:
        p_2 = np.zeros((4, 4, 4))   # probability in order 2

    Olen = len(O)  # the length of O, the denominator of probability
    total_p = 0  # total probability, show in log base 2

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
