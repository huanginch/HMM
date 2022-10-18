import math
import pandas as pd
from pyfaidx import Fasta
import numpy as np
from sklearn import preprocessing
import MM
import HMM
import re

labelencoder = preprocessing.LabelEncoder()  # use to encode the DNA sequence
labelencoder.fit(['a', 'c', 'g', 't'])  # fit the DNA sequence

genes = Fasta('GRCh38_latest_genomic.fna') # get the gene file

S = genes.get_seq('NC_000006.12', 100000, 1100000) # get S segment
S = str(S) # change to string
S = S.lower()  # change to lowercase
S = list(S) # change to list
S = labelencoder.transform(S)  # encode S; a: 0, c: 1, g: 2, t: 3

# Transition Probabilities
# two state in the problem: 0 and 1
# a = [[0.7, 0.3]
#      [0.2, 0.8]]
# a[i, j]: statei -> statej
a = np.array([[0.68, 0.32], [0.24, 0.76]])

# Emission Probabilities
# b = [[0.4, 0.3, 0.1, 0.2]
#      [0.2, 0.2, 0.3, 0.3]]
# b[i, j]: statei to generate j (a, t, c, g)
b = np.array([[0.1, 0.3, 0.05, 0.55], [0.4, 0.1, 0.3, 0.2]])

# Equal Probabilities for the initial distribution
pi = np.array((0.3, 0.7))

## Execute plain Markov Model ##
print("**plain Markov Model**")
print("probability of order 0:", MM.plainMarkov(S, 4, 0))
print("probability of order 1:", MM.plainMarkov(S, 4, 1))
print("probability of order 2:", MM.plainMarkov(S, 4, 2))

## Execute Baum-Welch Algorithm ## 
print("**Baum-Welch Algorithm**")
print("Computing...")

#train model
n_iter = 50
a_model, b_model = HMM.BaumWelch(S, a.copy(), b.copy(), pi.copy(), n_iter=n_iter)

print(f'Custom model A is \n{a_model} \n \nCustom model B is \n{b_model}')
print("HMM Observation Probability for S:", HMM.Forward(S, a_model, b_model, pi))

T = genes.get_seq('NC_000007.14', 100000, 1100000)  # get T segment
T = str(T)  # change to string
T = T.lower()  # change to lowercase
list_T = re.split('n+', T)  # split the sequence by 'n'
P_T = 0  # total probability of T

for i in range(len(list_T)):
    subT = list(list_T[i])  # change to list
    subT = labelencoder.transform(subT)  # encode S; a: 0, c: 1, g: 2, t: 3
    P_T += HMM.Forward(subT, a_model, b_model, pi)

print("HMM Observation Probability for T:", P_T)

# get my_chromosome segment
my_chromosome = genes.get_seq('NC_000009.12', 100000, 1100000)
my_chromosome = str(my_chromosome)  # change to string
my_chromosome = my_chromosome.lower()  # change to lowercase
my_chromosome = list(my_chromosome)  # change to list
# encode my_chromosome; a: 0, c: 1, g: 2, t: 3
my_chromosome = labelencoder.transform(my_chromosome)

print("HMM Observation Probability for My Chromosome:", HMM.Forward(my_chromosome, a_model, b_model, pi))
