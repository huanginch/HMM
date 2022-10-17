import math
import pandas as pd
from pyfaidx import Fasta
import numpy as np
from sklearn import preprocessing
import MM
import HMM

labelencoder = preprocessing.LabelEncoder()  # use to encode the DNA sequence
labelencoder.fit(['a', 'c', 'g', 't'])  # fit the DNA sequence

genes = Fasta('GRCh38_latest_genomic.fna') # get the gene file

S = genes.get_seq('NC_000006.12', 100000, 1100000) # get S segment
S = str(S) # change to string
S = S.lower()  # change to lowercase
S = list(S) # change to list
S = labelencoder.transform(S)  # encode S; a: 0, c: 1, g: 2, t: 3

T = genes.get_seq('NC_000007.14', 100000, 1100000) # get T segment
T = str(T) # change to string
T = T.lower() # change to lowercase
T = list(T)  # change to list

print(T[139263])
# encode S; a: 0, c: 1, g: 2, t: 3
# if error appears, it meanns the T segment has 
# a different base other than a, c, g, t
T = labelencoder.transform(T)  
print(T[139263])

# get my_chromosome segment
my_chromosome = genes.get_seq('NC_000009.12', 100000, 1100000)
my_chromosome = str(my_chromosome)  # change to string
my_chromosome = my_chromosome.lower()  # change to lowercase
my_chromosome = list(my_chromosome)  # change to list
# encode my_chromosome; a: 0, c: 1, g: 2, t: 3
my_chromosome = labelencoder.transform(my_chromosome)

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

## Execute plain Markov Model ##
print("**plain Markov Model**")
print("probability of order 0:", MM.plainMarkov(S, 4, 0))
print("probability of order 1:", MM.plainMarkov(S, 4, 1))
print("probability of order 2:", MM.plainMarkov(S, 4, 2))

## Execute Baum-Welch Algorithm ## 
print("**Baum-Welch Algorithm**")
print("Computing...")

#train model
n_iter = 5
a_model, b_model = HMM.BaumWelch(S, a.copy(), b.copy(), pi.copy(), n_iter=n_iter)

print(f'Custom model A is \n{a_model} \n \nCustom model B is \n{b_model}')
print("HMM Observation Probability for S:", HMM.Forward(S, a_model, b_model, pi))
print("HMM Observation Probability for T:", HMM.Forward(T, a_model, b_model, pi))
print("HMM Observation Probability for My Chromosome:", HMM.Forward(my_chromosome, a_model, b_model, pi))
