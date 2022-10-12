# Markov Model and Hidden Markov Model in Python
>[Problem Link](https://paulhorton.gitlab.io/coursesTaught/GenomeInformatics/GenomeInformatics202210_homework1.html)

* 姓名: 黃映慈
* 學號: P76114309

## Problem 1: plain Markov Model (Variable Order Markov Model, VOMM)
1. 根據order0、1、2來計算每種組合在S(目標基因體序列)中發生的數量
2. 計算S的長度
3. 根據order0、1、2來計算每種組合發生的機率
4. 將機率相乘
5. 執行結果(以log2為底表示): 
    > * probability of order 0: -1987487.013403616
    > * probability of order 1: -3932508.748509369
    > * probability of order 2: -5865361.526298552
* 結論: **以 order 2 計算發生的機率明顯較低**

## Problem 2: Use Baum-Welch Algorithm to learn HMM parameters
### Forward Algorithm
### Backward Algorithm
### Baum-Welch Algorithm
