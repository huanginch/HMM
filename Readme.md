# Markov Model and Hidden Markov Model in Python
>[Problem Link](https://paulhorton.gitlab.io/coursesTaught/GenomeInformatics/GenomeInformatics202210_homework1.html)

* 姓名: 黃映慈
* 學號: P76114309

## Problem 1: Plain Markov Model (Variable Order Markov Model, VOMM)
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
* 使用transition matrix、emission matrix、initial state probability來計算每個state的機率
* 從 Time = 0 開始計算，每個state的機率都是由前一個state的機率乘上transition matrix的值，再乘上emission matrix的值
* 因為數字太小會溢位所以使用log2為底表示機率
### Backward Algorithm
* 使用transition matrix、emission matrix來計算每個state的機率
* * 從 Time = T + 1 開始計算，每個state的機率都是由後一個state的機率乘上transition matrix的值，再乘上emission matrix的值
### Baum-Welch Algorithm
* 使用Forward Algorithm、Backward Algorithm來更新transition matrix、emission matrix
  * gamma: 當time = t時，在state i的機率
  * xi: 當time = t時，在state i，⽽當time = t + 1時，在state j的機率
  * 使用gamma、xi來更新transition matrix、emission matrix
* 重複執行更新transition matrix、emission matrix的步驟，這裡我使用了50次
* 利用更新後的transition matrix、emission matrix來計算目標基因序列的機率
  * 把更新的transition matrix、emission matrix帶入Forward Algorithm，求得S的機率
  * 根據題目要求將T與自選的基因序列帶入Forward Algorithm，求得T與自選的基因序列的機率
* 調整初始參數以求得最佳的結果

## Result
* 比較兩種模型在S的機率
  * Plain Markov Model: -5865361.526298552
  * Hidden Markov Model: -1980439.210874865
> HMM求得的機率明顯較高
* Probability of T:
* Probability of My Chromosome: -1977260.726176619