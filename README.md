# BayesSum
This repository contains the implementation of the code for the paper "BayesSum".

## Installation
To install the required packages, run the following command:
```bash
pip3 install -r requirements.txt
```

## Reproducing Results
1. Synthetic Data
   To reproduce the results for the synthetic experiment Figure 1 and Figure 2(Left), run the following files:
  - Poisson: `python3 poisson.py`
  - Uniform: `python3 potts.py --figure uniform` (Figure 1) `python3 potts.py --figure slope` (Table of convergence rate) `python3 potts.py --figure dim_boxplot` (Figure 2 Right), 
    `python3 potts.py --figure lambda_ablation` (Figure 2 Left)
  - Potts model: `python3 unnormalised_potts.py`
  - Mixed space: `python3 mixed_bq.py`
    
2. Sales Data
   To reproduce the results for the synthetic experiment Figure 2(Right) and Figure 3, run the following command:
   - `python3 cmp_salesdata.py`
   
4. Synthetic Protein Sequence
   To reproduce the results for the synthetic experiment Figure 4, run the following command:
   - `python3 protein_potts_mll_mix.py`
