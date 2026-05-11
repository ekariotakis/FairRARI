# FairRARI: A Plug and Play Framework for Fairness-Aware PageRank

[![paper](https://img.shields.io/badge/Paper-arXiv-red)](https://arxiv.org/abs/2602.08589)
&nbsp;

This is a public code repository for our publication in the International Conference on Machine Learning (ICML) 2026:
> [**FairRARI: A Plug and Play Framework for Fairness-Aware PageRank**](https://arxiv.org/abs/2602.08589)<br>
> Emmanouil Kariotakis, Aritra Konar <br>

## Abstract
PageRank (PR) is a fundamental algorithm in graph machine learning tasks. Owing to the increasing importance of algorithmic fairness, we consider the problem of computing PR vectors subject to various group-fairness criteria based on sensitive attributes of the vertices. At present, principled algorithms for this problem are lacking -  some cannot guarantee that a target fairness level is achieved, while others do not feature optimality guarantees. In order to overcome these shortcomings, we put forth a unified in-processing convex optimization framework, termed FairRARI, for tackling different group-fairness criteria in a ``plug and play'' fashion. Leveraging a variational formulation of PR, the framework computes fair PR vectors by solving a strongly convex optimization problem with fairness constraints, thereby ensuring that a target fairness level is achieved. We further introduce three different fairness criteria which can be efficiently tackled using FairRARI to compute fair PR vectors with the same asymptotic time-complexity as the original PR algorithm. Extensive experiments on real-world datasets showcase that FairRARI outperforms existing methods in terms of utility, while achieving the desired fairness levels across multiple vertex groups; thereby highlighting its effectiveness.

## File Arrangement

Here we summarize all files present in this repo and their purpose.
```
+-- datasets/: 
    all the datasets used

+-- fairPageRank.py: 
    implementations of FairRARI and Post-Processing, for all proposed group-fairness criteria (2 groups)
+-- fairPageRank_4c.py: 
    implementations of FairRARI and Post-Processing, for all proposed group-fairness criteria (4 groups)

+-- run_sum_fair_FairRARI.py:    
    example code to run FairRARI for \phi-sum-fairness (2 groups)
+-- run_sum_fair_FairRARI_4c.py: 
    example code to run FairRARI for \phi-sum-fairness (4 groups)
+-- run_sum_fair_post_processing.py:    
    example code to run Post-Processing for \phi-sum-fairness (2 groups)
+-- run_sum_fair_post_processing_4c.py: 
    example code to run Post-Processing for \phi-sum-fairness (4 groups)

+-- run_min_fair_FairRARI.py:        
    example code to run FairRARI for \alpha-min-fairness (2 groups)
+-- run_min_fair_post_processing.py: 
    example code to run Post-Processing for \alpha-min-fairness (2 groups)

+-- run_sum_min_fair_FairRARI.py:        
    example code to run FairRARI for \phi-sum + \alpha-min-fairness (2 groups)
+-- run_sum_min_fair_post_processing.py: 
    example code to run Post-Processing for \phi-sum + \alpha-min-fairness (2 groups)

+-- utils.py: 
    some general utils used

+-- exec_sum_fair.sh: 
    bash script example to execute FairRARI and Post-Processing, for \phi-sum-fairness, for various \phi values and datasets
+-- exec_min_fair.sh: 
    bash script example to execute FairRARI and Post-Processing, for \alpha-min-fairness, for various \alpha values and datasets
+-- exec_sum_min_fair.sh: 
    bash script example to execute FairRARI and Post-Processing, for \phi-sum + \alpha-min fairness, for various \phi and \alpha values and datasets

+-- create_network_plots.ipynb: 
    jupyter notebook to reproduce Fig. 1 of the paper
```

## Getting Started

The required packages are the following:
```
networkx==3.3
numpy==2.1.3
scipy==1.14.0
torch==2.5.1
tqdm==4.67.0
```

To reproduce the main experiments, you can run:

```bash
# Run FairRARI and Post-Processing with \phi-sum-fairness (2 groups)
sh exec_sum_fair.sh
# Run FairRARI and Post-Processing with \phi-sum-fairness (4 groups)
sh exec_sum_fair_4c.sh
# Run FairRARI and Post-Processing with \alpha-min-fairness
sh exec_min_fair.sh
# Run FairRARI and Post-Processing with \phi-sum + \alpha-min fairness
sh exec_sum_min_fair.sh
# Generate Fig.1
create_network_plots.ipynb
```
