# Adaptive Factorization Network
This is our Tensorflow implementation for the paper:

>Weiyu Cheng, Yanyan Shen, Linpeng Huang. Adaptive Factorization Network: Learning Adaptive-Order Feature Interactions. In AAAI'20, New York, NY, USA, February 07-12, 2020.

Author: Weiyu Cheng (weiyu_cheng at sjtu.edu.cn)

## Introduction
We propose the Adaptive Factorization Network (AFN), a new model that learns arbitrary-order cross features adaptively from data. The core of AFN is a logarithmic transformation layer that converts the power of each feature in a feature combination into the coefficient to be learned. The experimental results on CTR prediction tasks demonstrate the superior predictive performance of AFN against the state-of-the-arts.


## Citation 
If you want to use our codes in your research, please cite:
```
@article{cheng2019adaptive,
  title={Adaptive Factorization Network: Learning Adaptive-Order Feature Interactions},
  author={Cheng, Weiyu and Shen, Yanyan and Huang, Linpeng},
  journal={arXiv preprint arXiv:1909.03276},
  year={2019}
}
```
## Environment Requirement
The code has been tested running under Python 3.6. The required packages are as follows:
* tensorflow == 1.10.0

## About Dataset
Here we only provide the Frappe and Movielens datasets in this repository due to the large size of the other two datasets. For the Criteo and Avazu datasets, we have uploaded our preprocessed data to codalab. If you'd like to also run experiments on Criteo and Avazu datasets, **please first run the downloading script**:
```
cd src
python download_criteo_and_avazu.py
```

## Example to Run the Codes
```
cd src
sh ./run_experiments.sh
```

