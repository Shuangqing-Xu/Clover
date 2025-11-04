# Harnessing Sparsification in Federated Learning: A Secure, Efficient, and Differentially Private Realization

This implementation accompanies our paper by Shuangqing Xu, Yifeng Zheng and Zhongyun Hua at ACM CCS'25.

## Usage

Download the repository and install all required packages as listed in requirements.txt.

#### Federated Differentially Private Model Training on MNIST

To evaluate the model utility on MNIST, run the following command

``` shell
python Clover-main.py --model 'lenet5' --dataset 'mnist' --partition_method 'IID' --method='FedAvg' --batch_size 50 --lr 0.1 --epochs 30 --client_num_in_total 100 --frac 0.1 --comm_round 120 --C 0.1 --sigma 0.95 --gpu 1 --alpha 0.01
```
#### Federated Differentially Private Model Training on CIFAR-10

    To evaluate the model utility on CIFAR-10, run the following command

``` shell
python Clover-main.py --model 'resnet18' --dataset 'cifar10' --partition_method 'IID' --method='FedAvg' --batch_size 50 --lr 0.1 --epochs 30 --client_num_in_total 100 --frac 0.1 --comm_round 300 --C 0.2 --sigma 0.95 --gpu 0 --alpha 0.01
```

#### Federated Differentially Private Model Training on FashionMNIST

To evaluate the model utility on FashionMNIST, run the following command

``` shell
python Clover-main.py --model 'lenet5' --dataset 'fmnist' --partition_method 'IID' --method='FedAvg' --batch_size 50 --lr 0.1 --epochs 30 --client_num_in_total 100 --frac 0.1 --comm_round 300 --C 0.1 --sigma 0.95 --gpu 1 --alpha 0.01
```

#### Maliciously Security

To run the maliciously secure secret-shared shuffle, run the following command

``` shell
cd MaliciouslySecurity
python rss_shuffle_mali.py
```

**WARNING**: This is an academic proof-of-concept prototype and has not received careful code review. This implementation is NOT ready for production use.

## Acknowledgments

- Part of our federated differentially private model training implementation is based on the public implementation in Shi et al.'s [paper](https://arxiv.org/abs/2303.11242) with code repo [here](https://github.com/YMJS-Irfan/DP-FedSAM)
