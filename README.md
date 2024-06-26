# CS6910_Assignment1


In this assignment I implemented a feedforward neural network along with backpropagation for Fashion MNIST dataset from scratch only using numpy, matplotlib & wandb. More specifically, given an input image (28 x 28 = 784 pixels) from the Fashion-MNIST dataset, the network will be trained to classify the image into 1 of 10 classes. I used various deep learning techniques & optimizers to train NN and get high train & validation accuracy in classifying different fashion items.

## Dataset

I imported Fashion MNIST dataset from keras.datasets which consisted of 60,000 training datasets and 10,000 test dataset belonging to 10 different fashion categories.

## Model Architecture Implemetation

- Flattened Input Layer : 784 Neurons
- Output Layer : 10 Neurons
- Activation Functions : sigmoid, ReLU, tanh
- Optimization Functions : sgd, momentum, nesterov, rmsprop, adam & nadam
- Loss Functions : cross entropy, mse
- Weight Initialization : random, Xavier
- Hidden Layers : Used different combinations of hidden layers & no. of neurons within it

## Hyperparameter Searching

Used Bayesian Sweep functionality provided by wandb to find the best values for the hyperparameters listed below :
- No. of epochs
- No. of hidden layers
- Hidden layer size
- Learning Rate
- Weight Decay
- Batch Size
- Optimizers
- Activation

## Best Model Configurations

- Optimizer : nadam
- Activation : tanh
- Loss function : cross entropy
- No. of epochs : 10
- Batch size : 64
- Learning Rate : 0.001
- Hidden Size : 128
- No. of hidden layers : 5
- Weight Initialization : Xavier
- Weight Decay : 0
- Momentum : 0.5
- Beta : 0.9

## Python Script for Training Model

I designed a python script (train.py) to train above neural network with different parameters. Inorder to execute it just run below command :
```
python train.py --wandb_project "myprojectname" --wandb_entity "myname" --dataset "fashion_mnist" --epoch 10 --batch_size 32 --loss "cross_entropy" --optimizer "nadam" --learning_rate 0.001 --activation "tanh" --num_layers 5 --hidden_size 32 --weight_decay 0
```
OR 

To just run best model just execute :
```
python train.py
```

## Results

Wandb allowed us to keep track of training process using sweep functionality and helped me to identify best set of hyperparameters that gave best accuracy. I used wandb to address Q1, Q4, Q5, Q6, Q7 of assignments and obtained the results which can be checked from the given link :  https://wandb.ai/cs23m009/DL_Assignment_1/reports/CS6910-Assignment-1--Vmlldzo3MTI3NDc0


## Requirements

Install numpy, pandas, keras (for dataset) and wandb before execution or use jupyter notebook/colab to directly visualize.





  



