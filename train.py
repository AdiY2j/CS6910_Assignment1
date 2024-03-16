import wandb
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
np.set_printoptions(suppress=True)
from keras.datasets import fashion_mnist
from keras.datasets import mnist

parser = argparse.ArgumentParser()
parser.add_argument('-wp', '--wandb_project', default="DL_Assignment_1", required=False, metavar="", type=str, help='Project name used to track experiments in Weights & Biases dashboard')
parser.add_argument('-we', '--wandb_entity', default="cs23m009", required=False, metavar="", type=str, help='Wandb Entity used to track experiments in the Weights & Biases dashboard')
parser.add_argument('-d', '--dataset', default="fashion_mnist", required=False, metavar="", type=str, choices= ["mnist", "fashion_mnist"], help='Dataset Name choices: ["mnist", "fashion_mnist"]') 
parser.add_argument('-e', '--epochs', default=10, required=False, metavar="", type=int, help='Number of epochs to train neural network')
parser.add_argument('-b', '--batch_size', default=64, required=False, metavar="", type=int, help='Batch size used to train neural network')
parser.add_argument('-l', '--loss', default="cross_entropy", required=False, metavar="", type=str, choices= ["mean_squared_error", "cross_entropy"], help='Loss function choices: ["mean_squared_error", "cross_entropy"]')
parser.add_argument('-o', '--optimizer', default="nadam", required=False, metavar="", type=str, choices= ["sgd", "momentum", "nag", "rmsprop", "adam", "nadam"], help='Optimization function choices: ["sgd", "momentum", "nag", "rmsprop", "adam", "nadam"]')
parser.add_argument('-lr', '--learning_rate', default=0.001, required=False, metavar="", type=float, help='Learning rate used to optimize model parameters')
parser.add_argument('-m', '--momentum', default=0.5,  required=False, metavar="", type=float, help='Momentum used by momentum and nag optimizers')
parser.add_argument('-beta', '--beta', default=0.9,  required=False, metavar="", type=float, help='Beta used by rmsprop optimizer')
parser.add_argument('-beta1', '--beta1', default=0.9,  required=False, metavar="", type=float, help='Beta1 used by adam and nadam optimizers')
parser.add_argument('-beta2', '--beta2', default=0.999,  required=False, metavar="", type=float, help='Beta2 used by adam and nadam optimizers')
parser.add_argument('-eps', '--epsilon', default=0.000001,  required=False, metavar="", type=float, help='Epsilon used by optimizers')
parser.add_argument('-w_d', '--weight_decay', default=0.0, required=False, metavar="", type=float, help='Weight decay used by optimizers')
parser.add_argument('-w_i', '--weight_init', default="Xavier", required=False, metavar="", type=str, choices=["random", "Xavier"], help='Weight Initialization choices: ["random", "Xavier"]')
parser.add_argument('-nhl', '--num_layers', default=5, required=False, metavar="", type=int, help='Number of hidden layers used in feedforward neural network')
parser.add_argument('-sz', '--hidden_size', default=128, required=False, metavar="", type=int, help='Number of hidden neurons in a feedforward layer')
parser.add_argument('-a', '--activation', default="tanh", required=False, metavar="", type=str, choices=["sigmoid", "tanh", "ReLU"], help='Activation Function choices: ["sigmoid", "tanh", "ReLU"]')
args = parser.parse_args()



wandb.login(key='')

if args.dataset == "fashion_mnist" :
  (X_train, y_train), (X_test, y_test) = fashion_mnist.load_data()
else :
  (X_train, y_train), (X_test, y_test) = mnist.load_data()

if args.loss == "mean_squared_loss":
  args.loss = "squared_loss"
if args.optimizer == "nag":
  args.optimizer = "nesterov"
if args.activation == "ReLU":
  args.activation = "relu"

X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.1, random_state=42)
X_train = np.array(X_train)
X_val = np.array(X_val)
X_test = np.array(X_test)
# Reshape input feature from 28*28 to 784*1
X_train = X_train.reshape(X_train.shape[0], -1) / 255.0
X_val = X_val.reshape(X_val.shape[0], -1) / 255.0
X_test = X_test.reshape(X_test.shape[0], -1) / 255.0

class_labels = ["T-shirt/top", "Trouser", "Pullover", "Dress", "Coat", "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot"]

images = []
labels = []
# Function to plot one sample image for each class
def plot_data(x, y, class_labels):
    plt.figure(figsize=(10, 6))

    for i in range(len(class_labels)):
        id = np.where(y == i)[0][0]
        image = x[id]
        plt.subplot(2, 5, i+1)
        plt.axis('off')
        plt.imshow(image, cmap='gray')
        plt.title(class_labels[i])
        images.append(image)
        labels.append(class_labels[i])

    plt.tight_layout()
    plt.show()

#plot_data(X_train, y_train, class_labels)
# wandb.log({"Question 1 " : [wandb.Image(image , caption = f"Label: {label}") for image, label in zip(images, labels)]})
# wandb.finish()

def initializeParam(input_neurons, output_neurons, hidden_neurons, num_layers, activation, weight_init = "Xavier"):
  params = {}
  hidden_layer = [hidden_neurons] * num_layers
  layers = [input_neurons] + hidden_layer + [output_neurons]
  for i in range(1, len(layers)):
    if weight_init == "Xavier" or activation == "relu":
      params['W' + str(i)] = np.random.randn(layers[i], layers[i-1]) * np.sqrt(2/float(layers[i-1]))
    elif weight_init == "random":
      params['W' + str(i)] = np.random.randn(layers[i], layers[i-1])

    params['b' + str(i)] = np.zeros((layers[i], 1))
  return params

def sigmoid(x):
  return 1 / (1 + np.exp(-x))

def relu(x):
  return np.maximum(0, x)

def tanh(x):
  return np.tanh(x)

def identity(x):
  return x

def sigmoid_dev(x):
  return x * (1 - x)

def relu_dev(x):
  return np.where(x > 0, 1, 0)

def tanh_dev(x):
  return 1 - x**2

def identity_dev(x):
  return np.ones_like(x)

def softmax(x):
  x = np.clip(x, -700, 700)
  return np.exp(x) / np.sum(np.exp(x), axis = 0)

"""Q2] Implement a feedforward neural network which takes images from the fashion-mnist data as input and outputs a probability distribution over the 10 classes."""

def forward_prop(X, layers, params, activation):
  func = {}
  func['h0'] = X
  func['a0'] = X
  output = []
  for i in range(1, len(layers)):
    a = np.dot(params['W' + str(i)], func['h' + str(i-1)].T) + params['b' + str(i)]
    h = np.empty_like(a)
    if(i == len(layers) - 1) :
      h = softmax(a)
      output = h
    else :
      if activation == "sigmoid" :
        h = sigmoid(a)
      elif activation == "relu" :
        h = relu(a)
      elif activation == "tanh" :
        h = tanh(a)
      elif activation == "identity":
        h = identity(a)

    func['a' + str(i)] = a.T
    func['h' + str(i)] = h.T



  return output, func

"""Q3] Implement the backpropagation algorithm with support for the following optimisation functions"""

def back_prop(func, params, y, pred_y, L, loss_func, activation):
  m = y.size
  gradients = {}
  if loss_func == "cross_entropy" :
    dL_a = pred_y - y.T
  elif loss_func == "squared_loss" :
    dL_a = 2*(pred_y - y.T) * pred_y

  h_prev = func['h'+str(L)]
  act_dev = sigmoid_dev

  if activation == "sigmoid":
    act_dev = sigmoid_dev
  elif activation == "relu":
    act_dev = relu_dev
  elif activation == "tanh":
    act_dev = tanh_dev
  elif activation == "identity":
    act_dev = identity_dev

  for k in range(L, -1, -1):
    dL_W = np.dot(dL_a, h_prev)
    dL_b = np.sum(dL_a, axis=1, keepdims=True)

    dL_h = np.dot(params['W'+str(k+1)].T,  dL_a)
    dL_a = dL_h * act_dev(h_prev).T  #sigmoid (h_prev * (1- h_prev)).T

    gradients['dW' + str(k+1)] = dL_W
    gradients['db' + str(k+1)] = dL_b
    if(k-1 > -1):
      h_prev = func['h' + str(k-1)]
  return gradients

def sgd(params, gradients, eta, L, decay):
  for i in range(1, L):
    params['W' + str(i)] -=  eta * (gradients['dW' + str(i)] + decay * params['W' + str(i)])
    params['b' + str(i)] -=  eta * (gradients['db' + str(i)] + decay * params['b' + str(i)])

def mgd(params, gradients, history, max_epochs, momentum, eta, L, decay):
  for key in params.keys():
    if key not in history:
      history[key] = np.zeros_like(params[key])
    history[key] = momentum * history[key] + eta * (gradients['d' + key] + decay * params[key])
    params[key] -= history[key]

def nag(params, gradients, history, max_epochs, momentum, eta, L, decay):
  for i in range(1, L):
    history['W' + str(i)] = momentum * history['W' + str(i)] + eta * (gradients['dW' + str(i)] + decay * params['W' + str(i)])
    history['b' + str(i)] = momentum * history['b' + str(i)] + eta * (gradients['db' + str(i)] + decay * params['b' + str(i)])

    params['W' + str(i)] -= history['W' + str(i)]
    params['b' + str(i)] -= history['b' + str(i)]

def rmsprop(params, gradients, history, max_epochs, beta, eta, eps, L, decay):
  for key in params.keys():
    if key not in history:
      history[key] = np.zeros_like(params[key])

  for i in range(1, L):
    history['W' + str(i)] = beta * history['W' + str(i)] + (1 - beta) * gradients['dW' + str(i)]**2
    history['b' + str(i)] = beta * history['b' + str(i)] + (1 - beta) * gradients['db' + str(i)]**2

    params['W' + str(i)] -= (eta * gradients['dW' + str(i)]/(np.sqrt(history['W' + str(i)]) + eps)) + (eta * decay * params['W' + str(i)])
    params['b' + str(i)] -= (eta * gradients['db' + str(i)]/(np.sqrt(history['b' + str(i)]) + eps)) + (eta * decay * params['b' + str(i)])

def adam(params, gradients, history, moment, max_epochs, beta1, beta2, eta, eps, L, t, decay):
  for key in params.keys():
    if key not in history:
      history[key] = np.zeros_like(params[key])
    if key not in moment:
      moment[key] = np.zeros_like(params[key])

  for i in range(1, L):
    moment['W'+str(i)] = beta1 * moment['W'+str(i)] + (1-beta1) * gradients['dW'+str(i)]
    moment['b'+str(i)] = beta1 * moment['b'+str(i)] + (1-beta1) * gradients['db'+str(i)]
    history['W'+str(i)] = beta2 * history['W'+str(i)] + (1-beta2) * gradients['dW'+str(i)]**2
    history['b'+str(i)] = beta2 * history['b'+str(i)] + (1-beta2) * gradients['db'+str(i)]**2

    m_w_hat = moment['W'+str(i)]/(1 - (beta1 ** t))
    m_b_hat = moment['b'+str(i)]/(1 - (beta1 ** t))
    v_w_hat = history['W'+str(i)]/(1 - (beta2 ** t))
    v_b_hat = history['b'+str(i)]/(1 - (beta2 ** t))

    params['W'+str(i)] -= eta * m_w_hat/(np.sqrt(v_w_hat) + eps) + (eta * decay * params['W' + str(i)])
    params['b'+str(i)] -= eta * m_b_hat/(np.sqrt(v_b_hat) + eps) + (eta * decay * params['b' + str(i)])

def nadam(params, gradients, history, moment, max_epochs, beta1, beta2, eta, eps, L, t, decay):
  for key in params.keys():
    if key not in history:
      history[key] = np.zeros_like(params[key])
    if key not in moment:
      moment[key] = np.zeros_like(params[key])

  for i in range(1, L):
    moment['W'+str(i)] = beta1 * moment['W'+str(i)] + (1-beta1) * gradients['dW'+str(i)]
    moment['b'+str(i)] = beta1 * moment['b'+str(i)] + (1-beta1) * gradients['db'+str(i)]
    history['W'+str(i)] = beta2 * history['W'+str(i)] + (1-beta2) * gradients['dW'+str(i)]**2
    history['b'+str(i)] = beta2 * history['b'+str(i)] + (1-beta2) * gradients['db'+str(i)]**2

    m_w_hat = moment['W'+str(i)]/(1 - (beta1 ** t))
    m_b_hat = moment['b'+str(i)]/(1 - (beta1 ** t))
    v_w_hat = history['W'+str(i)]/(1 - (beta2 ** t))
    v_b_hat = history['b'+str(i)]/(1 - (beta2 ** t))

    params['W'+str(i)] -= (eta/np.sqrt(v_w_hat + eps)) * (beta1 * m_w_hat + (1-beta1) * gradients['dW'+str(i)] / (1 - beta1**t)) + (eta * decay * params['W' + str(i)])
    params['b'+str(i)] -= (eta/np.sqrt(v_b_hat + eps)) * (beta1 * m_b_hat + (1-beta1) * gradients['db'+str(i)] / (1 - beta1**t)) + (eta * decay * params['b' + str(i)])

def compute_loss(y, y_pred, loss, epsilon=1e-10):
  if loss == "squared_loss":
    return np.sum((y - y_pred)**2)/y.shape[0]
  elif loss == "cross_entropy":
    return -np.sum(np.multiply(y, np.log(y_pred + epsilon)))/y.shape[0]

def loss_dev(y, y_pred, loss):
  if loss == "cross_entropy":
    return -y/y_pred
  elif loss == "squared_loss":
    return y_pred - y

def get_prediction(output):
  return np.argmax(output, axis=0)

def get_accuracy(prediction, Y):
  return np.sum(prediction == Y)/Y.size

def cal_confusion(prediction, Y):
  count = 0
  confusion_mat = np.zeros((10, 10))
  for i in range(len(prediction)):
    confusion_mat[prediction[i]][Y[i]] += 1
  print(confusion_mat)

def lookahead(params, history, momentum):
  for key in params.keys():
    if key not in history:
      history[key] = np.zeros_like(params[key])
    else :
      history[key] = momentum * history[key]
    params[key] -= history[key]

def train(X_train, y_train, params, hidden_layer_size, no_layers, loss_func, activation_func, wt_init, learning_rate, epochs, batch_size, optimizer, weight_decay):
  input_neurons = X_train.shape[1]
  hidden_neurons = hidden_layer_size 
  num_layers = no_layers 
  hidden_layer = [hidden_neurons] * num_layers
  output_neurons = 10
  layers = [input_neurons] + hidden_layer + [output_neurons]
  y_train_onehot = np.eye(10)[y_train]
  y_val_onehot = np.eye(10)[y_val]
  loss_function = loss_func 
  activation = activation_func 
  weight_init = wt_init 

  eta = learning_rate 
  max_epochs = epochs 
  momentum = 0.5
  beta = 0.9
  beta1 = 0.9
  beta2 = 0.999
  eps = 0.000001
  batch_size= batch_size 
  optimizer = optimizer 
  decay = weight_decay 
  final_res = []
  history = {}
  moment = {}

  for epoch in range(max_epochs):
    for i in range(0, X_train.shape[0], batch_size):
      X_batch = X_train[i : i + batch_size]
      y_batch = y_train_onehot[i : i + batch_size]

      output, func = forward_prop(X_batch, layers, params, activation)
      if(optimizer == "nesterov"):
        lookahead(params, history, momentum)
      gradients = back_prop(func, params, y_batch, output, len(hidden_layer), loss_function, activation)
      match optimizer:
        case "sgd":
          sgd(params, gradients, eta, len(hidden_layer)+2, decay)
        case "momentum":
          mgd(params, gradients, history, max_epochs, momentum, eta, len(hidden_layer)+2, decay)
        case "nesterov":
          nag(params, gradients, history, max_epochs, momentum, eta, len(hidden_layer)+2, decay)
        case "rmsprop":
          rmsprop(params, gradients, history, max_epochs, beta, eta, eps, len(hidden_layer)+2, decay)
        case "adam":
          adam(params, gradients, history, moment, max_epochs, beta1, beta2, eta, eps, len(hidden_layer)+2, epoch+1, decay)
        case "nadam":
          nadam(params, gradients, history, moment, max_epochs, beta1, beta2, eta, eps, len(hidden_layer)+2, epoch+1, decay)
        case default:
          sgd(params, gradients, eta, len(hidden_layer)+2, decay)

    y_predict, func = forward_prop(X_train, layers, params, activation)
    train_loss = compute_loss(y_train_onehot, y_predict.T, loss_function)
    train_accuracy = get_accuracy(get_prediction(y_predict), y_train)

    y_val_predict, _ = forward_prop(X_val, layers, params, activation)
    val_loss = compute_loss(y_val_onehot, y_val_predict.T, loss_function)
    val_accuracy = get_accuracy(get_prediction(y_val_predict), y_val)

    #final_res = get_prediction(y_predict)
    print(f'Epoch {epoch+1}, Train Loss: {train_loss}, Train Accuracy: {train_accuracy}, Val Loss: {val_loss}, Val Accuracy: {val_accuracy}')
    wandb.log({'train_loss': train_loss, 'train_accuracy': train_accuracy, 'val_loss' : val_loss, 'val_accuracy' : val_accuracy, 'epoch': epoch+1})

def confusion_matrix(X_test, y_test, params, hidden_layer_size, hidden_layers, loss_func, activation_func, wt_init, learning_rate, epochs, batch, opt_func, wt_decay):
  #wandb.init(project = 'DL_Assignment_1', entity = 'cs23m009')
  input_neurons = X_test.shape[1]
  hidden_neurons = hidden_layer_size
  num_layers = hidden_layers
  hidden_layer = [hidden_neurons] * num_layers
  output_neurons = 10
  layers = [input_neurons] + hidden_layer + [output_neurons]
  y_test_onehot = np.eye(10)[y_test]
  loss_function = loss_func
  activation = activation_func
  weight_init = wt_init

  eta = learning_rate
  max_epochs = epochs
  momentum = 0.5
  beta = 0.9
  beta1 = 0.9
  beta2 = 0.999
  eps = 0.000001
  batch_size= batch
  optimizer = opt_func
  decay = wt_decay
  final_res = []
  history = {}
  moment = {}

  y_predict, func = forward_prop(X_test, layers, params, activation)
  test_loss = compute_loss(y_test_onehot, y_predict.T, loss_function)
  test_accuracy = get_accuracy(get_prediction(y_predict), y_test)
  cal_confusion(get_prediction(y_predict), y_test)
  #wandb.log({"Confusion Matrix " : wandb.sklearn.plot_confusion_matrix(y_test, get_prediction(y_predict), class_labels)})
  print(test_accuracy, test_loss)
  #wandb.finish()

wandb.init(project = args.wandb_project, entity = args.wandb_entity)
params = initializeParam(784, 10, args.hidden_size, args.num_layers, args.activation, args.weight_init)
train(X_train, y_train, params, args.hidden_size, args.num_layers, args.loss, args.activation, args.weight_init, args.learning_rate, args.epochs, args.batch_size, args.optimizer, args.weight_decay)
#confusion_matrix(X_test, y_test, params, args.hidden_size, args.num_layers, args.loss, args.activation, args.weight_init, args.learning_rate, args.epochs, args.batch_size, args.optimizer, args.weight_decay)
wandb.finish()
