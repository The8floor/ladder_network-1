# ladder_network

## Description
Python package implementing  the Ladder Network presented by Valpola et al. in the paper [Semi-Supervised Learning with Ladder Networks](https://arxiv.org/abs/1507.02672). The objective of the package is to allow users to train state-of-the-art semi-supervised neural architectures with minimal effort. The model implemented is the 'vanilla' ladder network, namely the one built on top of the multi-layer perceptron.

## Requirements
TensorFlow >= 1.0.0
## Installation
`pip install ladder_network`
## Usage

### Training and testing on MNIST data
```python
from ladder_network.ladder import LadderNet
from ladder_network.helper import get_MNIST_data

# number of labeled data points you wish to use from the MNIST data set
n_labelled = 100
# batch size for unlabeled data points
# Note: the total batch size will be n_labelled + batch_size = 200 in this case.
#  If the number of labeled data is high, the total batch size will be high as well and this may cause memory issues. This will be fixed in later releases.
batch_size = 100

X_l, X_u, y_l, y_u = get_MNIST_data(n_labelled)
# X_l: labeled data, numpy ndarray with shape (100, 784)
# X_u: unlabeled data, numpy ndarray with shape (64900, 784)
# y_l, y_u: corresponding labels, one-hot encoded
# Note that y_u contains labels that are used for the sole purpose of testing.
# In general, y_u does not exist in semi-supervised learning (else it would be supervised learning)


param = {'batch_size':batch_size, 'num_epochs':20}
model = LadderNet(**param)
model.train(X_l, X_u, y_l, y_u)

model.predict

```
### Validation on MNIST data

## Source
* [Semi-Supervised Learning with Ladder Networks](https://arxiv.org/abs/1507.02672)
* [Original Theano implementation by the paper's authors](https://github.com/CuriousAI/ladder)
* [Tensorflow implementation](https://github.com/rinuboney/ladder)

