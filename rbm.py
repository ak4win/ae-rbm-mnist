"""Implementation of a Restricted Boltzmann Machine"""

import tensorflow as tf
from tensorflow.linalg import matmul
import tensorflow_probability as tfp

from rbm_utils import forward_pass, backward_pass, loss

class RBM():
    """Implementation of a Restricted Boltzmann Machine
    
    Note that this implementation does not use Pytorch's nn.Module
    because we are updating the weights ourselves

    """
    def __init__(self, visible_dim, hidden_dim, gaussian_hidden_distribution=False,
     pretrained_weights=None, pretrained_v_bias=None, pretrained_h_bias=None):
        """Initialize a Restricted Boltzmann Machine

        Parameters
        ----------
        visible_dim: int
            number of dimensions in visible (input) layer
        hidden_dim: int
            number of dimensions in hidden layer
        gaussian_hidden_distribution: bool
            whether to use a Gaussian distribution for the values of the hidden dimension instead of a Bernoulli
    
        """
        self.visible_dim = visible_dim
        self.hidden_dim = hidden_dim
        self.gaussian_hidden_distribution = gaussian_hidden_distribution

        # intialize parameters
        if pretrained_weights is not None: 
            self.W = tf.Variable(pretrained_weights, dtype=tf.float32)
        else:
            x = tf.random.truncated_normal((visible_dim, hidden_dim), mean=0.0, stddev=0.05, seed=None)
            self.W = tf.Variable(x)
        self.h_bias = tf.Variable(pretrained_h_bias, dtype=tf.float32) if pretrained_h_bias is not None else tf.zeros(hidden_dim)  # v --> h
        self.v_bias = tf.Variable(pretrained_v_bias, dtype=tf.float32) if pretrained_v_bias is not None else tf.zeros(visible_dim)  # h --> v
        
        # parameters for learning with momentum
        self.W_momentum = tf.zeros((visible_dim, hidden_dim))
        self.h_bias_momentum = tf.zeros(hidden_dim)  # v --> h
        self.v_bias_momentum = tf.zeros(visible_dim)  # h --> v

    def predict(self, v):
        """Make a full forward and backward pass to 'predict'"""
        # img = tf.constant(img.T, dtype=tf.float32)
        _, sequence_predictions = self.sample_h(v)
        return self.sample_v(sequence_predictions)

    def sample_h(self, v):
        """Get sample hidden values and activation probabilities

        Parameters
        ----------
        v: Tensor
            tensor of input from visible layer

        """
        activation = forward_pass(v, self.W) + self.h_bias
        if self.gaussian_hidden_distribution:
            return activation, tf.random.normal(activation.shape, mean=activation)
        else:
            p = tf.sigmoid(activation)
            samples = tfp.distributions.Bernoulli(probs=p).sample(sample_shape=())
            return p, tf.cast(samples, tf.float32)

    def sample_v(self, h):
        """Get visible activation probabilities

        Parameters
        ----------
        h: Tensor
            tensor of input from hidden

        """
        activations = backward_pass(h, self.W) 
        activation = activations.read(0) + self.v_bias
        p = tf.sigmoid(activation)
        return p

    def update_weights(self, v0, vk, ph0, phk, lr, momentum_coef, weight_decay, batch_size):
        """Learning step: update parameters 

        Uses contrastive divergence algorithm as described in

        Parameters
        ----------
        v0: Tensor
            initial visible state
        vk: Tensor
            final visible state
        ph0: Tensor
            hidden activation probabilities for v0
        phk: Tensor
            hidden activation probabilities for vk
        lr: float
            learning rate
        momentum_coef: float
            coefficient to use for momentum
        weight_decay: float
            coefficient to use for weight decay
        batch_size: int
            size of each batch

        """
        self.W_momentum *= momentum_coef
        self.W_momentum = tf.cast(self.W_momentum, tf.float32)
        self.W_momentum = self.W_momentum + matmul(tf.transpose(v0), ph0) - matmul(tf.transpose(vk), phk)
        self.h_bias_momentum *= momentum_coef
        self.h_bias_momentum += tf.math.reduce_sum((ph0 - phk), 0)

        self.v_bias_momentum *= momentum_coef
        self.v_bias_momentum += tf.math.reduce_sum((v0 - vk), 0)

        self.W = self.W + lr*self.W_momentum/batch_size
        self.h_bias = self.h_bias + lr*self.h_bias_momentum/batch_size
        self.v_bias = self.v_bias + lr*self.v_bias_momentum/batch_size

        self.W -= self.W * weight_decay # L2 weight decay

def train_rbm(current_x_train, visible_dim, hidden_dim, k=1, num_epochs=10, batch_size=10, lr=0.1, use_gaussian=False, pretrained_rbm=None):
    rbm = RBM(visible_dim, hidden_dim, gaussian_hidden_distribution=use_gaussian)
    if pretrained_rbm is not None: rbm = pretrained_rbm

    assert current_x_train.shape[0] >= batch_size

    num_batches = int(current_x_train.shape[0] / batch_size)
    x_train = current_x_train[:num_batches*batch_size]
    
    losses = []
    for epoch in range(num_epochs):
        train_loss = 0
        for i in range(num_batches):
            current_batch = x_train[i*batch_size:(i+1)*batch_size]

            sample_data = current_batch
            v0, pvk = sample_data, sample_data

            # Gibbs sampling: sample from the distribution of the visible layer -> the data
            for i in range(k):
                _, hk = rbm.sample_h(pvk)
                pvk = rbm.sample_v(hk)
            
            # compute ph0 and phk for updating weights
            ph0, _ = rbm.sample_h(v0)
            phk, _ = rbm.sample_h(pvk)
            
            # update weights
            rbm.update_weights(v0, pvk, ph0, phk, lr, 
                                # momentum_coef=0.5 if epoch < 5 else 0.9,
                                momentum_coef = 0.1,
                                weight_decay=2e-4, 
                                batch_size=1)

                # track loss
            train_loss += loss(v0, pvk)
        if epoch%50==0: losses.append(train_loss)
        if epoch == 1: print(f"epoch {epoch}: {train_loss}")
        if epoch == 10: print(f"epoch {epoch}: {train_loss}")
        if epoch%100==0: print(f"epoch {epoch}: {train_loss}")
    print(f"epoch {epoch}: {train_loss}")
    return rbm, v0, pvk, losses