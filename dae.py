"""Implementation of a Deep Autoencoder"""

import tensorflow as tf


class DAE():
    """A Deep Autoencoder that takes a list of RBMs as input"""

    def __init__(self, models):
        """Create a deep autoencoder based on a list of RBM models

        Parameters
        ----------
        models: list[RBM]
            a list of RBM models to use for autoencoding
        """

        # extract weights from each model
        encoders = []
        encoder_biases = []
        decoders = []
        decoder_biases = []
        for model in models:
            encoders.append(tf.Variable(model.W))
            encoder_biases.append(tf.Variable(model.h_bias))
            decoders.append(tf.Variable(model.W))
            decoder_biases.append(tf.Variable(model.v_bias))

        # build encoders and decoders
        self.encoders = encoders
        self.encoder_biases = encoder_biases
        self.decoders = [x for x in reversed(encoders)]
        self.decoder_biases = [x for x in reversed(decoder_biases)]
        # self.encoders = tf.TensorArray(tf.float32, size=len(encoders))
        # print(len(encoders))
        # for i, model in enumerate(encoders): self.encoders = self.encoders.write(i, model)

        # self.encoder_biases = tf.TensorArray(tf.float32, size=len(encoders))
        # for i, bias in enumerate(encoder_biases): self.encoder_biases = self.encoder_biases.write(i, bias)

        # self.decoders = tf.TensorArray(tf.float32, size=len(decoders))
        # for i, model in enumerate(reversed(decoders)): self.decoders = self.decoders.write(i, model)

        # self.decoder_biases = tf.TensorArray(tf.float32, size=len(decoders))
        # for i, bias in enumerate(reversed(decoder_biases)): self.decoder_biases = self.decoder_biases.write(i, bias)

    @tf.function
    def predict(self, v):
        """Forward step

        Parameters
        ----------
        v: Tensor
            input tensor

        Returns
        -------
        Tensor
            a reconstruction of v from the autoencoder

        """
        # encode
        p_h = self.encode(v)

        # decode
        p_v = self.decode(p_h)

        return p_v

    @tf.function
    def encode(self, v):  # for visualization, encode without sigmoid
        """Encode input

        Parameters
        ----------
        v: Tensor
            visible input tensor

        Returns
        -------
        Tensor
            the activations of the last layer

        """
        p_v = v
        activation = v

        # length = self.encoders.size()
        length = len(self.encoders)
        for i in range(length):
            W = self.encoders[i]
            h_bias = self.encoder_biases[i]
            activation = tf.linalg.matmul(p_v, W) + h_bias
            p_v = tf.sigmoid(activation)

        # for the last layer, we want to return the activation directly rather than the sigmoid
        return activation

    @tf.function
    def decode(self, h):
        """Encode hidden layer

        Parameters
        ----------
        h: Tensor
            activations from last hidden layer

        Returns
        -------
        Tensor
            reconstruction of original input based on h

        """
        p_h = h
        # length = self.decoders.size()
        length = len(self.encoders) # can't use decoders bc reversed
        for i in range(length):
            W = self.decoders[i]
            v_bias = self.decoder_biases[i]
            activation = tf.linalg.matmul(p_h, tf.transpose(W)) + v_bias
            p_h = tf.sigmoid(activation)
        return p_h