import numpy as np

from layers import (
    FullyConnectedLayer, ReLULayer,
    ConvolutionalLayer, MaxPoolingLayer, Flattener,
    softmax_with_cross_entropy, l2_regularization,
    softmax
    )


class ConvNet:
    """
    Implements a very simple conv net

    Input -> Conv[3x3] -> Relu -> Maxpool[4x4] ->
    Conv[3x3] -> Relu -> MaxPool[4x4] ->
    Flatten -> FC -> Softmax
    """
    def __init__(self, input_shape, n_output_classes, conv1_channels, conv2_channels):
        """
        Initializes the neural network

        Arguments:
        input_shape, tuple of 3 ints - image_width, image_height, n_channels
                                         Will be equal to (32, 32, 3)
        n_output_classes, int - number of classes to predict
        conv1_channels, int - number of filters in the 1st conv layer
        conv2_channels, int - number of filters in the 2nd conv layer
        """
        # TODO Create necessary layers
        self.out_classes = n_output_classes
        image_width, image_height, in_channels = input_shape

        self.Conv1 = ConvolutionalLayer(in_channels, conv1_channels, 3, 1)
        self.ReLU1 = ReLULayer()
        self.MaxPool1 = MaxPoolingLayer(4, 4)
        self.Conv2 = ConvolutionalLayer(conv1_channels, conv2_channels, 3, 1)
        self.ReLU2 = ReLULayer()
        self.MaxPool2 = MaxPoolingLayer(4, 4)
        self.Flatten = Flattener()
        self.FC = FullyConnectedLayer(4 * conv2_channels, n_output_classes)

    def compute_loss_and_gradients(self, X, y):
        """
        Computes total loss and updates parameter gradients
        on a batch of training examples

        Arguments:
        X, np array (batch_size, height, width, input_features) - input data
        y, np array of int (batch_size) - classes
        """
        # Before running forward and backward pass through the model,
        # clear parameter gradients aggregated from the previous pass

        # TODO Compute loss and fill param gradients
        # Don't worry about implementing L2 regularization, we will not
        # need it in this assignment
        for param in self.params().values():
            param.grad = np.zeros_like(param.value)

        preds = self.Conv1.forward(X)
        preds = self.ReLU1.forward(preds)
        preds = self.MaxPool1.forward(preds)
        preds = self.Conv2.forward(preds)
        preds = self.ReLU2.forward(preds)
        preds = self.MaxPool2.forward(preds)
        preds = self.Flatten.forward(preds)
        preds = self.FC.forward(preds)

        loss, grad = softmax_with_cross_entropy(preds, y)

        grad = self.FC.backward(grad)
        grad = self.Flatten.backward(grad)
        grad = self.MaxPool2.backward(grad)
        grad = self.ReLU2.backward(grad)
        grad = self.Conv2.backward(grad)
        grad = self.MaxPool1.backward(grad)
        grad = self.ReLU1.backward(grad)
        grad = self.Conv1.backward(grad)

        return loss

    def predict(self, X):
        # You can probably copy the code from previous assignment
        preds = self.Conv1.forward(X)
        preds = self.ReLU1.forward(preds)
        preds = self.MaxPool1.forward(preds)
        preds = self.Conv2.forward(preds)
        preds = self.ReLU2.forward(preds)
        preds = self.MaxPool2.forward(preds)
        preds = self.Flatten.forward(preds)
        preds = self.FC.forward(preds)

        probs = softmax(preds)
        return np.argmax(probs, axis=1)

    def params(self):
        # TODO: Aggregate all the params from all the layers
        # which have parameters
        return {'Conv1.W': self.Conv1.W, 'Conv1.B': self.Conv1.B, 
                'Conv2.W': self.Conv2.W, 'Conv2.B': self.Conv2.B,
                'FC.W': self.FC.W, 'FC.B': self.FC.B}
