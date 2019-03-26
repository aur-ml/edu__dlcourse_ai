import numpy as np

from layers import FullyConnectedLayer, ReLULayer, softmax_with_cross_entropy, l2_regularization, softmax


class TwoLayerNet:
    """ Neural network with two fully connected layers """

    def __init__(self, n_input, n_output, hidden_layer_size, reg):
        """
        Initializes the neural network

        Arguments:
        n_input, int - dimension of the model input
        n_output, int - number of classes to predict
        hidden_layer_size, int - number of neurons in the hidden layer
        reg, float - L2 regularization strength
        """
        self.reg = reg
        # TODO Create necessary layers
        self.first = FullyConnectedLayer(n_input, hidden_layer_size)
        self.relu = ReLULayer()
        self.second = FullyConnectedLayer(hidden_layer_size, n_output)

    def compute_loss_and_gradients(self, X, y):
        """
        Computes total loss and updates parameter gradients
        on a batch of training examples

        Arguments:
        X, np array (batch_size, input_features) - input data
        y, np array of int (batch_size) - classes
        """
        # Before running forward and backward pass through the model,
        # clear parameter gradients aggregated from the previous pass
        # TODO Set parameter gradient to zeros
        # Hint: using self.params() might be useful!
        for param in self.params().values():
            param.grad = np.zeros_like(param.value)
        
        # TODO Compute loss and fill param gradients
        # by running forward and backward passes through the model
        preds = self.first.forward(X)
        preds = self.relu.forward(preds)
        preds = self.second.forward(preds)

        loss, grad = softmax_with_cross_entropy(preds, y)

        grad = self.second.backward(grad)
        grad = self.relu.backward(grad)
        grad = self.first.backward(grad)

        # After that, implement l2 regularization on all params
        # Hint: self.params() is useful again!
        for param in self.params().values():
            loss_l2, grad_l2 = l2_regularization(param.value, self.reg)
            loss += loss_l2
            param.grad += grad_l2

        return loss

    def predict(self, X):
        """
        Produces classifier predictions on the set

        Arguments:
          X, np array (test_samples, num_features)

        Returns:
          y_pred, np.array of int (test_samples)
        """
        # TODO: Implement predict
        # Hint: some of the code of the compute_loss_and_gradients
        # can be reused
        preds = self.first.forward(X)
        preds = self.relu.forward(preds)
        preds = self.second.forward(preds)

        probs = softmax(preds)
        return np.argmax(probs, axis=1)

    def params(self):
        # TODO Implement aggregating all of the params

        return {'first.W': self.first.W, 'first.B': self.first.B, 
                'second.W': self.second.W, 'second.B': self.second.B}
