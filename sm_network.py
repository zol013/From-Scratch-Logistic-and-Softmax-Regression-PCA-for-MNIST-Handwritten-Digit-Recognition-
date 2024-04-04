import numpy as np
import data
import time
import tqdm

"""
NOTE
----
Start by implementing your methods in non-vectorized format - use loops and other basic programming constructs.
Once you're sure everything works, use NumPy's vector operations (dot products, etc.) to speed up your network.
"""

def sigmoid(a):
    """
    Compute the sigmoid function.

    f(x) = 1 / (1 + e ^ (-x))

    Parameters
    ----------
    a
        The internal value while a pattern goes through the network
    Returns
    -------
    float
       Value after applying sigmoid (z from the slides).
    """
    pass

def softmax(a):
    """
    Compute the softmax function.

    f(x) = (e^x) / Σ (e^x)

    Parameters
    ----------
    a
        The internal value while a pattern goes through the network
    Returns
    -------
    float
       Value after applying softmax (z from the slides).
    """
    x = np.exp(a)
    return x/np.sum(x, axis = 1, keepdims = True)

def binary_cross_entropy(y, t):
    """
    Compute binary cross entropy.

    L(x) = t*ln(y) + (1-t)*ln(1-y)

    Parameters
    ----------
    y
        The network's predictions
    t
        The corresponding targets
    Returns
    -------
    float 
        binary cross entropy loss value according to above definition
    """
    pass

def multiclass_cross_entropy(y, t):
    """
    Compute multiclass cross entropy.

    L(x) = - Σ (t*ln(y))

    Parameters
    ----------
    y
        The network's predictions
    t
        The corresponding targets
    Returns
    -------
    float 
        multiclass cross entropy loss value according to above definition
    """
    
    
    entropy = np.multiply(np.log(y), t)
    
    return - np.sum(entropy)

class Network:
    def __init__(self, hyperparameters, activation, loss, out_dim):
        """
        Perform required setup for the network.

        Initialize the weight matrix, set the activation function, save hyperparameters.

        You may want to create arrays to save the loss values during training.

        Parameters
        ----------
        hyperparameters
            A Namespace object from `argparse` containing the hyperparameters
        activation
            The non-linear activation function to use for the network
        loss
            The loss function to use while training and testing
        """
        self.hyperparameters = hyperparameters
        self.activation = activation
        self.loss = loss

        self.weights = np.zeros((self.hyperparameters.p + 1, out_dim))

    def forward(self, X):
        """
        Apply the model to the given patterns

        Use `self.weights` and `self.activation` to compute the network's output

        f(x) = σ(w*x)
            where
                σ = non-linear activation function
                w = weight matrix

        Make sure you are using matrix multiplication when you vectorize your code!

        Parameters
        ----------
        X
            Patterns to create outputs for
        """
        return self.activation(X.dot(self.weights))
        

    def __call__(self, X):
        return self.forward(X)
    
    
    def onehot_decode(self, y):
        y_decoded = np.argmax(y, axis = 1)
    
        return y_decoded

    def train(self, minibatch):
        """
        Train the network on the given minibatch

        Use `self.weights` and `self.activation` to compute the network's output
        Use `self.loss` and the gradient defined in the slides to update the network.

        Parameters
        ----------
        minibatch
            The minibatch to iterate over

        Returns
        -------
        tuple containing:
            average loss over minibatch
            accuracy over minibatch
        """
        train_images, train_labels = minibatch
        B = len(train_labels)
        sm_scores = self.activation(train_images @ self.weights)
        avarage_loss = self.loss(sm_scores, train_labels) / B
        predict_labels = np.argmax(sm_scores, axis = 1)
        decoded_train_labels = self.onehot_decode(train_labels)
        accuracy = sum(predict_labels == decoded_train_labels)/B
        gradient = train_images.T @ (train_labels - sm_scores)
        self.weights += self.hyperparameters.learningrate * gradient/ B 
        
        return avarage_loss, accuracy

    def test(self, minibatch):
        """
        Test the network on the given minibatch

        Use `self.weights` and `self.activation` to compute the network's output
        Use `self.loss` to compute the loss.
        Do NOT update the weights in this method!

        Parameters
        ----------
        minibatch
            The minibatch to iterate over

        Returns
        -------
            tuple containing:
                average loss over minibatch
                accuracy over minibatch
        """
        test_images, test_labels = minibatch
        B = len(test_labels)
        sm_scores = self.activation(test_images @ self.weights)
        decoded_test_labels = self.onehot_decode(test_labels)
        avarage_loss = self.loss(sm_scores, test_labels) / B
        predict_labels = np.argmax(sm_scores, axis = 1)
     
        accuracy = sum(predict_labels == decoded_test_labels)/B
        
        return (avarage_loss,accuracy)