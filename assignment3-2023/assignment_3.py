# %% [markdown]
# # PART I: Theory Questions
# 
# 1) Consider the convolutional neural network defined by the layer|s in the left column below. Fill in the shape of the output volume and the number of parameters at each layer. You can write the shapes in the numpy format (e.g. (64,64,3))
# 
# | Layer | Output Volume Shape | Number of parameters |
# |---|---|---|
# | Input | (127x127x4) | 0 |
# | CONV3-10 | (125x125x10) | 3x3x10x4+10=370 |
# | POOL-3 | (41x41x10) | 0 |
# | CONV3-10 | (39x39x10) | 3x3x10x10+10=910 |
# | POOL-2 | (19x19x10) | 0 |
# | FC-20 | (20) | 20x19x19x10+20=72220 |
# | FC-10 | (10) | 20x10+10=210 |
# 
# 
# 2) Consider the simple neuron structure below:
# 
# ![Question 2 Image](q2.png)
# 
# 
# Assume that the weights for the neuron are w1 = 3, w2 = -5, and w3 = 2 with activation function below:
# $$v(x) =
# \begin{cases}
# 1, & \text{if } x > 0 \\
# 0, & \text{otherwise}
# \end{cases}$$
# 
# find the output y values for the input patterns below:
# 
# | INPUT | I1 | I2 | I3 | I4 |
# |---|---|---|---|---|
# | x1 | 1 | 0 | 1 | 1 |
# | x2 | 0 | 1 | 0 | 1 |
# | x3 | 0 | 1 | 1 | 1 |
# 
# | Output | I1 | I2 | I3 | I4 |
# |---|---|---|---|---|
# | Output | v(3x1-5x0+2x0)=v(3)=1 | v(3x0-5x1+2x1)=v(-3)=0 | v(3x1-5x0+2x1)=v(5)=1 | v(3x1-5x1+2x1)=v(0)=0|
# 
# 3) Consider the multi-layer neural network below:
# 
# ![Question 3 Image](q3.png)
# 
# - Find how many weight variables the network has in total (Ignore bias values). Show your calculations.
# 
#     - Input-1st Hidden Layer: 3*5 = 15
#     - 1st Hidden Layer - 2nd Hidden Layer: 5*3 = 15
#     - 2nd Hidden Layer - Output Layer: 3*2 = 6
# 
#     - In total 36
# 
# - Find how many weight variables the network has in total if the network is considered as fully connected (Ignore bias values). Show your calculations 
#     - The same as the first part, as Neural Network is Fully Connected.
#     
#     - Input-1st Hidden Layer: 3*5 = 15
#     - 1st Hidden Layer - 2nd Hidden Layer: 5*3 = 15
#     - 2nd Hidden Layer - Output Layer: 3*2 = 6
# 
#     - In total 36
# 
# - State the dependency information for nodes given number values, which are about which node takes information from which previous node. State also these dependencies for both forward and back-propagation streams.
# 
# Lets assume;
# $x_{ij}$ represents a node
# 
# i: represents layer id, from 0 to number of layers
# 
# j: represents node id.
# 
# ##### Forward Propagation:
# As it is a fully connected layer, we know that every node in $x_{i+1}$, is dependent to previous layer's ($x_{i}$) nodes and the weights $w_{i/i+1}$.
# 
# we can formulate it as this:
# 
# #### $n_{i+1}$ = $w_{i/i+1}.T$ @ $n_{i}$
# 
# T: takes the transpose of matrix given,
# @: Matrix Multiplication.
# 
# As the formula above, every node values are dependent to previous node values.
# 
# ##### Backward Propagation:
# On backward propagation we calculate loss then, apply derivatives to each layer, from output layer to input layer, give derivative information using Chain Rule.
# 
# The formula looks like this:
# #### $ \frac{d_{L}}{d_{w_{i/i+1}}} = \frac{d_{L}}{d_{w_{i-1/i}}} . \frac{d_{w_{i/i+1}}}{d_{w_{i-1/i}}} $
# 
# As every weight is dependent to loss information coming from the next layer, and nodes are dependent to weight values, every node is dependent to next nodes also in backward pass context.
# 
# 

# %% [markdown]
# # Part 2
# 
# ## Multi Layer Neural Network
# 
# - In this part, we are expected to classify mel spectogram image dataset with respect to age information, but as data have gender and accent information also, let's classify as multilabel classification.

# %%
'''
Importing important libraries
_____________________________
numpy:matrix operations
pillow:to read images
pandas: csv and excel operations
os: folder operations
sklearn: to One hot encode and calculate accuracy
tqdm: to see progress bar.
logging: to log the output values
'''
import numpy as np
from PIL import Image
import pandas as pd
import os
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import accuracy_score
from tqdm import tqdm
import logging

# %%
logging.basicConfig(filename='logfile.txt', level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s\n')

# %%
# Reading CSV files
train_df = pd.read_csv('dataset_a3/voice_dataset/train_data.csv')
test_df = pd.read_csv('dataset_a3/voice_dataset/test_data.csv')

# %%
train_df.head()

# %%
test_df.head()

# %%
def read_images(paths, ratio):
    image = Image.open(paths[0])
    images = np.zeros((len(paths), image.size[1] // ratio, image.size[0] // ratio, 4))
    for i, path in enumerate(paths):
        image = Image.open(path)
        image = image.resize((image.size[0] // ratio, image.size[1] // ratio))
        image = np.array(image) / 255
        images[i] = image
    return images

def encode(train, test):
    # Reshape to a 2D array (column vector)
    train = np.array(train).reshape(-1, 1)
    test = np.array(test).reshape(-1, 1)

    # Initialize the OneHotEncoder
    encoder = OneHotEncoder()

    # Fit and transform the training data
    train_encoded = encoder.fit_transform(train).toarray()

    # Transform the test data
    test_encoded = encoder.transform(test).toarray()

    return encoder, train_encoded, test_encoded

# %%
train_paths = train_df['filename']
test_paths = test_df['filename']
age_encoder, train_age, test_age = encode(train_df['age'], test_df['age'])
gender_encoder, train_gender, test_gender = encode(train_df['gender'], test_df['gender'])
accent_encoder, train_accent, test_accent = encode(train_df['accent'], test_df['accent'])

# %%
class Dataset():
    '''
        Dataset class to replicate Pytorch Datasets
        ____________________________________________

        Parameters
        ____________

        image_dir : str
            the directory contains images

        image_paths : list[str]
            the relative paths to image_dir

        age : np.array
            Age array

        gender : np.array
            Gender array

        accent : np.array
            Accent array

    '''
    def __init__(self, image_dir: str, 
                 image_paths: list[str], 
                 age: np.array, 
                 gender: np.array, 
                 accent: np.array):
        self.image_dir = image_dir
        self.image_paths = list(map(lambda p: os.path.join(image_dir, p), image_paths))
        self.age = age
        self.gender = gender
        self.accent = accent

    def __getitem__(self, indices):
        return read_images(self.image_paths[indices], 10), \
                self.age[indices], \
                self.gender[indices], \
                self.accent[indices]
    
    def __len__(self):
        return len(self.image_paths)
    
    def shuffle(self, random_seed=None):
        if random_seed is not None:
            np.random.seed(random_seed)
        indices = np.arange(len(self.image_paths))
        np.random.shuffle(indices)

        self.image_paths = [self.image_paths[i] for i in indices]
        self.age = self.age[indices]
        self.gender = self.gender[indices]
        self.accent = self.accent[indices]
    
    def split_dataset(self, train_ratio=0.8, random_seed=None):
        if random_seed is not None:
            np.random.seed(random_seed)

        num_samples = len(self.image_paths)
        indices = np.arange(num_samples)
        np.random.shuffle(indices)

        train_size = int(train_ratio * num_samples)

        train_indices = indices[:train_size]
        val_indices = indices[train_size:]

        train_dataset = Dataset(
            image_dir='',
            image_paths=[self.image_paths[i] for i in train_indices],
            age=self.age[train_indices],
            gender=self.gender[train_indices],
            accent=self.accent[train_indices]
        )

        val_dataset = Dataset(
            image_dir='',
            image_paths=[self.image_paths[i] for i in val_indices],
            age=self.age[val_indices],
            gender=self.gender[val_indices],
            accent=self.accent[val_indices]
        )

        return train_dataset, val_dataset
    
class DataLoader():
    '''
        The dataLoader class to replicate Pytorch Dataloaders
        _____________________________________________________

        Parameters
        __________
        
        batch_size : int
            Batch Size

        dataset : Dataset
            Dataset
    '''
    def __init__(self, batch_size: int, dataset: Dataset) -> None:
        self.batch_size = batch_size
        self.dataset = dataset
        self.idx = 0

    def __iter__(self):
        return self
    
    def __next__(self):
        if self.idx >= len(self.dataset):
            self.idx = 0
            raise StopIteration

        batch = self.dataset[self.idx:self.idx+self.batch_size]

        self.idx += self.batch_size

        return batch
    
    def __len__(self):
        return len(self.dataset) // self.batch_size
    
    def __getitem__(self, indices):
        return self.dataset[indices]

# %%
train_dataset = Dataset(image_dir='dataset_a3/voice_dataset/train/', 
                        image_paths=train_paths, 
                        age=train_age,
                        gender=train_gender,
                        accent=train_accent)

test_dataset = Dataset(image_dir='dataset_a3/voice_dataset/test/', 
                        image_paths=test_paths, 
                        age=test_age,
                        gender=test_gender,
                        accent=test_accent)

train_dataset, val_dataset = train_dataset.split_dataset(0.8, 42)
train_dataset.shuffle(42)
val_dataset.shuffle(42)
test_dataset.shuffle(42)

# %%
from scipy.sparse import issparse
def softmax(x):
    '''
        Softmax activation function
        ___________________________
        Parameters
        __________
        x : np.array
            mini batch array will be activated.

        Formula
        _________
        x = e**(x) / sum(e**(x))
    '''
    e_x = np.exp(x - np.max(x, axis=-1, keepdims=True))
    return e_x / np.sum(e_x, axis=-1, keepdims=True)

def softmax_derivative(x):
    '''
        Softmax activation function derivative
        ___________________________
        Parameters
        __________
        x : np.array
            mini batch array will be derived.
    '''
    s = softmax(x)
    return s * (1 - s)

def negative_log_likelihood_loss(y_true, y_pred):
    '''
        Negative Log Likelihood (NLL) Loss calculation Function
        _______________________________________________________

        Parameters
        _______________
        y_true : np.array
            True Labels

        y_pred : np.array
            Predicted Labels

        Formula
        ____________
        loss = - Sigma(n: 1 to N), Sigma(k: 1 to K) y_true[k][n] * log(y_pred[k][n])
    '''

    # Epsilon value to avoid log(0)
    epsilon = 1e-10

    # Clip values utilizing epsilon
    y_pred = np.clip(y_pred, epsilon, 1 - epsilon)
    
    if issparse(y_true):
        y_true = y_true.toarray()

    loss = -np.sum(np.multiply(y_true, np.log(y_pred + epsilon))) / len(y_true)
    return loss

def nll_loss_derivative(y_true, y_pred, epsilon=1e-10):
    '''
        Negative Log Likelihood (NLL) Loss derivative calculation Function

        Parameters
        _______________
        y_true : np.array
            True Labels

        y_pred : np.array
            Predicted Labels

        epsilon : float, optional
            A small value to avoid division by zero, by default 1e-10
    '''
    y_pred = np.clip(y_pred, epsilon, 1 - epsilon)
    return -(y_true / y_pred)

def relu(x):
    '''
        ReLU activation function, it is easy as below 0 values will be mapped to 0 and upper 0 values will be keeped the same.
        ____________

        Parameters
        __________
        x : np.array
            mini batch will be activated
    '''
    return np.maximum(0, x)


def relu_derivative(x):
    '''
        Derivative of ReLU
        __________________
        it is simple as relu contains x or 0
        derivative of x is 1
        derivative of 0 is 0

        Parameters
        __________
        x : np.array
            mini batch, derivative will be taken from
    '''
    return np.where(x > 0, 1, 0)

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def sigmoid_derivative(x):
    '''
    Derivative of Sigmoid
    _____________________

    It is similar to softmax derivative.

    Parameters
    __________
    x : np.array
        mini batch, derivative will be taken from
    '''
    sig = sigmoid(x)
    return sig * (1 - sig)

def hardmax(x):
  """Hardmax activation for prediction."""
  return np.argmax(x, axis=1)

# %%
class Layer:
    '''
    The Base Layer
    '''
    def __init__(self) -> None:
        self.input = None
        self.output = None

    def forward(self, input):
        # TODO: Return output
        pass

    def backward(self, output_gradient, learning_rate):
        # TODO: Update parameters and return input gradient.
        pass


class FullyConnectedLayer(Layer):
    '''
    Fully Connected (Dense) Layer
    
    Parameters:
        input_size: Input size of Layer
        output_Size: Output Size of Layer

    Arguments:
        weights: Weight Array
        bias: Bias Array
        Input: Stores input came to backpropagate.
    '''
    def __init__(self, input_size, output_size) -> None:
        super().__init__()
        self.weights = np.random.randn(input_size, output_size) * np.sqrt(2 / (input_size + output_size))
        self.bias = np.zeros((1, output_size))
        self.input = None
        
    def forward(self, x):
        self.input = x
        return x @ self.weights + self.bias
    
    def backward(self, output_error, learning_rate):
        weights_delta = self.input.T @ output_error
        bias_delta = np.sum(output_error, axis=0)

        # Save the updated error in a temporary variable
        updated_error = output_error @ self.weights.T

        # Update weights
        self.weights -= learning_rate * weights_delta

        # Update bias
        self.bias -= learning_rate * bias_delta.reshape(1, -1)

        return updated_error
    
    def __call__(self, x):
        return self.forward(x)
    
class ActivationLayer(Layer):
    '''
    Activation Layer

    Parameters:
        activation_function: ReLU or Sigmoid
        activation_deriavative: Derivative Function of Activation Funciton
    '''
    def __init__(self, activation_function, activation_derivative) -> None:
        super().__init__()
        self.activation_function = activation_function
        self.activation_derivative = activation_derivative
        self.input = None
    def forward(self, x):
        self.input = x
        return self.activation_function(x)
    
    def backward(self, output_error):
        return np.multiply(output_error, self.activation_derivative(self.input))
            
    def __call__(self, x):
        return self.forward(x)
    
    
class NeuralNetwork():
    '''
    Multi Layer Perceptron

    Parameters:
        input_size: input size of data
        hidden_size: hidden layer size
        age_size: Age output size
        gender_size: Gender output size
        accent_size: Accent output size
        activation_function: ReLU or Sigmoid
        activation_derivative: Derivative of Activation Function
        hidden_layer_size: int
    '''
    def __init__(self, 
                 input_size: int, 
                 hidden_size: int, 
                 age_size: int, 
                 gender_size: int, 
                 accent_size: int, 
                 activation_function, 
                 activation_derivative, 
                 hidden_layer_size: int) -> None:
        
        self.hidden_layers = []
        for _ in range(hidden_layer_size):
            self.hidden_layers.append(FullyConnectedLayer(input_size, hidden_size))
            self.hidden_layers.append(ActivationLayer(activation_function, activation_derivative))
            input_size = hidden_size
            hidden_size //= 2

        self.age_fc = FullyConnectedLayer(input_size, age_size)
        self.age_af = ActivationLayer(softmax, softmax_derivative)

        self.gender_fc = FullyConnectedLayer(input_size, gender_size)
        self.gender_af = ActivationLayer(softmax, softmax_derivative)

        self.accent_fc = FullyConnectedLayer(input_size, accent_size)
        self.accent_af = ActivationLayer(softmax, softmax_derivative)

    def forward(self, x):
        for layer in self.hidden_layers:
            x = layer(x)

        age = self.age_fc(x)
        age = self.age_af(age)

        gender = self.gender_fc(x)
        gender = self.gender_af(gender)

        accent = self.accent_fc(x)
        accent = self.accent_af(accent)

        return age, gender, accent
    
    def backward(self, y_true, y_pred, learning_rate):
        age_true, gender_true, accent_true = y_true
        age_pred, gender_pred, accent_pred = y_pred

        age_error = nll_loss_derivative(age_true, age_pred)
        age_error = self.age_af.backward(age_error)
        age_error = self.age_fc.backward(age_error, learning_rate)

        gender_error = nll_loss_derivative(gender_true, gender_pred)
        gender_error = self.gender_af.backward(gender_error)
        gender_error = self.gender_fc.backward(gender_error, learning_rate)

        accent_error = nll_loss_derivative(accent_true, accent_pred)
        accent_error = self.accent_af.backward(accent_error)
        accent_error = self.accent_fc.backward(accent_error, learning_rate)

        for layer in reversed(self.hidden_layers):
            if isinstance(layer, FullyConnectedLayer):
                age_error = layer.backward(age_error, learning_rate)
                gender_error = layer.backward(gender_error, learning_rate)
                accent_error = layer.backward(accent_error, learning_rate)
            elif isinstance(layer, ActivationLayer):
                age_error = layer.backward(age_error)
                gender_error = layer.backward(gender_error)
                accent_error = layer.backward(accent_error)

        return age_error + gender_error + accent_error


    def train(self, train_loader, epochs, learning_rate, val_loader=None):
        for epoch in range(epochs):
            self._run_epoch(train_loader, epochs, learning_rate, epoch, "Train")

            if val_loader is not None:
                self._run_epoch(val_loader, epochs, learning_rate, epoch, "Validation")


    def _run_epoch(self, data_loader, epochs, learning_rate, epoch, mode="Train"):
        prefix = "Epoch" if mode == "Train" else "Val"
        epoch_loss = 0
        epoch_age_acc, epoch_gender_acc, epoch_accent_acc = 0, 0, 0

        # Use tqdm to create a progress bar
        for x, age_true, gender_true, accent_true in tqdm(data_loader, desc=f"{prefix} {epoch+1}/{epochs}"):
            x = x.reshape(x.shape[0], -1)
            age_pred, gender_pred, accent_pred = self.forward(x)

            age_accuracy = accuracy_score(hardmax(age_true), hardmax(age_pred))
            gender_accuracy = accuracy_score(hardmax(gender_true), hardmax(gender_pred))
            accent_accuracy = accuracy_score(hardmax(accent_true), hardmax(accent_pred))

            age_loss = negative_log_likelihood_loss(age_true, age_pred)
            gender_loss = negative_log_likelihood_loss(gender_true, gender_pred)
            accent_loss = negative_log_likelihood_loss(accent_true, accent_pred)

            loss = age_loss + gender_loss + accent_loss
            epoch_loss += loss

            epoch_age_acc += age_accuracy
            epoch_gender_acc += gender_accuracy
            epoch_accent_acc += accent_accuracy

            if mode == "Train":
                # Backward propagation only in training mode
                self.backward(y_true=(age_true, gender_true, accent_true),
                            y_pred=(age_pred, gender_pred, accent_pred),
                            learning_rate=learning_rate)

        epoch_loss /= len(data_loader)
        epoch_age_acc /= len(data_loader)
        epoch_gender_acc /= len(data_loader)
        epoch_accent_acc /= len(data_loader)

        logging.info(f'\n{prefix} Loss: {epoch_loss:.4f}, Age Accuracy: {epoch_age_acc:.4f}, Gender Accuracy: {epoch_gender_acc:.4f}, Accent Accuracy: {epoch_accent_acc:.4f}')
        logging.info('__________________________________________________________________________________')


    def test(self, test_loader):
        epoch_age_acc, epoch_gender_acc, epoch_accent_acc = 0, 0, 0

        # Use tqdm to create a progress bar
        for (x, age_true, gender_true, accent_true) in tqdm(test_loader, desc="Testing"):
            x = x.reshape(x.shape[0], -1)
            age_pred, gender_pred, accent_pred = self.forward(x)

            age_accuracy = accuracy_score(hardmax(age_true), hardmax(age_pred))
            gender_accuracy = accuracy_score(hardmax(gender_true), hardmax(gender_pred))
            accent_accuracy = accuracy_score(hardmax(accent_true), hardmax(accent_pred))

            epoch_age_acc += age_accuracy
            epoch_gender_acc += gender_accuracy
            epoch_accent_acc += accent_accuracy

        epoch_age_acc /= len(test_loader)
        epoch_gender_acc /= len(test_loader)
        epoch_accent_acc /= len(test_loader)

        logging.info(f'\Test Age Accuracy: {epoch_age_acc:.4f}, Gender Accuracy: {epoch_gender_acc:.4f}, Accent Accuracy: {epoch_accent_acc:.4f}')
        logging.info('__________________________________________________________________________________')

        return epoch_age_acc, epoch_gender_acc, epoch_accent_acc


# %%
def train_and_evaluate_model(model_name, hidden_layer_size, batch_size, activation_function, learning_rate):
    '''
    Train and Evaluate function for MLP model.
    Parameters:
        model_name: Model's name will to save results
        hidden_layer_Size: Size of Hidden Layers.
        batch_size: Batch Size
        activation_function, ReLU or Sigmoid activation Function
        learning_rate: Learning rate alpha to write learning rates.

    Returns:
        Dictionary that contains information about model, training and accuracies. 
    '''
    activation_functions = {'relu':relu, 'sigmoid':sigmoid}
    activation_derivatives = {'relu':relu_derivative, 'sigmoid':sigmoid_derivative}

    model = NeuralNetwork(input_size=30*77*4,
                          hidden_size=64,
                          age_size=len(train_df['age'].unique()),
                          gender_size=len(train_df['gender'].unique()),
                          accent_size=len(train_df['accent'].unique()),
                          activation_function=activation_functions[activation_function],
                          activation_derivative=activation_derivatives[activation_function],
                          hidden_layer_size=hidden_layer_size)

    train_loader = DataLoader(batch_size, train_dataset)
    val_loader = DataLoader(batch_size, val_dataset)

    model.train(train_loader=train_loader, 
                epochs=10, 
                learning_rate=learning_rate,
                val_loader=val_loader)

    test_loader = DataLoader(batch_size, test_dataset)  # Provide your evaluation data
    age_accuracy, gender_accuracy, accent_accuracy = model.test(test_loader)

    return {
        'Model': model_name,
        'Number of Hidden Layers': hidden_layer_size,
        'Batch Size': batch_size,
        'Activation Function': activation_function,
        'Learning Rate': learning_rate,
        'Age Accuracy': age_accuracy,
        'Gender Accuracy': gender_accuracy,
        'Accent Accuracy': accent_accuracy
    }

# %%
model_configurations = [
    ('MLP', 0, 16, 'relu', 1e-5),
    ('MLP', 0, 64, 'relu', 1e-5),
    ('MLP', 0, 16, 'sigmoid', 1e-5),
    ('MLP', 0, 64, 'sigmoid', 1e-5),
    ('MLP', 0, 16, 'relu', 1e-4),
    ('MLP', 0, 64, 'relu',  1e-4),
    ('MLP', 0, 16, 'sigmoid', 1e-4),
    ('MLP', 0, 64, 'sigmoid', 1e-4),
    
    ('MLP', 1, 16, 'relu', 1e-5),
    ('MLP', 1, 64, 'relu', 1e-5),
    ('MLP', 1, 16, 'sigmoid', 1e-5),
    ('MLP', 1, 64, 'sigmoid', 1e-5),
    ('MLP', 1, 16, 'relu', 1e-4),
    ('MLP', 1, 64, 'relu', 1e-4),
    ('MLP', 1, 16, 'sigmoid', 1e-4),
    ('MLP', 1, 64, 'sigmoid', 1e-4),

    ('MLP', 2, 16, 'relu', 1e-5),
    ('MLP', 2, 64, 'relu', 1e-5),
    ('MLP', 2, 16, 'sigmoid', 1e-5),
    ('MLP', 2, 64, 'sigmoid', 1e-5),
    ('MLP', 2, 16, 'relu', 1e-4),
    ('MLP', 2, 64, 'relu', 1e-4),
    ('MLP', 2, 16, 'sigmoid', 1e-4),
    ('MLP', 2, 64, 'sigmoid', 1e-4),
]
model_configurations = []

# %%
results_df = pd.DataFrame(columns=['Model', 'Number of Hidden Layers', 'Batch Size', 'Activation Function', 'Learning Rate', 'Age Accuracy', 'Gender Accuracy', 'Accent Accuracy'])
results_df = pd.read_excel('model_results.xlsx')

# %%
""" for model_name, hidden_layer_size, batch_size, activation_function, learning_rate in model_configurations:
    logging.info('////////////////////////////////////')
    results = train_and_evaluate_model(model_name, hidden_layer_size, batch_size, activation_function, learning_rate)
    results_df = pd.concat([results_df, pd.DataFrame([results], columns=results_df.columns)], ignore_index=True) """

# %%
results_df

# %%
results_df.to_excel('model_results.xlsx', index=False)

# %%
import torch
import torch.nn as nn
from torch.utils.data import Dataset, random_split, DataLoader
import torch.nn.functional as F
from torch.optim import Adam

# %%
class ImageDataset(Dataset):
    '''
        Dataset class to replicate Pytorch Datasets
        ____________________________________________

        Parameters
        ____________

        image_dir : str
            the directory contains images

        image_paths : list[str]
            the relative paths to image_dir

        age : np.array
            Age array

        gender : np.array
            Gender array

        accent : np.array
            Accent array

    '''
    def __init__(self, image_dir: str, 
                 image_paths: list[str], 
                 age: np.array, 
                 gender: np.array, 
                 accent: np.array):
        self.image_dir = image_dir
        self.image_paths = list(map(lambda p: os.path.join(image_dir, p), image_paths))
        self.age = torch.Tensor(age)
        self.gender = torch.Tensor(gender)
        self.accent = torch.Tensor(accent)

    def __getitem__(self, indices):
        if isinstance(indices, int):
            indices = slice(indices, indices+1)

        return torch.Tensor(read_images(self.image_paths[indices], ratio=5)), \
                self.age[indices], \
                self.gender[indices], \
                self.accent[indices]
    
    def __len__(self):
        return len(self.image_paths)
    

# %%
train_dataset = ImageDataset(image_dir='dataset_a3/voice_dataset/train/', 
                        image_paths=train_paths, 
                        age=train_age,
                        gender=train_gender,
                        accent=train_accent)

test_dataset = ImageDataset(image_dir='dataset_a3/voice_dataset/test/', 
                        image_paths=test_paths, 
                        age=test_age,
                        gender=test_gender,
                        accent=test_accent)

val_size = int(0.2 * len(train_dataset))
train_size = len(train_dataset) - val_size

# Use random_split to create training and validation datasets
train_dataset, val_dataset = random_split(train_dataset, [train_size, val_size])

# %%
train_loader_16_cnn = DataLoader(train_dataset, batch_size=16, shuffle=True)
test_loader_16_cnn = DataLoader(test_dataset, batch_size=16, shuffle=True) 

# %%
def hardmax(x):
    return torch.argmax(x, dim=1)

# %%
class CNN(nn.Module):

    def __init__(self,
                 hidden_cnn_layers,
                 age_classes,
                 gender_classes,
                 accent_classes,
                 activation_function) -> None:
        super().__init__()
        self.conv_layers = nn.ModuleList()
        in_channels = 4
        out_channels = in_channels*4
        for _ in range(hidden_cnn_layers):
            self.conv_layers.append(
                nn.Conv2d(in_channels=in_channels, 
                           out_channels=out_channels, 
                           kernel_size=5, 
                           stride=1, 
                           padding=2)
            )
            in_channels = out_channels
            out_channels *= 4
        # 39, 94
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        dummy_input = torch.randn(1, 4, 61, 155)
        self.fc1 = nn.Linear(self._get_conv_output_shape(dummy_input).shape[1], 512)
        self.age = nn.Linear(512, len(age_classes))
        self.gender = nn.Linear(512, len(gender_classes))
        self.accent = nn.Linear(512, len(accent_classes))
        assert activation_function == 'relu' or activation_function == 'sigmoid', 'Activation Layer must be relu or sigmoid.'
        self.af = F.relu if activation_function == 'relu' else F.sigmoid
    def _get_conv_output_shape(self, x):
        for conv_layer in self.conv_layers:
            x = conv_layer(x)
            x = self.pool(x)
        return x.view(x.size(0), -1)
    
    def forward(self, x):
        for conv in self.conv_layers:
            x = self.af(conv(x))
            x = self.pool(x)
        x = x.reshape(x.shape[0], -1)
        x = self.fc1(x)
        age = self.age(x)
        gender = self.gender(x)
        accent = self.accent(x)

        return age, gender, accent
    

    def train_model(self, train_loader, epochs, optimizer, criterion, val_loader=None):
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.to(device)
        for _ in range(epochs):
            self._run_epoch(train_loader, device, optimizer, criterion, "Train")

            if val_loader is not None:
                self._run_epoch(val_loader, device, optimizer, criterion, "Validation")


    def _run_epoch(self, data_loader, device, optimizer, criterion, mode="Train"):
        if mode == 'Validation':
            torch.set_grad_enabled(False)
            self.eval()
        else:
            torch.set_grad_enabled(True)
            self.train()
        prefix = "Epoch" if mode == "Train" else "Val"
        epoch_loss = 0
        epoch_age_acc, epoch_gender_acc, epoch_accent_acc = 0, 0, 0

        # Use tqdm to create a progress bar
        for x, age_true, gender_true, accent_true in tqdm(data_loader):
            x = torch.squeeze(x)
            x = x.permute(0, 3, 1, 2)
            x = x.to(device)

            age_true = age_true.to(device).squeeze()
            gender_true = gender_true.to(device).squeeze()
            accent_true = accent_true.to(device).squeeze()
            if mode == 'Train':
                # Zero the gradients
                optimizer.zero_grad()

            age_pred, gender_pred, accent_pred = self(x)

            # Compute the loss
            loss = criterion(age_pred, age_true) + criterion(gender_pred, gender_true) + criterion(accent_pred, accent_true)
            
            if mode == 'Train':
                # Backward pass
                loss.backward()

                # Update weights
                optimizer.step()

            # Assuming age_true, age_pred are PyTorch tensors
            age_true_numpy = hardmax(age_true).cpu().numpy()
            age_pred_numpy = hardmax(age_pred).cpu().numpy()

            # Now use accuracy_score with the NumPy arrays
            age_accuracy = accuracy_score(age_true_numpy, age_pred_numpy)

            # Repeat the same for gender and accent if needed
            gender_true_numpy = hardmax(gender_true).cpu().numpy()
            gender_pred_numpy = hardmax(gender_pred).cpu().numpy()
            gender_accuracy = accuracy_score(gender_true_numpy, gender_pred_numpy)

            accent_true_numpy = hardmax(accent_true).cpu().numpy()
            accent_pred_numpy = hardmax(accent_pred).cpu().numpy()
            
            accent_accuracy = accuracy_score(accent_true_numpy, accent_pred_numpy)
            epoch_loss += loss

            epoch_age_acc += age_accuracy
            epoch_gender_acc += gender_accuracy
            epoch_accent_acc += accent_accuracy

        epoch_loss /= len(data_loader)
        epoch_age_acc /= len(data_loader)
        epoch_gender_acc /= len(data_loader)
        epoch_accent_acc /= len(data_loader)

        logging.info(f'\n{prefix} Loss: {epoch_loss:.4f}, Age Accuracy: {epoch_age_acc:.4f}, Gender Accuracy: {epoch_gender_acc:.4f}, Accent Accuracy: {epoch_accent_acc:.4f}')
        logging.info('__________________________________________________________________________________')


    def test(self, test_loader):
        epoch_age_acc, epoch_gender_acc, epoch_accent_acc = 0, 0, 0
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.to(device)
        # Use tqdm to create a progress bar
        for x, age_true, gender_true, accent_true in tqdm(test_loader):
            x = torch.squeeze(x)
            x = x.permute(0, 3, 1, 2)
            x = x.to(device)

            age_true = age_true.to(device).squeeze()
            gender_true = gender_true.to(device).squeeze()
            accent_true = accent_true.to(device).squeeze()

            age_pred, gender_pred, accent_pred = self(x)

            # Assuming age_true, age_pred are PyTorch tensors
            age_true_numpy = hardmax(age_true).cpu().numpy()
            age_pred_numpy = hardmax(age_pred).cpu().numpy()

            # Now use accuracy_score with the NumPy arrays
            age_accuracy = accuracy_score(age_true_numpy, age_pred_numpy)

            # Repeat the same for gender and accent if needed
            gender_true_numpy = hardmax(gender_true).cpu().numpy()
            gender_pred_numpy = hardmax(gender_pred).cpu().numpy()
            gender_accuracy = accuracy_score(gender_true_numpy, gender_pred_numpy)

            accent_true_numpy = hardmax(accent_true).cpu().numpy()
            accent_pred_numpy = hardmax(accent_pred).cpu().numpy()
            
            accent_accuracy = accuracy_score(accent_true_numpy, accent_pred_numpy)

            epoch_age_acc += age_accuracy
            epoch_gender_acc += gender_accuracy
            epoch_accent_acc += accent_accuracy

        epoch_age_acc /= len(test_loader)
        epoch_gender_acc /= len(test_loader)
        epoch_accent_acc /= len(test_loader)

        logging.info(f'\Test Age Accuracy: {epoch_age_acc:.4f}, Gender Accuracy: {epoch_gender_acc:.4f}, Accent Accuracy: {epoch_accent_acc:.4f}')
        logging.info('__________________________________________________________________________________')

        return epoch_age_acc, epoch_gender_acc, epoch_accent_acc



# %%
# Function to train and evaluate the model
def train_and_evaluate_model_cnn(model_name, hidden_layer_size, batch_size, activation_function, learning_rate):
    '''
    Train and Evaluate function for CNN model.
    Parameters:
        model_name: Model's name will to save results
        hidden_layer_Size: Size of Hidden Layers.
        batch_size: Batch Size
        activation_function, ReLU or Sigmoid activation Function
        learning_rate: Learning rate alpha to write learning rates.

    Returns:
        Dictionary that contains information about model, training and accuracies. 
    '''

    torch.cuda.empty_cache()
    model = CNN(age_classes=train_df['age'].unique(),
                gender_classes=train_df['gender'].unique(),
                accent_classes=train_df['accent'].unique(),
                activation_function=activation_function,
                hidden_cnn_layers=hidden_layer_size)

    optimizer = Adam(model.parameters(), lr=1e-3)
    criterion = nn.CrossEntropyLoss()
    # Train the model for 5 epochs (you can adjust the number of epochs)
    train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size)
    val_loader = DataLoader(dataset=val_dataset, batch_size=batch_size)

    model.train_model(train_loader=train_loader, 
                epochs=10,
                criterion=criterion,
                optimizer=optimizer,
                val_loader=val_loader)
 
    # Evaluate accuracy for each category (Age, Gender, Accent)
    test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size)  # Provide your evaluation data
    age_accuracy, gender_accuracy, accent_accuracy = model.test(test_loader)

    # Return the results
    return {
        'Model': model_name,
        'Number of Hidden Layers': hidden_layer_size,
        'Batch Size': batch_size,
        'Activation Function': activation_function,
        'Learning Rate': learning_rate,
        'Age Accuracy': age_accuracy,
        'Gender Accuracy': gender_accuracy,
        'Accent Accuracy': accent_accuracy
    }

# %%
model_configurations = [    
    ('CNN', 1, 16, 'relu', 1e-3),
    ('CNN', 1, 64, 'relu', 1e-3),
    ('CNN', 1, 16, 'sigmoid', 1e-3),
    ('CNN', 1, 64, 'sigmoid', 1e-3),
    ('CNN', 1, 16, 'relu', 1e-4),
    ('CNN', 1, 64, 'relu', 1e-4),
    ('CNN', 1, 16, 'sigmoid', 1e-4),
    ('CNN', 1, 64, 'sigmoid', 1e-4),

    ('CNN', 2, 16, 'relu', 1e-3),
    ('CNN', 2, 64, 'relu', 1e-3),
    ('CNN', 2, 16, 'sigmoid', 1e-3),
    ('CNN', 2, 64, 'sigmoid', 1e-3),
    ('CNN', 2, 16, 'relu', 1e-4),
    ('CNN', 2, 64, 'relu', 1e-4),
    ('CNN', 2, 16, 'sigmoid', 1e-4),
    ('CNN', 2, 64, 'sigmoid', 1e-4),
]
# model_configurations = []

# %%
for model_name, hidden_layer_size, batch_size, activation_function, learning_rate in model_configurations:
    logging.info('////////////////////////////////////')
    results = train_and_evaluate_model_cnn(model_name, hidden_layer_size, batch_size, activation_function, learning_rate)
    results_df = pd.concat([results_df, pd.DataFrame([results], columns=results_df.columns)], ignore_index=True)

# %%
results_df

# %%
results_df.to_excel('model_results.xlsx', index=False)

# %% [markdown]
# ### Analyze
# 
# - We pretrained our models 10 epoch and most of them can still learn something, but for sake of our research lets assume these results are descent.
# 
# - When we look into the table there are somethings that are apparent.
# 
#     - Relu is more effective activation function for CNN than sigmoid is, but for MLP they seem to be showing equivalent performance.
# 
#     - CNN outperformed MLP for the most cases.
# 
#     - Gender looks like the easiest one to learn but as the data may be unevenly distributed we need to check it first. 

# %%
age_distribution = train_df['age'].value_counts()
gender_distribution = train_df['gender'].value_counts()
accent_distribution = train_df['accent'].value_counts()
print(str(age_distribution) + '\n-------\n' + str(gender_distribution) + '\n-------\n' + str(accent_distribution))

# %% [markdown]
# It seems ages are distributed evenly but gender is biased to male and accent is biased to us. So, we can think the models that do not represent good results are memorizing the distribution. Sometimes these results with models that are worse than rendom distribution.
# 
# ##### Analyzing Number of Hidden Layers

# %%
analyze_n0 = results_df[results_df['Number of Hidden Layers'] == 0].describe()
analyze_n1 = results_df[results_df['Number of Hidden Layers'] == 1].describe()
analyze_n2 = results_df[results_df['Number of Hidden Layers'] == 2].describe()

# %%
analyze_n0

# %%
analyze_n1

# %%
analyze_n2

# %% [markdown]
# - The tables above show us the affect of Number of Hidden Layers.
# 
# - 0 Layered model only was in MLP, but it performed well even equivalent or better to 1 Layer and 2 Layer models in average and min values.
# 
# - 2 Layered models performs slightly better than 1 layered ones. And it can perform well if the models are more trained.
# 
# - CNN models are outperformed but it seems in MLP part 0 layered model learned faster and more robust. Lets check 1 and 2 Layered CNNs and 

# %%
analyze_n1_MLP = results_df[(results_df['Number of Hidden Layers'] == 1) & (results_df['Model'] == 'MLP')].describe()
analyze_n2_MLP = results_df[(results_df['Number of Hidden Layers'] == 2) & (results_df['Model'] == 'MLP')].describe()

# %%
analyze_n1_MLP

# %%
analyze_n2_MLP

# %% [markdown]
# - on MLP context 1 Layer and 2 Layer shows mostly similar results.
# 
# - 0 Layered MLP outperformed these both with training 10 epoch.

# %%
analyze_n1_CNN = results_df[(results_df['Number of Hidden Layers'] == 1) & (results_df['Model'] == 'CNN')].describe()
analyze_n2_CNN = results_df[(results_df['Number of Hidden Layers'] == 2) & (results_df['Model'] == 'CNN')].describe()

# %%
analyze_n1_CNN

# %%
analyze_n2_CNN

# %% [markdown]
# - 2 layered CNN and 1 layered CNN shows similar outcome, but standard deviation of 2 Layered CNN is much more than 1 Layered have.
# 
# - This indicates, 2 layered CNN is more diverse than 1 Layered, it is because, it is more slow learner than 1 Layered yet more robust. 
# 
# - Some of models did not learn much in 10 epochs but some of models outperform 1 Layered CNN.

# %% [markdown]
# ##### Analyzing Learning Rates

# %%
analyze_lr_1e5 = results_df[results_df['Learning Rate'] == 1e-5].describe()
analyze_lr_1e4_MLP = results_df[(results_df['Learning Rate'] == 1e-4) & (results_df['Model'] == 'MLP')].describe()
analyze_lr_1e4_CNN = results_df[(results_df['Learning Rate'] == 1e-4) & (results_df['Model'] == 'CNN')].describe()
analyze_lr_1e3 = results_df[results_df['Learning Rate'] == 1e-3].describe()

# %%
analyze_lr_1e5

# %%
analyze_lr_1e4_MLP

# %% [markdown]
# - We can see on the table 10**(-4) learning rate performs better than 10**(-5) learning rate.
# 
# - The Standard deviation for 10**(-4) learning rate is higher.
# 
# - But, 10**(-4) learning rate results have outlier(s) on accent accuracy, and it affected standard deviation more.

# %%
analyze_lr_1e4_CNN

# %%
analyze_lr_1e3

# %% [markdown]
# - CNN's we tried 10**(-3) and 10**(-4) learning rates.
# 
# - 10**(-4) performs better than 10**(-3) in age category, almost equal in Gender category, and worse in Accent category. and overall they perform similar.
# 
# ##### Analyzing Batch Size

# %%
analyze_batch16_MLP = results_df[(results_df['Batch Size'] == 16) & (results_df['Model'] == 'MLP')].describe()
analyze_batch16_CNN = results_df[(results_df['Batch Size'] == 16) & (results_df['Model'] == 'CNN')].describe()
analyze_batch64_MLP = results_df[(results_df['Batch Size'] == 64) & (results_df['Model'] == 'MLP')].describe()
analyze_batch64_CNN = results_df[(results_df['Batch Size'] == 64) & (results_df['Model'] == 'CNN')].describe()

# %%
analyze_batch16_MLP

# %%
analyze_batch64_MLP

# %% [markdown]
# - There is a clear difference in terms of batch size that shows batch size 64 is better than 16 for MLP task, but as the number of epochs is limited and models are underfit, we only can say that batch size 64 learns faster than batch size 16.

# %%
analyze_batch16_CNN

# %%
analyze_batch64_CNN

# %% [markdown]
# - In CNN the effect of the difference of batch size is reduced but still batch size 64 learns faster.

# %% [markdown]
# ##### Analyzing Activation Function

# %%
analyze_relu_MLP = results_df[(results_df['Activation Function'] == 'relu') & (results_df['Model'] == 'MLP')].describe()
analyze_relu_CNN = results_df[(results_df['Activation Function'] == 'relu') & (results_df['Model'] == 'CNN')].describe()
analyze_sigmoid_MLP = results_df[(results_df['Activation Function'] == 'sigmoid') & (results_df['Model'] == 'MLP')].describe()
analyze_sigmoid_CNN = results_df[(results_df['Activation Function'] == 'sigmoid') & (results_df['Model'] == 'CNN')].describe()

# %%
analyze_relu_MLP

# %%
analyze_sigmoid_MLP

# %% [markdown]
# - In terms of Activation Function in MLP, there is no significant difference between relu and sigmoid.

# %%
analyze_relu_CNN

# %%
analyze_sigmoid_CNN

# %% [markdown]
# - In terms of Activation Function of CNN, Relu outperformed Sigmoid activation function clearly.
# 
# - The results can be changed when looking at higher nummber of layers and different learning rates.

# %% [markdown]
# ### Outcomes
# 
# - Number of Hidden Layers affect the models learning speed, it can be more robust increasing the number of layers if number of epochs are increased.
# 
# - Batch size also affects speed of learning, when increased it can show higher performance. 
# 
# - Learning Rate affects the learning of speed and if it is too small it can get into local minimum, but it also can be more robust.
# 
# - When choosing activation function, we can trust on Relu instead of Sigmoid. It can give more accurate results esspecially in CNN architectures.


