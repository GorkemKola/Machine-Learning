# %% [markdown]
# # Part 1
# 
# ### Maximum Likelihood Estimate (MLE)
# 
# - You have a coin that you think is biased, you flip it 5 times and get the sequence HTHTH. What is the maximum likelihood estimate for the probability of getting heads?
#     
#     - As result shows 3 Heads and 2 Tails, our likelyhood of getting heads should be 3 / 5
# 
# - You know that candy prices are normally distributed with mean μ and standard deviation σ. You have three candy pricing 2, 3, 5 lira. What is the maximum likelihood for μ?
# 
#     - As we want to estimate likelihood for μ, we need to get mean of 2, 3, 5 and that is (2+3+5) / 3 = 10 /3 ~= 3.33
# 
# - Suppose that X is a discrete random variable with the following probability mass function: where 0 ≤ θ ≤ 1 is a parameter. The following 10 independent observations
# 
# ![Question 3 Table](imgs/q3.png)
# 
# were taken from such a distribution: (3, 0, 2, 1, 3, 2, 1, 0, 2, 1). What is the maximum likelihood estimate of θ
# 
# ![Question 4 L(θ)](imgs/q4_1.png)
# ![Question 4 Solution](imgs/q4_2.png)
# 
# The solution is like up above, but the problem is there roots contains 0 and 1, and if we use them in formula we get 0 on some values, so we can conclude they are not roots so root 0.5 is MLE(θ)
# 
# ### Naive Bayes
# 
# - The bank uses the information of its customers to give credits. The attributes used are sex, education, age , income and credit ( it can be yes or no).
# - Can a male 23-year-old university graduate working class customer get a credit?
# 
# | Ex | Sex     | Education  | Age   | Income       | Credit |
# |----|---------|------------|-------|--------------|--------|
# | 1  | female  | high school| 16-25 | poverty class| no     |
# | 2  | female  | none       | 16-25 | poverty class| no     |
# | 3  | female  | high school| 26-39 | upper class  | yes    |
# | 4  | male    | high school| 40-64 | poverty class| no     |
# | 5  | male    | university | 26-39 | upper class  | yes    |
# | 6  | female  | university | 16-25 | working class| no     |
# | 7  | male    | none       | 26-39 | working class| yes    |
# | 8  | male    | university | 40-64 | upper class  | yes    |
# | 9  | female  | university | 26-39 | working class| no     |
# | 10 | female  | high school| 40-64 | upper class  | yes    |
# 
# ![Naive Bayes Formula](imgs/naive1.png)
# ![Yes Likelihood](imgs/naive_yes.png)
# ![No Likelihood](imgs/naive_no.png)
# 
# We applied Laplace smoothing because there were 0s in likelihoods, we added 1 to numerator and unique class counts to denominator.
# 
# As 0.69 > 0.23, we can say that the answer is no
# 

# %% [markdown]
# # Part 2

# %%
'''
    Importing important libraries

    numpy: Array operations
    numba.jit: faster Array operations on numpy
    sklearn.model_selection.train_test_split: splitting into train and test
'''
import numpy as np
from sklearn.model_selection import train_test_split
from numba import jit

# %%
# Classes to use in visualization
classes = np.load('data/classes.npy')
classes

# %%
# reading All datas
data = np.load('data/train_data.npy')
labels = np.load('data/train_labels.npy')
mask = np.load('data/train_mask.npy')

# %%
data = data[mask > 0]
labels = labels[mask > 0]

# %%
# Splitting all datas into train and test
train_data, test_data, train_labels, test_labels, = train_test_split(data, labels, test_size=0.2, random_state=42)

# %%
train_rgb, train_infrared = train_data[:, :3], train_data[:, 3:]
test_rgb, test_infrared = test_data[:, :3], test_data[:, 3:]

# %%
# Train Shapes
train_data.shape, train_labels.shape, train_rgb.shape, train_infrared.shape

# %%
# Test Shapes
test_data.shape, test_labels.shape, test_rgb.shape, test_infrared.shape

# %% [markdown]
# train_data = np.load('data/train_data.npy')
# train_labels = np.load('data/train_labels.npy')
# train_mask = np.load('data/train_mask.npy')

# %%
@jit(nopython=True)
def _calculate_prob(x, mean, std):
    '''
        Definition
        __________
        Calculates Probability for a feature

        Formula
        __________
        exponent = e**(-((X - mean) ** 2 / ( 2 * STD ** 2)))
        1 / (SQRT(2 * PI) * STD) * exponent

        # it gets likelyhood from Gaussian Distribution
    '''
    exponent = np.exp(-((x - mean) ** 2) / (2 * std ** 2))
    return (1 / (np.sqrt(2 * np.pi) * std)) * exponent

@jit(nopython=True)
def calculate_probabilities(X, priors, parameters):
    '''
        This method returns likelyhoods of each class
    '''
    probabilities = np.zeros((X.shape[0], len(priors)))
    
    for i in range(len(priors)):
        prior = priors[i]
        params = parameters[i]
        prob = prior + np.sum(np.log(_calculate_prob(X, params[:, 0], params[:, 1])), axis=1)
        probabilities[:, i] = prob
    
    return probabilities
def fit(X, y):
    '''
        Definition
        ___________
        Fit function to get parameters and priors

        Parameters
        ___________
        X: np.array
            train data

        y: np.array
            train labels

    '''
    labels = np.unique(y)
    channels = X.shape[-1]
    priors = []
    parameters = np.zeros((len(labels), channels, 2))
    
    for i, c in enumerate(labels):
        X_c = X[y == c]
        prior = np.log(X_c.shape[0] / X.reshape(-1, channels).shape[0])
        priors.append(prior)
        
        for j, feature in enumerate(X_c.T):
            mean = np.mean(feature)
            std = np.std(feature)
            parameters[i, j] = mean, std
    
    return priors, parameters, channels
@jit(nopython=True)
def predict(X, priors, parameters):
    '''
        Definition
        ___________
        This function predict labels according to given test value

        Parameters
        ___________
        X : np.array
            Given Matrix to segment

        priors : List[int]
            prior knowledges

        parameters : np.array
            Mean and Standard deviation parameters for each label and features

        Returns
        ________
        Class probabilities
    '''
    return calculate_probabilities(X, priors, parameters)

def GNB(X_train, y_train, X_test):
    '''
        Definition
        ___________
        This function runs Gaussian Naive Bayes on given train and test data

        Parameters
        ___________
        X_train : np.array
            Training data

        y_train : np.array
            Training labels

        X_test : np.array
            Test data

        Returns
        ________
        predictions : np.array
            Predicted labels for test data
    '''
    priors, parameters, channels = fit(X_train, y_train)
    predictions = predict(X_test.reshape(-1, channels), priors, parameters)
    return predictions.reshape((*X_test.shape[:-1], predictions.shape[-1]))

# %% [markdown]
# ### Naive Bayes Applied to each layer

# %%
predictions = []
for i in range(train_data.shape[-1]):
    train_X = np.expand_dims(train_data[:, i], axis=-1)
    test_X = np.expand_dims(test_data[:, i], axis=-1)
    pred = GNB(train_X, train_labels, test_X)
    predictions.append(pred)

# %% [markdown]
# ### Naive Bayes applied to RGB channels

# %%
pred_rgb = GNB(train_rgb, train_labels, test_rgb)

# %% [markdown]
# ### Naive Bayes applied to Infrared channels

# %%
pred_infra = GNB(train_infrared, train_labels, test_infrared)

# %% [markdown]
# ### Naive Bayes applied to All channels

# %%
pred_all = GNB(train_data, train_labels, test_data)

# %% [markdown]
# ## Evaluate Models

# %%
def confusion_matrix(true, pred, n_class):

    cm = np.zeros((n_class, n_class))
    for i in range(n_class):
        for j in range(n_class):
            cm[i, j] = np.sum((true == i) & (pred == j))
    return cm

def accuracy(true, pred):
    cm = confusion_matrix(true, pred, 10)
    tp_tn = np.diag(cm)
    return np.sum(tp_tn) / np.sum(cm)

# %%
def calculate_accuracies(*preds, labels):
    for pred in preds:
        yield accuracy(np.argmax(pred, axis=1), labels)

# %%
texts = [
    'Accuracy of channel Red Channel',
    'Accuracy of channel Green Channel',
    'Accuracy of channel Blue Channel',
    'Accuracy of channel Infrared 1 Channel',
    'Accuracy of channel Infrared 2 Channel',
    'Accuracy of channel Infrared 3 Channel',
    'Accuracy of channel RGB Channels',
    'Accuracy of channel Infrared Channels',
    'Accuracy of all channels'
]

# %%
accuracies = calculate_accuracies(*predictions, pred_rgb, pred_infra, pred_all, labels=test_labels)
list(zip(texts, accuracies))

# %% [markdown]
# - The RGB channels and Infrared channels are getting worse results than they seperately get. The reason why probably they are highly correlated and extract similar features (mean, std). This made this models worse.
# - This Results indicates that, infrared channels is better than RGB channels, both together and seperately. But even though they are better than RGB channels when they get together they achieve more robust results. The reason why should be, RGB channels and Infrared channels are not correlated with each other and it would get better results as they extract different features.


