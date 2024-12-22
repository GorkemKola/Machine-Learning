# %% [markdown]
# # Assignment 1
# ## Part 1 - Theory Questions

# %% [markdown]
# ### k-Nearest Neighbor Classification
# Q1: Assume that you have a large training dataset. Specify a disadvantage of the k-Nearest Neighbor method when using it during testing. State also your reason about your answer. 
# - Ans: During Testing K-Nearest Neighbor method requires O(N) computational power, and that means when using a large scale dataset such as million or billion scaled, we need to compare the query with all the data in there and it is not efficient.
#   
# Q2: Considering the image below, state an optimal k-value depending on that the algorithm you are using is k-Nearest Neighbor. State also your reason behind the optimal value you preferred.
# 
# ![imgx](pictures/p1_1.png)
# 
# - Ans: The preffered optimal value is 12 according to the image, as validation error is minimum there, we can conclude that optimal k value is 12
# 
# ![Figure 1](pictures/p1_2.png)
# 
# Q3: Assume that you have the following training set of positive (+), negative (-) instances and a single test instance (o) in the image below (Figure 1). Assume also that the Euclidean metric is used for measuring the distance between instances. Finally consider that every nearest neighbor instance affects the final vote equally.
# 
# - What is the class appointed to the test instance for K=1? State also reason behind your answer.
#     - In figure 1 we can say that test instance's most clossest K point is -, so the test instance should be - as every nearest neighbor instance affects the final vote equally.
# - What is the class appointed to the test instance for K=3? State also reason behind your answer.
#     - In figure 1 we can say that test instance's most clossest K point is (-, -, +), so the test instance should be - as every nearest neighbor instance affects the final vote equally.
# - What is the class appointed to the test instance for K=5? State also reason behind your answer.
#     - In figure 1 we can say that test instance's most clossest K point is (-, -, +, +, +), so the test instance should be + as every nearest neighbor instance affects the final vote equally.
# 
# Q4: Fill the blanks with T (True) or F (False) for the statements below:
# 
# - instances of the data have the same scale then k-Nearest Neighbor’s performance increases drastically. (T)
# - While k-Nearest Neighbor performs well with a small number of input variables, it’s performance decreases when the number of inputs becomes large. (T)
# -  k-Nearest Neighbor makes supposes nothing about the functional form of the problem it handles. (T)
# 
# 
# 

# %% [markdown]
# ### Linear Regression
# 
# Q1:  Assume that you have five students have registered to a class and the class have a midterm and the final exam. You have obtained a set of their marks on two exams, which is in the table below:
# 
# ![linear1](pictures/p1_3.png)
# You plan to a model which form’s is fθ(x) = θ0 +θ1x1 +θ2x2 for fitting the data above. The x1 shows midterm exam score while x2 shows square of the midterm score. Besides you plan to use feature scaling (using divide operation by the ”max-min”, or range, of a feature) and mean normalization. What is the normalized value of the feature x(5)2?
# 
# - Ans: As x2 shows Square of Midterm score, x(5)2 should be 2025, but as there is a normalization part, and 2025 is the minimum value in Square of Midterm Scores, X(5)2 is now 0
# 
# Q2: Considering the figure below, which of the offsets used in linear regressions least square line fit? Assume that horizontal axis represents independent variable and vertical axis represents dependent variable. State your answer with your proper explanation.
# 
# ![linear2](pictures/p1_4.png)
# 
# - Ans: As we are searching for best weights and biases for the data we have in Linear regression, we try to minimize the loss of predicted values for the data we have and orignal values. And that means we are looking into vertical offset as we try to minimize it.
# 
# Q3: Considering the table below, consisting of four training examples:
# 
# ![linear3](pictures/p1_5.png)
# 
# Assume that you are trying to fit the data above to the linear regression model fθ(x) = θ0+θ1x1. Find the θ0 and θ1 values by using closed form solution (θ = (X^TX)**(-1) X^Ty) Also state dimension values of X, y, and θ matrices. Finally, show your calculations step by step.
# 
# - Ans: as X consists x0 (which is 1) and x1 values we can write X as;
# X = [[1, 1], [1, 2], [1, 4], [1, 0]]
# 
# and that means;
# X^T = [[1, 1, 1, 1], [1, 2, 4, 0]]
# 
# and that makes
# X^TX = [[1, 1, 1, 1], [1, 2, 4, 0]] @ [[1, 1], [1, 2], [1, 4], [1, 0]] = [[4, 7], [7, 21]]
# 
# As the formula below says;
# 
# ![linear4](pictures/p1_6.png)
# 
# inv(X^TX) = 1 / (4*21-7*7) * [[21, -7], [-7, 4]] => inv(X^TX) = 1 / (84-49) * [[21, -7], [-7, 4]] => 1 / 35 * [[21, -7], [-7, 4]] = [[3/5, -1/5], [-1/5, 4/35]]
# 
# inv(X^TX) = [[3/5, -1/5], [-1/5, 1/35]]
# now calculate the X^Ty;
# 
# X^Ty = [[1, 1, 1, 1], [1, 2, 4, 0]] @ [0.5, 1, 2, 0] = [[3.5], [10.5]]
# 
# and when we multiplicate this matrices
# 
# inv(X^TX)X^Ty = θ = [[3/5, -1/5], [-1/5, 4/35]] @ [[3.5], [10.5]] = [3/5*3.5-1/5*10.5, -1/5*3.5+4/35*10.5] = [0, 0.5]
# 
# θ = [0, 0.5]
# 
# Q4: State a valid reason for feature scaling and explain why it is a valid reason with respect to your reasoning.
# 
# - Ans: There are several reasons why feature scaling is important;
# 1) Every feature can be different so one of them can be too big and that makes other features contribute less even if they are important.
# 
# 2) It helps converge faster and finding global minimum faster.

# %% [markdown]
# # PART 2 - Anime Recommendation and Rating Prediction
# #### In this Part We need to make a anime recommendation and rating prediction system using anime data and user data

# %% [markdown]
# ### Installing Gdown to download files from gdrive

# %%
!pip install gdown

# %%
!gdown --id 1kCL8dZLHQUlBUzH-DmYHRNi7xMLrgS4b

# %%
# unzipping the folder , the contents are placed in Data->output->/kaggle/working 
# Currently each user is limited to 20GB data in kaggle 
! unzip Dataset_A1.zip
! rm Dataset_A1.zip

# %%
!pip install joblib numba

# %% [markdown]
# #### Importing some important libraries

# %%
import pandas as pd
from PIL import Image
import numpy as np

# %% [markdown]
# #### Reading files and creating Pandas Dataframes for animes and users train and test data

# %%
anime_df = pd.read_csv('animes.csv').dropna()
train_df = pd.read_csv('user_rates_train.csv').dropna()
test_df = pd.read_csv('user_rates_test.csv').dropna()

# %% [markdown]
# ### Preprocessing Anime Dataframe

# %%
# Replacing UNKNOWN to UnknownGenre
anime_df['Genres'] = anime_df['Genres'].replace('UNKNOWN', 'UnkownGenre').apply(lambda x: x.split(','))
anime_df

# %%
genres = set()
anime_df['Genres'].apply(lambda x: genres.update(x))

for genre in genres:
    anime_df[genre] = 0
    

for i, genre in enumerate(anime_df['Genres']):
    for g in genre:
        anime_df.loc[i, g] = 1 
    
anime_df.drop('Genres', axis=1, inplace=True)
anime_df

# %%
# Replace 'UNKNOWN' with 'UnknownType'
anime_df['Type'] = anime_df['Type'].replace('UNKNOWN', 'UnkownType')

# Get one-hot encoding of 'Type'
one_hot = pd.get_dummies(anime_df['Type']).astype(int)

# Drop column 'Type' as it is now encoded
anime_df.drop('Type', axis=1, inplace=True)
anime_df = anime_df.join(one_hot)

anime_df

# %%

# # As we can drop Studios column with the new update, I decided to drop it as it makes complexity much higher
# # Replace 'UNKNOWN' with 'UnknownStudio'
# anime_df['Studios'] = anime_df['Studios'].replace('UNKNOWN', 'UnkownStudios')

# # Get one-hot encoding of 'Studios'
# one_hot = pd.get_dummies(anime_df['Studios']).astype(int)

# # Drop column 'Studios' as it is now encoded
# anime_df.drop('Studios', axis=1, inplace=True)
# anime_df = anime_df.join(one_hot)

# anime_df

anime_df.drop('Studios', axis=1, inplace=True)
anime_df

# %%
# Replace 'UNKNOWN' with 'UnknownSource'
anime_df['Source'] = anime_df['Source'].replace('Music', 'MusicSource')

# Get one-hot encoding of 'Source'
one_hot = pd.get_dummies(anime_df['Source']).astype(int)

# Drop column 'Source' as it is now encoded
anime_df.drop('Source', axis=1, inplace=True)
anime_df = anime_df.join(one_hot)

anime_df

# %%
def handle_duration(x):
    '''
        This Function handles duration and turn it into integer format in mins.
    '''
    mins = 0
    if "hr" in x:
        x = x.split("hr")
        mins = int(x[0])*60
        x = x[1]
    
    return int(x.split()[0]) + mins if "min" in x else mins

# %%
anime_df['Duration'] = anime_df['Duration'].apply(handle_duration)
maximum = np.max(anime_df['Duration'])
mean = np.mean(anime_df['Duration'])
anime_df['Duration'].fillna(mean, inplace=True)
anime_df['Duration'] /= maximum
anime_df

# %%
anime_names = dict(zip(anime_df['anime_id'], anime_df['Name']))
train_user_names = dict(zip(train_df['user_id'], train_df['Username']))
test_user_names = dict(zip(test_df['user_id'], test_df['Username']))

recommend_X, recommend_y = anime_df.drop(['anime_id', 'Name', 'Image URL'], axis=1), anime_df['Name']

anime_df.drop(['Name', 'Image URL'], axis=1, inplace=True)
train_df.drop(['Username', 'Anime Title'], axis=1, inplace=True)
test_df.drop(['Username', 'Anime Title'], axis=1, inplace=True)

# %%
train_merged = train_df.merge(anime_df, on='anime_id')
test_merged = test_df.merge(anime_df, on='anime_id')
merged_df = pd.concat([train_merged, test_merged])

# %%
merged_df

# %%
user_item_matrix = merged_df.pivot_table(index='user_id', columns='anime_id', values='rating')
user_item_matrix.fillna(0, inplace=True)
user_item_matrix /= 10
user_item_matrix

# %%
anime_df.set_index('anime_id', inplace=True)

# %%
# Using Item Based Matrix we can recommend Animes without training.
# We show users as Anime Features and that makes we can understand what genre,type,studio,source users like.
item_based_matrix = pd.DataFrame(columns=anime_df.columns)
for user in user_item_matrix.index:
    row = user_item_matrix.loc[user]
    non_zeros = row[row != 0]
    idxs = non_zeros.index.tolist()
    mean = anime_df.loc[idxs].mul(non_zeros.loc[idxs], axis=0).mean()
    item_based_matrix.loc[user] = mean
item_based_matrix

# %% [markdown]
# ## Now we have User Based Data and Item Based Data so we can predict ratings using User Based Data and we can recommend animes using Item Based Data
# 
# #### Lets start with making KNN
# 
# I decided to use Jotlib to Parallelize CPU and make faster predictions, and numba for faster numpy calculations because when I tried on kaggle it was so slow, even 1 fold of cross validation lasted 1 hour and 54 minutes. So, it makes the calculation much and much faster.

# %%
from tqdm.notebook import tqdm
from joblib import Parallel, delayed
from numba import jit

@jit(nopython=True)
def dot(x1, x2):
    '''
        Paramaters
        _____________
        x1, x2 : np.array, np.array
            Numpy arrays to be multiplied
            
        Description
        ______________
        as Numba's implementation of np.dot doesn't work well on contigous arrays, this function helps dot product.
    '''
    x1 = np.ascontiguousarray(x1)
    x2 = np.ascontiguousarray(x2)
    return np.dot(x1, x2)

@jit(nopython=True)
def cosine_similarity(x1, x2):
    '''
        Paramaters
        _____________
        x1, x2 : np.array, np.array
            Numpy arrays that be looked similarities
            
        Description
        ______________
        Cosine Similarity Function
    '''
    dot_product = dot(x1, x2)
    norm_x1 = np.linalg.norm(x1)
    norm_x2 = np.linalg.norm(x2)
    similarity = dot_product / (norm_x1 * norm_x2)
    return similarity

@jit(nopython=True)
def k_nearest_indices(X, x, k):
    '''
        Paramaters
        _____________
        X : np.array
            This matrix is the searched matrix.

        x : np.array
            This vector is the query vector.
            
        k : int
            how many nearest indices are asked.
        Description
        ______________
        This function helps to find nearest indices of a query
    '''
    similarities = np.array([-cosine_similarity(x, x1) for x1 in X])
    sorted_indices = np.argsort(similarities)
    return sorted_indices[:k]

class KNN():
    '''
        Parameters
        ___
        
        k : int
            number of neighbours will be calculated
        
        metric : str | int
            distance metric that wanted be utilized.
            
        weighted : bool
            decides if KNN is weighted
            
        task : str
            declares the task (classification, regression, recommendation)
    '''
    
    def __init__(self,
                 k: int,
                 weighted : bool,
                 task : str,
                ) -> None:
        
        self.k = k
        self.weighted = weighted
        self.task = task
    def fit(self, X, y):
        '''
            Fits the X and y
            
            Parameters
            ___
            
            X : pd.DataFrame
                input dataframe
            y: pd.DataFrame
                labels
        '''
        self.X = X
        self.y = y
        
    def _predict(self, x):
        '''
            Description
            _______________
            
            Looks k nearest neighbors and predict the class
        '''
        nearest_indices = k_nearest_indices(self.X, x, self.k)
        nearest_similarities = np.array([cosine_similarity(x, self.X[idx]) for idx in nearest_indices])

        if self.weighted:
            labels = {l: np.sum(nearest_similarities[np.where(self.y[nearest_indices] == l)]) for l in np.unique(self.y[nearest_indices])}
        else:
            labels = {l: len(np.where(self.y[nearest_indices] == l)[0]) for l in np.unique(self.y[nearest_indices])}

        return max(labels, key=labels.get)

    def regression(self, x):
        '''
            Description
            ______
            
            Look k nearest neighbors and make a regression.
        '''
        nearest_indices = k_nearest_indices(self.X, x, self.k)
        nearest_labels = self.y[nearest_indices]

        if self.weighted:
            similarities = np.array([cosine_similarity(x, self.X[idx]) for idx in nearest_indices])
        else:
            similarities = np.ones(len(nearest_indices))

        summation = np.sum(similarities)
        prediction = np.sum(nearest_labels * similarities) / summation if summation != 0 else 0
        return prediction
    
    def recommend(self, x):
        '''
        Description
        ________________

        We recommend most similar y values related to input data x.
        '''
        nearest_indices = k_nearest_indices(self.X, x, self.k)
        nearest_labels = self.y[nearest_indices]
        return nearest_labels
    
    def predict(self, X):
        '''
            Description
            ______________
            
            Predict a query matrix.
        '''
        if self.task == 'classification':
            predicted_labels = Parallel(n_jobs=-1)(delayed(self._predict)(x) for x in tqdm(X, desc="Predicting"))
            return np.array(predicted_labels)
        elif self.task == 'regression':
            regressions = Parallel(n_jobs=-1)(delayed(self.regression)(x) for x in tqdm(X, desc="Predicting"))
            return np.array(regressions)
        elif self.task == 'recommendation':
            recommendations = Parallel(n_jobs=-1)(delayed(self.recommend)(x) for x in tqdm(X, desc="Predicting"))
            return recommendations
    '''staticmethod
    @jit(nopython=True)
    def cosine_similarity(self, x1, x2):
        dot_product = np.dot(x1, x2)
        norm_x1 = np.linalg.norm(x1)
        norm_x2 = np.linalg.norm(x2)
        similarity = dot_product / (norm_x1 * norm_x2)
        return similarity
    @staticmethod
    @jit(nopython=True)
    def k_nearest_indices(self, x):
        similarities = np.array([-self.cosine_similarity(x, x1) for x1 in self.X])
        sorted_indices = np.argsort(similarities)
        return sorted_indices[:self.k]

    '''

# %%
train_X, train_y = train_merged, train_merged["rating"]
test_X, test_y = test_merged, test_merged["rating"]

# %%
from sklearn.preprocessing import StandardScaler

# %%
merged_X = pd.concat([train_X, test_X])
scaler = StandardScaler().fit(merged_X)
train_X = scaler.transform(train_X)
test_X = scaler.transform(test_X)

# %%
def MAE(y_true, y_pred):
    '''
        Parameters
        ______________
        y_true, y_pred : np.array, np.array
            True and Predicted vectors will be looked at L1 distance.
            
        Description
        _______________
        Give a number that shows error, looking at L1 distance.
    '''
    diff = abs(y_true - y_pred)
    mae = np.mean(diff)
    return mae

def cv(k, model, X, y, shuffle=True, random_state = 42):
    '''
        Parameters
        _______________
        k : int
            how many folds will be done
            
        model
            model will give predictions
            
        X : np.array
            Feature matrix
        
        y : np.array
            Label vector
            
        shuffle : bool
            shuffles X, y if true
            
        random_state : int
            seed to give same random numbers each time.
    '''
    # Shuffle X and y
    if shuffle:
        np.random.seed(random_state)
        perm = np.random.permutation(len(X))
    X, y = X[perm], y[perm]
    errors = []
    # validate
    fold_size = len(X) // k
    for i in range(k):
        start = i * fold_size
        end = (i+1) * fold_size if i != k-1 else len(X)
        
        X_train = np.concatenate((X[:start], X[end:]))
        y_train = np.concatenate((y[:start], y[end:]))
        
        X_val = X[start:end]
        y_val = y[start:end]
        
        model.fit(X_train, y_train)
        y_pred = model.predict(X_val)
        
        mae = MAE(y_val, y_pred)
        
        errors.append(mae)
    return np.mean(errors)

# %%
'''errors = []
for k in [3, 5, 7]:
    knn = KNN(k=k, weighted=False, task="regression")
    error = cv(k=5, model=knn, X=train_X, y=train_y)
    errors.append(error)'''
errors = [0.20233095623987034, 0.24104943273905993, 0.2728988191711044]
errors

# %% [markdown]
# ![result](pictures/Screenshot%20from%202023-10-30%2011-03-57.png)

# %%
'''weighted_errors = []
for k in [3, 5, 7]:
    knn = KNN(k=k, weighted=True, task="regression")
    error = cv(k= 5, model=knn, X=train_X, y=train_y)
    weighted_errors.append(error)'''
weighted_errors = [0.2021547042864606, 0.2406848509179383, 0.27243754988611396]
weighted_errors

# %% [markdown]
# ![result2](pictures/Screenshot%20from%202023-10-30%2011-04-07.png)

# %%
# Optimal KNN
knn = KNN(k=[3, 5, 7][np.argmin(errors)], weighted=False, task='regression')
knn.fit(train_X, train_y)
predictions = knn.predict(test_X)
quantized_predictions = [round(p) for p in predictions]
MAE(test_y, predictions), MAE(test_y, quantized_predictions)

# %%
# Optimal Weighted KNN
wknn = KNN(k=[3, 5, 7][np.argmin(weighted_errors)], weighted=True, task='regression')
wknn.fit(train_X, train_y)
wpredictions = knn.predict(test_X)
quantized_wpredictions = [round(p) for p in wpredictions]
MAE(test_y, wpredictions), MAE(test_y, quantized_wpredictions)

# %%
# We can See that MAE of Weighted KNN and KNN is the same. 
# We can conclude probably they are same. So, lets check it.
all(predictions == wpredictions)

# %%
# As predictions are the same, quantized versions must be same
quantized_predictions == quantized_wpredictions

# %% [markdown]
# ## Evaluation Part

# %%
from sklearn import metrics

# %%
metrics.accuracy_score(test_y, quantized_wpredictions)

# %%
metrics.precision_score(test_y, quantized_predictions, average='micro'), metrics.precision_score(test_y, quantized_predictions, average='macro')

# %%
metrics.recall_score(test_y, quantized_predictions, average='micro'), metrics.recall_score(test_y, quantized_predictions, average='macro'), 

# %%
metrics.f1_score(test_y, quantized_predictions, average='micro'), metrics.f1_score(test_y, quantized_predictions, average='macro')

# %% [markdown]
# As we can see on the scores, all accuracy is %91.7 and micro precision, recall and f1 score are 0.917 are the same. As micro averaging gives the same weight each inctance, we can conclude that KNN model is shown high performance on instance level, but on the other hand, macro averaging gives range to (0.64 to 0.68) and it shows imbalance between classes. 

# %% [markdown]
# ## Recommendation Part

# %%
merged_users = pd.Series({**train_user_names, **test_user_names})
merged_users

# %%
def recommendation(model, query, k: int, weighted, data_X, data_y):
    scaler = StandardScaler().fit(data_X)
    data_X = scaler.transform(data_X)
    users = merged_users.loc[query.index]
    query = scaler.transform(query)
    model = model(k=k, weighted=weighted, task='recommendation')
    model.fit(data_X, data_y)
    
    return pd.Series(dict(zip(users, model.predict(query))))

# %%
recs = recommendation(KNN, item_based_matrix, 25, True, recommend_X, recommend_y)
recs

# %%
def get_recommendation(user):
    if isinstance(user, str):
        return recs[user]
    return recs[merged_users[user]]

# %%
# Sample Recommendation by id
get_recommendation(549)

# %%
# Sample Recommendation by username
get_recommendation('dotGif')

# %%



