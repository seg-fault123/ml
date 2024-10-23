# importing neccessary libraries
import numpy as np
import pandas as pd
import scipy.sparse as sparse


### Functions for loading and creating the dataset

def load_a():
    data=pd.read_csv('a.csv')
    data=data.rename(columns={'email': 'text'}).replace({'label': {'spam': 1, 'ham': 0}})
    return data

def load_b():
    data=pd.read_csv('b.csv')
    data=data.replace({'label': {'spam': 1, 'not spam': 0}})
    return data

def load_c():
    data=pd.read_csv('c.csv', usecols=['text', 'label'], na_values={'text': ['empty', np.nan]})
    data=data.dropna()
    data=data.loc[data['label'].isin(['1', '0']), :]
    data['label']=data['label'].astype(int)
    return data

def load_d():
    data=pd.read_csv('d.csv')
    data=data.replace({'label': {'spam': 1, 'ham': 0}})
    return data

def load_e():
    data=pd.read_csv('e.csv')
    return data

def load_f():
    data=pd.read_csv('f.csv')
    data=data.dropna().replace({'label': {'Spam': 1, 'Ham': 0}})
    return data


def sample_points(n, dataset, random_state=None):
    '''
    This function samples `n` data points each from class 1 and class 0 present in `dataset` 
    '''
    dataset=dataset.sample(frac=1.0, random_state=random_state) ## shuffle the rows of the dataset
    label_index=(dataset['label']==1) # boolean index of the rows with label 1
    label1=dataset.loc[label_index, :].iloc[:n, :]  # first n rows with label 1 of the shuffled dataset
    label0=dataset.loc[~label_index, :].iloc[:n, :] # first n rows with label 0 of the shuffled dataset
    return pd.concat((label1, label0), ignore_index=True) # combine both subsets for the final result


def combine_datasets(combinations, include_points=1000, random_state=None):
    '''
    This function combines datasets gathered from different sources.

    ### Parameters

    `combinations` : a list that specifies which datasets to combine. Example: [1,3,4] means Datasets 1, 3, and 4 have to be combined.
    
    `include_points` : specifies how many data points from each dataset and from each class have to be included to form the final dataset. 
     Example, a value of 500 means that 500 data points will be sampled from each class and from each dataset specified in `combinations`.
     If the dataset does not have enough data points as specified in any class, then all the data points are returned for that class.

    `random_state` : for reproducible results

    ### Returns

    pandas.Dataframe object with the sampled data points.

    '''
    loaders=[load_a, load_b, load_c, load_d, load_e, load_f]
    data=None
    for i in combinations:
        if data is None:
            data=sample_points(include_points, loaders[i-1](), random_state)
        else:
            subset=sample_points(include_points, loaders[i-1](), random_state)
            data=pd.concat((data, subset), ignore_index=True)
    return data
    


# class implementing Naive Bayes
class NaiveBayes:
    def __init__(self, prior=0):
        self.p=prior
        self.p1=None    # array that will contain the estimated parameters for class 1.
                        # ith value of the array give P(xi=1 | y=1), probablity that ith feature is 1 given that label is 1
        
        self.p0=None    # array that will contain the estimated parameters for class 0.
                        # ith value of the array give P(xi=1 | y=0), probablity that ith feature is 1 given that label is 0
    
    def estimate_params(self, X, y):
        '''
        Function that estimates the parameters for both classes. Returns 2 arrays (p0, p1). Used to estimate the parameters for both classes
        
        ### Parameters

        `X` : a sparse matrix (scipy.sparse.csr_matrix) with data points as rows and features as columns

        `y` : a numpy 1-D array with labels 1 or 0.

        ### Returns

        a tuple of numpy 1-D arrays (p0, p1), where p0 is in same format as self.p0 and p1 is in the same format as self.p1.
        '''
        subset_index=(y==0)  # boolean index of rows that have label 0
        return np.array((X[subset_index, :]>0).mean(axis=0))[0], np.array((X[~subset_index, :]>0).mean(axis=0))[0]

    def add_smoothing(self, X, y):
        '''
        This function adds smoothing datapoints to the dataset passed.

        ### Parameters
        
        `X` : a sparse matrix (scipy.sparse.csr_matrix) with data points as rows and features as columns

        `y` : a numpy 1-D array with labels 1 or 0.

        ### Returns

         a tuple `(new_data_matrix, new_labels)`, where `new_data_matrix` is a scipy.sparse.csr_matrix object with smoothing rows added and 
         `new_labels` is a numpy 1-D array with labels 1 and 0 having the smoothing labels added.
        '''
        n_features=X.shape[1] 
        empty=sparse.csr_matrix(0, shape=(1, n_features)) # corresponding to empty mail
        full=sparse.csr_matrix(1, shape=(1, n_features)) # corresponding to mails with all words
        new_data_matrix=sparse.vstack((X, empty, empty.copy(), full, full.copy())) # add to the data matrix
        new_labels=np.concatenate((y, [0, 1, 0, 1]))  # add to the label vector
        return new_data_matrix, new_labels 

    def fit(self, X, y):
        '''
        This functions fits the model by adding smoothing and estimating the required parameters

        ### Parameters
        
        `X` : a sparse matrix (scipy.sparse.csr_matrix) with data points as rows and features as columns

        `y` : a numpy 1-D array with labels 1 or 0.
        '''
        X_new, y_new=self.add_smoothing(X, y) ## add smoothing before fitting
        if self.p==0: # if prior is not specified
            self.p=(y_new==1).mean() # estimate the prior
        self.p0, self.p1=self.estimate_params(X_new, y_new) # estimate the parameters for both the classes
        self.p0[self.p0==0]=1e-6   # if somehow the estimations contain 0, replace them with small value 
        self.p1[self.p1==0]=1e-6
        return self
    
    def predict(self, X):
        '''
        This functions predicts the labels for the data points in `X`

        ### Parameters
        
        `X` : a sparse matrix (scipy.sparse.csr_matrix) with data points as rows and features as columns

        ### Returns

        a numpy 1-D array with corresponding predicted labels 0 or 1.
        '''
        X_new=(X>0).astype(int) # convert the dataset such that if for a sample, a feature is present then the value of that feature is 1
                                # otherwise 0.
        predictions= X_new @ np.log( (self.p1 * (1-self.p0))  /  (self.p0 * (1-self.p1)) )  # apply the definition of the decision boundary
        predictions+= np.sum( np.log( (1-self.p1) / (1-self.p0) ) ) + np.log( self.p/(1-self.p) )
        return (predictions>=0).astype(int)
    


## class that implements Logistic Regression for sparse matrices with gradient descent
class LogReg:
    def __init__(self, iterations, step_size):
        self.iterations=iterations 
        self.step_size=step_size
        self.coef_=None  # array that will contain the weights of features used in the predictions.
                         # the ith element of the array gives the weight of the ith feature
    
    def sigmoid(self, z):
        '''
        This function calculates the sigmoid of each element of `z`.

        ### Parameters

        `z` : a numpy 1-D array having float values

        ### Returns

        a numpy 1-D array having elements as sigmoid of corresponding elements of `z`.
        '''
        return 1/(1+ np.exp(-z))

    def loss(self, X, y):
        '''
        This function returns the cross-entropy loss of the dataset with the learnt weights.

        ### Parameters

        `X` : a sparse matrix (scipy.sparse.csr_matrix) with data points as rows and features as columns

        `y` : a numpy 1-D array with labels 1 or 0.

        ### Returns

        The cross-entropy loss (float) of the dataset with the learnt weights.
        '''
        probs= self.sigmoid(X @ self.coef_)
        return -(y*np.log(probs) + (1-y)*np.log(1-probs)).mean()

    def gradient(self, X, y):
        '''
        Calculates the gradient vector of the dataset with the learnt weights

        ### Parameters
        
        `X` : a sparse matrix (scipy.sparse.csr_matrix) with data points as rows and features as columns

        `y` : a numpy 1-D array with labels 1 or 0.

        ### Returns

        a numpy 1-D array which is the gradient vector.
        '''
        y_hat=self.sigmoid(X @ self.coef_)
        differences=y_hat-y
        return np.array((sparse.diags(differences) @ X).sum(axis=0))[0]
    
    def gradient_descent(self, X, y):
        '''
        This function runs the gradient descent algorithm on the dataset with inital weights as 0.

        ### Parameters
        
        `X` : a sparse matrix (scipy.sparse.csr_matrix) with data points as rows and features as columns

        `y` : a numpy 1-D array with labels 1 or 0.
        '''
        self.coef_=np.zeros(X.shape[1])
        for i in range(self.iterations):
            grad=self.gradient(X, y)
            self.coef_-=self.step_size*grad

    def normalize(self, X):
        '''
        This function normalizes each column of the matrix X. The norm of each column of the resultant matrix is striclty less than 1.

        ### Parameters
        
        `X` : a sparse matrix (scipy.sparse.csr_matrix) with data points as rows and features as columns

        ### Returns

        a scipy.sparse.csr_matrix with each column of the matrix normalized.
        '''
        return (X @ sparse.diags((np.array( X.power(2).sum(axis=0) )[0]+1)**-0.5))

    def fit(self, X, y):
        '''
        This function fits the model on the passed dataset.
        
        ### Parameters
        
        `X` : a sparse matrix (scipy.csr_matrix) with data points as rows and features as columns

        `y` : a numpy 1-D array with labels 1 or 0.
        '''
        self.gradient_descent(self.normalize(X), y)
    
    def predict(self, X):
        '''
        This functions predicts the labels for the data points in `X`

        ### Parameters
        
        `X` : a sparse matrix (scipy.sparse.csr_matrix) with data points as rows and features as columns

        ### Returns

        a numpy 1-D array with corresponding predicted labels 0 or 1.
        '''
        probs=self.normalize(X) @ self.coef_
        return (probs>=0).astype(int)        
        


def evaluate_model(model, vectorizer):
    '''
    This function prints the performance of the passed model on all datasets. All samples from all datasets are used for performance
    evaluation
    
    ### Parameters
        
    `model` : a fitted classification model that supports the `predit` method.

    `vectorizer` : an sklearn.feature_extraction.text.CountVectorizer instance that can extract text features from the the raw emails

    ### Returns

    a pandas.Series object with the dataset number as index and accuracy as the value
    '''
    loaders=[load_a, load_b, load_c, load_d, load_e, load_f]
    result={}
    for i in range(len(loaders)):
        dataset=loaders[i]()
        X, y = vectorizer.transform(dataset['text']), dataset['label']
        predictions=model.predict(X)
        accuracy=(predictions==y).mean()*100
        result['Dataset %d'%(i+1)]=accuracy
        print('\tAccuracy on Dataset %d : %.3f'%(i+1, accuracy))
        print('\tSpam %% : %.3f'%(y.mean()*100))
        print('\tSize : %d'%X.shape[0])
        print('-------')
    return pd.Series(result)


def genrate_splits(n, test_size, random_seed=None):
    '''
    This function yields split indices for training and validation sets of data base on the size of the dataset,
    and the proportion of points that go into the validation set. Can be used for cross validation.

    ### Parameters

    `n` : the size of the dataset (total points)

    `test_size` : the proportion of points that go into the validation subset.

    `random_state` : for reproducible results

    ### Returns

    an iterator that yields a tuple of numpy 1-D arrays `(train, test)` where `train` is the integer indices
    of the training subset, and `test` is the integer indices of validation subset.
    '''
    rng=np.random.default_rng(seed=random_seed)
    index=np.arange(n)
    sorted_index=index.copy()
    rng.shuffle(index)
    factor=int(n*test_size)+1
    iterations=0
    while (factor*iterations < index.shape[0]):
        test_index=(sorted_index>=iterations*factor) & (sorted_index<(iterations+1)*factor)
        train_index=~test_index
        yield index[train_index], index[test_index]
        iterations+=1
   


def cross_validate(X, y, model, test_size, random_seed=None):
    '''
    This function runs the cross validation algorithm for a given model and dataset.

    ### Parameters

    `X` : a sparse matrix (scipy.csr_matrix) with data points as rows and features as columns

    `y` : a numpy 1-D array with labels 1 or 0.

    `model` : a classification model that supports the `fit` and `predit` method.

    `test_size` : the proportion of data points in a sigle fold of K-Fold cross validation

    `random_state` : for reproducible results 

    ### Returns

    average of the accuracies on the validation set of each fold. 
    '''
    iterations=0
    accuracy=0
    for train_index, validation_index in genrate_splits(X.shape[0], test_size, random_seed):
        X_train, X_valid, y_train, y_valid=X[train_index, :], X[validation_index, :], y[train_index], y[validation_index]
        model.fit(X_train, y_train)
        predictions=model.predict(X_valid)
        accuracy+=(predictions==y_valid).mean()*100
        iterations+=1

    return accuracy/iterations