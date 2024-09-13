import numpy as np  # for matrix operations
import pandas as pd # for data loading
import seaborn as sns  # for plotting
sns.set_theme()

# the function that calculates the mean squared error
def mse(y_true, y_pred):
    return np.sum((y_true-y_pred)**2)/y_true.shape[0]


# class that implements the analytical solution of Linear Regression
class OLS:
    def fit(self, X, y):
        X_=X.copy()
        X_=np.concatenate((X, np.ones(shape=(X.shape[0], 1))), axis=1)
        self.coef_=(np.linalg.pinv(X_.T @ X_)) @ X_.T @ y

    def predict(self, X):
        X_=X.copy()
        X_=np.concatenate((X, np.ones(shape=(X.shape[0], 1))), axis=1)
        return X_ @ self.coef_
    

# class that implements the gradient descent solution for Linear Regression
class GradientDescent:
    def __init__(self, step_size=None, iterations=1000):
        self.step_size=step_size
        self.iterations=iterations
        self.coef_=None

    def gradient(self, X, y):
        return (X.T @ X @ self.coef_) - (X.T @ y)
    
    def fit(self, X, y, w_ml=None):
        X_=X.copy()
        X_=np.concatenate((X, np.ones(shape=(X.shape[0], 1))), axis=1)
        self.coef_=np.random.normal(scale=0.1, size=X_.shape[1])
        self.errors=[]
        self.actual_iterations=0
        calc_step_size=lambda x: self.step_size
        if self.step_size is None:
            calc_step_size=lambda x: 1/(1+x)
        if w_ml is None:
            w_ml=np.zeros(shape=self.coef_.shape) 
        for _ in range(self.iterations):
            grad=self.gradient(X_, y)
            if np.linalg.norm(grad)==0:
                break
            self.coef_-=calc_step_size(self.actual_iterations)*grad
            self.errors.append(np.linalg.norm(w_ml-self.coef_))
            self.actual_iterations+=1
        
    def predict(self, X):
        X_=X.copy()
        X_=np.concatenate((X, np.ones(shape=(X.shape[0], 1))), axis=1)
        return X_ @ self.coef_


# class thatt implements stochastic gradient descent for linear regression
class StochasticGradientDescent:
    def __init__(self, step_size=None, iterations=1000):
        self.step_size=step_size
        self.iterations=iterations
        self.coef_=None

    def gradient(self, X, y):
        return (X.T @ X @ self.coef_) - (X.T @ y)
    
    def fit(self, X, y, w_ml=None):
        X_=X.copy()
        X_=np.concatenate((X, np.ones(shape=(X.shape[0], 1))), axis=1)
        self.coef_=np.random.normal(scale=0.1, size=X_.shape[1])
        self.errors=[]
        self.actual_iterations=0
        calc_step_size=lambda x: self.step_size
        rng=np.random.default_rng(seed=3)
        converged=False
        result=self.coef_.copy()
        index=np.arange(stop=X.shape[0])
        if self.step_size is None:
            calc_step_size=lambda x: 1/(1+x) 
        if w_ml is None:
            w_ml=np.zeros(shape=self.coef_.shape) 
        while (not converged) and (self.actual_iterations<self.iterations):
            rng.shuffle(index)
            factor=100
            while ((factor-100)<index.shape[0]) and (self.actual_iterations<self.iterations):
                X_temp, y_temp=X_[index[factor-100:factor], :], y[index[factor-100:factor]]
                grad=self.gradient(X_temp, y_temp)
                if np.linalg.norm(grad)==0:
                    converged=True
                    break
                self.coef_-=calc_step_size(self.actual_iterations)*grad
                result+=self.coef_
                self.errors.append(np.linalg.norm(w_ml-self.coef_))
                self.actual_iterations+=1
                factor+=100
        
        self.coef_=result/self.actual_iterations
        
    def predict(self, X):
        X_=X.copy()
        X_=np.concatenate((X, np.ones(shape=(X.shape[0], 1))), axis=1)
        return X_ @ self.coef_


# class that implements ridge gradient descent method
class RidgeGD:
    def __init__(self, regu, step_size=None, iterations=1000):
        self.regu=regu
        self.step_size=step_size
        self.iterations=iterations
        self.coef_=None

    def gradient(self, X, y):
        penalty=self.regu*self.coef_
        return (X.T @ X @ self.coef_) - (X.T @ y) + penalty
    
    def fit(self, X, y, w_ml=None):
        X_=X.copy()
        X_=np.concatenate((X, np.ones(shape=(X.shape[0], 1))), axis=1)
        self.coef_=np.random.normal(scale=0.1, size=X_.shape[1])
        self.actual_iterations=0
        calc_step_size=lambda x: self.step_size
        if self.step_size is None:
            calc_step_size=lambda x: 1/(1+x) 
        if w_ml is None:
            w_ml=np.zeros(shape=self.coef_.shape)
        for _ in range(self.iterations):
            grad=self.gradient(X_, y)
            if np.linalg.norm(grad)==0:
                break
            self.coef_-=calc_step_size(self.actual_iterations)*grad
            self.actual_iterations+=1
        
    def predict(self, X):
        X_=X.copy()
        X_=np.concatenate((X, np.ones(shape=(X.shape[0], 1))), axis=1)
        return X_ @ self.coef_


# class that implements the kernel linear regression
class KernelRegression:
    def __init__(self, degree):
        self.degree=degree
        self.data=None
        self.alpha=None
    
    def  calc_kernel(self, X_new):
        return ((X_new @ self.data.T) + 1)**self.degree



    def fit(self, X, y):
        self.data=X
        self.alpha=np.linalg.pinv(self.calc_kernel(self.data)) @ y
    
    def predict(self, X):
        return self.calc_kernel(X) @ self.alpha
        



## function to generate splits in cross validation
def genrate_splits(n, test_size):
   rng=np.random.default_rng(seed=3)
   index=np.arange(n)
   sorted_index=index.copy()
   rng.shuffle(index)
   factor=int(n*test_size)+1
   iterations=0
   while (factor*iterations < index.shape[0]):
      test_index=(sorted_index>=iterations*factor) & (sorted_index<(iterations+1)*factor)
      train_index=~test_index
      yield train_index, test_index
      iterations+=1
   

## function to run cross validation on a model
def cross_validate(X, y, model, test_size):
   iterations=0
   errors=0
   for train_index, validation_index in genrate_splits(X.shape[0], test_size):
      X_train, X_valid, y_train, y_valid=X.loc[train_index, :], X.loc[validation_index, :], y[train_index], y[validation_index]
      model.fit(X_train, y_train)
      predictions=model.predict(X_valid)
      errors+=mse(y_valid, predictions)
      iterations+=1

   return errors/iterations


