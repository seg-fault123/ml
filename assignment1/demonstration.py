import numpy as np  # for matrix operations
import pandas as pd # for data loading
import matplotlib.pyplot as plt  # for plotting
from models import *  ## for algorithms
import seaborn as sns  # for plotting
sns.set_theme()


# Loading the data
train_path=r'D:\msc\workspace\ml\assignment1\FMLA1Q1Data_train.csv'
test_path=r'D:\msc\workspace\ml\assignment1\FMLA1Q1Data_test.csv'
training_data=pd.read_csv(train_path, header=None)
training_data.head()
X, y=training_data.drop(labels=2, axis=1), training_data[2]

reg=OLS()
reg.fit(X, y)
w_ml=reg.coef_
print('OLS Solution:')
print('\tW1=%1.3f'%w_ml[0])
print('\tW2=%1.3f'%w_ml[1])
print('\tIntercept=%1.3f'%w_ml[2])
print()

gd=GradientDescent(step_size=0.0001, iterations=200)
gd.fit(X, y, w_ml=w_ml)
print('Gradient Descent Solution:')
print('\tW1=%1.3f'%gd.coef_[0])
print('\tW2=%1.3f'%gd.coef_[1])
print('\tIntercept=%1.3f'%gd.coef_[2])
print()

plt.plot(range(gd.actual_iterations), gd.errors)
plt.xlabel('Iteration')
plt.ylabel('|| w_ml - w_t ||')
plt.title('Error vs Iterations for Gradient Descent')
plt.show()

sgd=StochasticGradientDescent(step_size=None, iterations=10000)
sgd.fit(X, y, w_ml=w_ml)
print('Stochastic Gradient Descent Solution:')
print('\tW1=%1.3f'%sgd.coef_[0])
print('\tW2=%1.3f'%sgd.coef_[1])
print('\tIntercept=%1.3f'%sgd.coef_[2])
print()

plt.plot(range(sgd.actual_iterations)[:200], sgd.errors[:200])
plt.xlabel('Iteration')
plt.ylabel('|| w_ml - w_t ||')
plt.title('Error vs Iterations for Stochastic Gradient Descent')
plt.show()

# cross validating the alpha
alphas = np.arange(0, 40, 1)
errors=[]
for alpha in alphas:
    errors.append(cross_validate(X, y, RidgeGD(alpha, 0.0001, 1000), 0.2, 3))
best_alpha=alphas[np.argmin(errors)]
rgd=RidgeGD(regu=3, step_size=0.0001, iterations=1000)
rgd.fit(X, y)
print('Best Ridge Gradient Descent Solution:')
print('\tW1=%1.3f'%rgd.coef_[0])
print('\tW2=%1.3f'%rgd.coef_[1])
print('\tIntercept=%1.3f'%rgd.coef_[2])
print('\tRegularization Parameter Value=%d'%best_alpha)
print()

print()
plt.plot(alphas, errors)
plt.xlabel('Regularization Lambda')
plt.ylabel('MSE on Validation Data')
plt.title('MSE vs Lambda for Ridge Regression')
plt.show()

#loading test data and comparing performance
testing_data=pd.read_csv(test_path, header=None)
X_test, y_test=testing_data.drop(labels=2, axis=1), testing_data[2]
predictions_ml=reg.predict(X_test) #predictions according to OLS
predictions_ridge=rgd.predict(X_test) # predictions according to Ridge

print('Comparing Performance of OLS and Ridge:')
print('\tOLS Test Error=%1.3f'%mse(y_test, predictions_ml))
print('\tRidge Test Error=%1.3f'%mse(y_test, predictions_ridge))
print()

# plotting feature scatter plots with the target value
fig, axes=plt.subplots(1, 2)
axes[0].scatter(X[0], y)
axes[0].set_xlabel('Feature 1')
axes[0].set_ylabel('Output Variable')
axes[0].set_title('Output Variable vs Feature 1')

axes[1].scatter(X[1], y)
axes[1].set_xlabel('Feature 2')
axes[1].set_ylabel('Output Variable')
axes[1].set_title('Output Variable vs Feature 2')
fig.tight_layout()
plt.show()

# cross validating the degree in the kernel
degrees=np.array(range(1, 5))
errors=[]
for degree in degrees:
    errors.append(cross_validate(X, y, KernelRegression(degree), 0.2))
plt.plot(degrees, errors)
plt.xlabel('Degree')
plt.ylabel('MSE on Validation Data')
plt.title('MSE vs Degree for Kernel Regression')
plt.show()

kreg=KernelRegression(degree=2)
kreg.fit(X, y)
predictions=kreg.predict(X)
print('Performance of Kernel Regression degree=2')
print('\tTraining Error=%1.3f'%mse(y, predictions))
predictions=kreg.predict(X_test)
print('\tTesting Error=%1.3f'%mse(y_test, predictions))