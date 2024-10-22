import numpy as np
import pandas as pd
import scipy.sparse as sparse
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split

from imblearn.under_sampling import RandomUnderSampler

from models_helpers import *
import joblib

sns.set_theme()

## loading the data and splitting into training and testing subsets
data=combine_datasets([1, 2, 4, 5], include_points=500, random_state=29)
X, y=data['text'], data['label']
X_train, X_test, y_train, y_test=train_test_split(X, y, test_size=0.25, random_state=0)


## fitting our vectorizer that will extract the features from the dataset
count_vec=CountVectorizer(stop_words='english')
count_vec.fit(X_train, y_train)

## extracting the features and storing them in appropriate variables
X_train_, X_test_=count_vec.transform(X_train), count_vec.transform(X_test)

## fitting a naive bayes instance
nb=NaiveBayes()
nb.fit(X_train_, y_train)

# performace of naive bayes on training and testing data
predictions=nb.predict(X_train_)
accuracy=(predictions==y_train).mean()*100
print('Naive Bayes')
print('Training Data Accuracy : %.3f'%accuracy)
predictions=nb.predict(X_test_)
accuracy=(predictions==y_test).mean()*100
print('Testing Data Accuracy : %.3f'%accuracy)
print('------------------------------------')
print()

# fitting a logistic regression instance
log_reg=LogReg(1000, 0.001)
log_reg.fit(X_train_, y_train.values)

# performace of Logistic Regression on training and testing data
predictions=log_reg.predict(X_train_)
accuracy=(predictions==y_train).mean()*100
print('Logistic Regression')
print('Training Data Accuracy : %.3f'%accuracy)
predictions=log_reg.predict(X_test_)
accuracy=(predictions==y_test).mean()*100
print('Testing Data Accuracy : %.3f'%accuracy)
print('------------------------------------')
print()


# fitting on the whole sampled dataset
X_=count_vec.fit_transform(X)
nb.fit(X_, y)
log_reg.fit(X_, y.values)

# printing the performance of each model on all the datasets
print('Naive Bayes')
result_nb=evaluate_model(nb, count_vec)
print('-------------------------------------')
print()

print('Logistic Regression')
result_lr=evaluate_model(log_reg, count_vec)
print('-------------------------------------')
print()


#visualizing the results
ax=plt.axes()
x=np.arange(6)
width=0.25

ax.bar(x-width, result_nb, width, align='edge', color='red', label='Naive Bayes', alpha=0.7, edgecolor='k')
ax.bar(x, result_lr, width, align='edge', color='Green', label='Logistic Regression', edgecolor='k')

ax.plot(x-width/2, result_nb, marker='o', color='red', )
ax.plot(x+width/2, result_lr, marker='o', color='green',)
ax.legend()

ax.set_ylim(bottom=40, top=110)
ax.set_yticks(ax.get_yticks()[:-1])

ax.set_xticks(x)
ax.set_xticklabels(result_lr.index)

ax.set_ylabel('Accuracy')
ax.set_title('Naive Bayes   vs   Logistic Regression')

plt.show()


## saving the learnt models 
joblib.dump(log_reg, 'log_reg.pkl')
joblib.dump(nb, 'nb.pkl')
joblib.dump(count_vec, 'count_vec.pkl')




