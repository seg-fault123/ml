import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC


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

## fitting and cross validating naive bayes
priors = np.linspace(0, 0.9, 50)
accuracies=[]
for prior in priors:
    accuracies.append(cross_validate(X_train_, y_train.values, NaiveBayes(prior), 0.25, random_seed=7))

plt.plot(priors, accuracies)
plt.xlabel('Prior')
plt.ylabel('Accuracy on Validation Data')
plt.title('Accuracy vs Prior for Naive Bayes')
plt.show()

# performace of naive bayes on training and testing data
best_proir=priors[np.argmax(accuracies)]
nb=NaiveBayes(prior=best_proir)
nb.fit(X_train_, y_train)
predictions=nb.predict(X_train_)
accuracy=(predictions==y_train).mean()*100
print('Naive Bayes')
print('Best Prior : %.3f'%best_proir)
print('Training Data Accuracy : %.3f'%accuracy)
predictions=nb.predict(X_test_)
accuracy=(predictions==y_test).mean()*100
print('Testing Data Accuracy : %.3f'%accuracy)
print('------------------------------------')
print()

# fitting and cross validating a logistic regression instance
rates = np.linspace(0.0001, 0.01, 20)
accuracies=[]
for rate in rates:
    accuracies.append(cross_validate(X_train_, y_train.values, LogReg(1000, rate), 0.25, random_seed=5))

plt.plot(rates, accuracies)
plt.xlabel('Step Size ($\eta$)')
plt.ylabel('Accuracy on Validation Data')
plt.title('Accuracy vs $\eta$ for Logistic Regression')
plt.show()

# performace of Logistic Regression on training and testing data
best_rate=rates[np.argmax(accuracies)]
log_reg=LogReg(1000, best_rate)
log_reg.fit(X_train_, y_train.values)
predictions=log_reg.predict(X_train_)
accuracy=(predictions==y_train).mean()*100
print('Logistic Regression')
print('Best Learning Rate : %.3f'%best_rate)
print('Training Data Accuracy : %.3f'%accuracy)
predictions=log_reg.predict(X_test_)
accuracy=(predictions==y_test).mean()*100
print('Testing Data Accuracy : %.3f'%accuracy)
print('------------------------------------')
print()


# fitting and cross validating svm
regus = np.linspace(1, 60, 20)
accuracies=[]
for regu in regus:
    accuracies.append(cross_validate(X_train_, y_train.values, SVC(kernel='rbf', C=regu), 0.25, random_seed=5))

plt.plot(regus, accuracies)
plt.xlabel('Reguralization Parameter')
plt.ylabel('Accuracy on Validation Data')
plt.title('Accuracy vs Regularization for SVM')
plt.show()

# performance of SVM
best_regu=regus[np.argmax(accuracies)]
svm=SVC(kernel='rbf', C=best_regu)
svm.fit(X_train_, y_train.values)
predictions=svm.predict(X_train_)
accuracy=(predictions==y_train).mean()*100
print('SVM')
print('Best Regularization Parameter : %.3f'%best_regu)
print('Training Data Accuracy : %.3f'%accuracy)
predictions=svm.predict(X_test_)
accuracy=(predictions==y_test).mean()*100
print('Testing Data Accuracy : %.3f'%accuracy)
print('------------------------------------')
print()



# fitting on the whole sampled dataset
X_=count_vec.fit_transform(X)
nb.fit(X_, y)
log_reg.fit(X_, y.values)
svm.fit(X_, y)

# printing the performance of each model on all the datasets
print('Naive Bayes')
result_nb=evaluate_model(nb, count_vec)
print('-------------------------------------')
print()

print('Logistic Regression')
result_lr=evaluate_model(log_reg, count_vec)
print('-------------------------------------')
print()

print('SVM')
result_svm=evaluate_model(svm, count_vec)
print('-------------------------------------')
print()


#visualizing the results
fig, axes=plt.subplots(ncols=2, sharey=True)
x=np.arange(18, step=3)
width=0.7

axes[0].bar(x-3*width/2, result_nb, width, align='edge', color='red', label='Naive Bayes', edgecolor='k')
axes[0].bar(x-width/2, result_lr, width, align='edge', color='green', label='Logistic Regression', edgecolor='k')
axes[0].bar(x+width/2, result_svm, width, align='edge', color='blue', label='SVM', edgecolor='k')
axes[0].legend(fontsize='large')

axes[1].plot(x-width, result_nb, marker='o', color='red', label='Naibe Bayes')
axes[1].plot(x, result_lr, marker='o', color='green', label='Logistic Regression')
axes[1].plot(x+width, result_svm, marker='o', color='blue', label='SVM')
axes[1].legend(fontsize='large')

axes[0].set_ylim(bottom=40, top=115)
axes[1].set_ylim(bottom=40, top=115)
axes[0].set_yticks(axes[0].get_yticks()[:-1])
axes[1].set_yticks(axes[0].get_yticks()[:-1])

axes[0].set_xticks(x)
axes[1].set_xticks(x)
axes[0].set_xticklabels(result_lr.index, fontsize='large')
axes[1].set_xticklabels(result_lr.index, fontsize='large')

axes[0].set_ylabel('Accuracy', fontsize='large')
axes[1].set_ylabel('Accuracy', fontsize='large')
axes[0].set_title('Naive Bayes vs Logistic Regression vs SVM', fontsize='large')
axes[1].set_title('Naive Bayes vs Logistic Regression vs SVM', fontsize='large')

fig.set_size_inches(20, 8)
fig.tight_layout()
plt.show()



## saving the learnt models 
joblib.dump(log_reg, 'log_reg.pkl')
joblib.dump(nb, 'nb.pkl')
joblib.dump(svm, 'svm.pkl')
joblib.dump(count_vec, 'count_vec.pkl')




