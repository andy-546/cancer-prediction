import pandas as pd
from sklearn.datasets import load_breast_cancer
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler

cancer=load_breast_cancer()

X_train,X_test,y_train,y_test=train_test_split(cancer.data, cancer.target, random_state=0)

svm= SVC()
svm.fit(X_train,y_train)
print('Accuracy on the training set: {:.3f}'.format(svm.score(X_train,y_train)))
print('Accuracy on the test set: {:.3f}'.format(svm.score(X_test,y_test)))

# the accuracy is bad due to scale as shown in the figure generated fromo below

plt.plot(X_train.min(axis=0), 'o', label='Min')
plt.plot(X_train.max(axis=0), 'v', label='Max')
plt.xlabel('Feature Index')
plt.ylabel('Feature Magnitude in Log Scale')
plt.yscale('log')
plt.legend(loc='upper right')
plt.show()

####################we will scale it
scaler=StandardScaler()
X_train_scaled=scaler.fit(X_train).transform(X_train)
X_test_scaled=scaler.fit(X_test).transform(X_test)

svm1= SVC()
svm1.fit(X_train_scaled,y_train)
print('Accuracy on the training set: {:.3f}'.format(svm1.score(X_train_scaled,y_train)))
print('Accuracy on the test set: {:.3f}'.format(svm1.score(X_test_scaled ,y_test)))

#####we can change parameters of SVC
#first we change C to 1000 to increase the complexity of model


svm2= SVC(C=1000)
svm2.fit(X_train_scaled,y_train)
print('Accuracy on the training set: {:.3f}'.format(svm2.score(X_train_scaled,y_train)))
print('Accuracy on the test set: {:.3f}'.format(svm2.score(X_test_scaled ,y_test)))

# this causes overfitting

