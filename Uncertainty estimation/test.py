import pandas as pd
from sklearn.datasets import load_breast_cancer
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler

cancer=load_breast_cancer()

X_train,X_test,y_train,y_test=train_test_split(cancer.data, cancer.target, random_state=0)

####################we will scale it
scaler=StandardScaler()
X_train_scaled=scaler.fit(X_train).transform(X_train)
X_test_scaled=scaler.fit(X_test).transform(X_test)


svm= SVC(C=1000)
svm.fit(X_train_scaled,y_train)
print('Accuracy on the training set: {:.3f}'.format(svm.score(X_train_scaled,y_train)))
print('Accuracy on the test set: {:.3f}'.format(svm.score(X_test_scaled ,y_test)))

########decision function
print('The decision function is:\n\n{}'.format(svm.decision_function(X_test_scaled)[:20]))
print('Threshold decision function:\n\n{}'.format(svm.decision_function(X_test_scaled)[:20]>0))

########Predicting Probabilities(predict proba)

svm2= SVC(C=1000,probability=True)
svm2.fit(X_train_scaled,y_train)
print('Prediced probabilties for the samples (malignant and benign):\n\n{}'.format(svm2.predict_proba(X_test_scaled)[:20]))
svm2.predict(X_test_scaled)




