import pandas as pd
from sklearn.datasets import load_breast_cancer
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

cancer=load_breast_cancer()
#matplotlib inline
import mglearn
#raw_data=pd.read_csv('data.csv', delimiter=',')
X_train,X_test,y_train,y_test=train_test_split(cancer.data, cancer.target, stratify=cancer.target, random_state=42)

knn= KNeighborsClassifier()
knn.fit(X_train,y_train)
print('Accuracy of KNN on n-5, on the training set: {:.3f}'.format(knn.score(X_train,y_train)))
print('Accuracy of KNN on n-5, on the test set: {:.3f}'.format(knn.score(X_test,y_test)))
#mglearn.plots.plot_knn_classification(n_neighbors=3)


X_train,X_test,y_train,y_test=train_test_split(cancer.data, cancer.target, stratify=cancer.target, random_state=66)

training_accuracy=[]
testing_accuracy=[]
neighbors_settings=range(1,11)
for n_neighbors in neighbors_settings:
    clf=KNeighborsClassifier(n_neighbors=n_neighbors)
    clf.fit(X_train,y_train)
    training_accuracy.append(clf.score(X_train,y_train))
    testing_accuracy.append(clf.score(X_test,y_test))

plt.plot(neighbors_settings, training_accuracy,label='Accuracy of training set')
plt.plot(neighbors_settings, testing_accuracy,label='Accuracy of test set')
plt.ylabel('Accuracy')
plt.xlabel('Number of neighbors')
#plt.legend()
plt.show()
