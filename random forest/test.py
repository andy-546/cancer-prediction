import pandas as pd
from sklearn.datasets import load_breast_cancer
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import numpy as np
cancer=load_breast_cancer()
#matplotlib inline
import mglearn
#raw_data=pd.read_csv('data.csv', delimiter=',')
X_train,X_test,y_train,y_test=train_test_split(cancer.data, cancer.target, random_state=0)

forest= RandomForestClassifier(n_estimators=100, random_state=0)
forest.fit(X_train,y_train)
print('Accuracy on the training set: {:.3f}'.format(forest.score(X_train,y_train)))
print('Accuracy on the test set: {:.3f}'.format(forest.score(X_test,y_test)))
#mglearn.plots.plot_knn_classification(n_neighbors=3)


n_features=cancer.data.shape[1]
plt.barh(range(n_features),forest.feature_importances_,align='center')
plt.yticks(np.arange(n_features),cancer.feature_names)
plt.xlabel('Feature Importance')
plt.ylabel('Feature')
plt.show()

