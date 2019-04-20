import pandas as pd
from sklearn.datasets import load_breast_cancer
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import mglearn
import graphviz
import numpy as np
from sklearn.tree import export_graphviz
cancer=load_breast_cancer()


X_train,X_test,y_train,y_test=train_test_split(cancer.data, cancer.target, stratify=cancer.target, random_state=42)

tree= DecisionTreeClassifier(random_state=0)
tree.fit(X_train,y_train)
print('Accuracy on the training set: {:.3f}'.format(tree.score(X_train,y_train)))
print('Accuracy on the test set: {:.3f}'.format(tree.score(X_test,y_test)))
#mglearn.plots.plot_knn_classification(n_neighbors=3)

#the classifier will divide the tree into many brnaches. so we will limit the number of conditions to 4

tree4= DecisionTreeClassifier(max_depth=4, random_state=0)
tree4.fit(X_train,y_train)
print('Accuracy on the training set: {:.3f}'.format(tree4.score(X_train,y_train)))
print('Accuracy on the test set: {:.3f}'.format(tree4.score(X_test,y_test)))

# by limiting the depth we decreased the overfittinng and inceased the accuracy of test subset
export_graphviz (tree4, out_file='cancertree.png',class_names=['malignant','benign'], feature_names=cancer.feature_names,impurity=False, filled=True)

#dot file can be converted to png online
n_features=cancer.data.shape[1]
plt.barh(range(n_features),tree.feature_importances_,align='center')
plt.yticks(np.arange(n_features),cancer.feature_names)
plt.xlabel('Feature Importance')
plt.ylabel('Feature')
plt.show()
