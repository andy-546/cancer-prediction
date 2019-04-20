import pandas as pd
from sklearn.datasets import load_iris
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

iris=load_iris()

X_train,X_test,y_train,y_test=train_test_split(iris.data, iris.target,  random_state=42)

gbrt= GradientBoostingClassifier()
gbrt.fit(X_train,y_train)
#####decision function
print('The decision function for the 3-class iris dataset:\n\n{}'.format(gbrt.decision_function(X_test[:10])))

#####predict probabilities
print('Prediced probabilties for the samples (malignant and benign):\n\n{}'.format(gbrt.predict_proba(X_test[:10])))
