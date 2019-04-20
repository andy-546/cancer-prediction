from sklearn.datasets import load_breast_cancer
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import mglearn
from sklearn.preprocessing import StandardScaler


cancer=load_breast_cancer()
scaler=StandardScaler()

X_train,X_test,y_train,y_test=train_test_split(cancer.data, cancer.target, random_state=0)



mlp= MLPClassifier(random_state=42)
mlp.fit(X_train,y_train)
print('Accuracy on the training set: {:.3f}'.format(mlp.score(X_train,y_train)))
print('Accuracy on the test set: {:.3f}'.format(mlp.score(X_test,y_test)))
##print('The maximum per each feature:\n{}'.format(cancer.data.max(axis=0)))
#the scale of different parameters are different. So we'll rescale them
X_train_scaled=scaler.fit(X_train).transform(X_train)
X_test_scaled=scaler.fit(X_test).transform(X_test)

mlp1 = MLPClassifier(max_iter=1000, random_state=42)
mlp1.fit(X_train_scaled,y_train)

print('Accuracy on the training set: {:.3f}'.format(mlp1.score(X_train_scaled,y_train)))
print('Accuracy on the test set: {:.3f}'.format(mlp1.score(X_test_scaled,y_test)))


#there exists discrepancy between performance of training subset and test subset.
#we can change type of solver, number of hidden layers, activation function
mlp1 = MLPClassifier(max_iter=1000, alpha=1, random_state=42)
mlp1.fit(X_train_scaled,y_train)
print('Accuracy on the training set: {:.3f}'.format(mlp1.score(X_train_scaled,y_train)))
print('Accuracy on the test set: {:.3f}'.format(mlp1.score(X_test_scaled,y_test)))
#the gap is decreased. we improved performance of model
plt.figure(figsize=(20,5))
plt.imshow(mlp1.coefs_[0], interpolation='None',cmap='GnBu')
plt.yticks(range(30),cancer.feature_names)
plt.xlabel('Columns in weight matrix')
plt.ylabel('input feature')
plt.colorbar()


