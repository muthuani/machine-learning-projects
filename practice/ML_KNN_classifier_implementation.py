#1. Write a python program to develop an ML model using KNN Classifier to predict the Species information for a given iris flower using Sepal Length, Sepal Width, Petal Length & Petal Width. Use the complete iris dataset for training.
# Use it to predict the species of an iris flower.
from sklearn.datasets import load_iris
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score,confusion_matrix

iris=load_iris()
#print(iris.feature_names)
x=iris.data
y=iris.target
#print(x.shape)
#print(y.shape)
knn=KNeighborsClassifier(3)
knn.fit(x,y)
#KNeighborsClassifier(n_neighbors=3)
p1=knn.predict(x)
print(p1)
print(y)

#2. Print the Accuracy Score and Confusion matrix for KNN Classifier using iris data.
# (Split iris dataset to train and test sets.)
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2)
#print(x_train.shape)
#print(x_test.shape)
#print(y_train.shape)
#print(y_test.shape)
knn2=KNeighborsClassifier(3)
knn2.fit(x_test,y_test)
p2=knn2.predict(x_test)
print(p2)
print(y_test)
ac_sc=accuracy_score(y_test,p2)
con_max=confusion_matrix(y_test,p2)
print("Accuracy Score :",ac_sc)
print("Confusion matrix :",con_max)
