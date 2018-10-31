from numpy import genfromtxt
from sklearn import svm, metrics
import numpy as np


x_train = genfromtxt('x_train_hog.csv')
y_train = genfromtxt('y_train_hu.csv')
x_test = genfromtxt('x_test_hog.csv')
y_test = genfromtxt('y_test_hu.csv')

#x_train = x_train*(10**12)
#x_test = x_test*(10**12)

print(x_test.shape, y_test.shape, x_train.shape, y_train.shape)

classifier = svm.SVC(gamma=0.001)#, kernel="linear")
classifier.fit(x_train, y_train)

y_predict = classifier.predict(x_test)
print("Classification report for classifier %s:\n%s\n" % (classifier, metrics.classification_report(y_test, y_predict)))
print("Classification report for AUC %s:\n" % (metrics.auc(y_test, y_predict)))